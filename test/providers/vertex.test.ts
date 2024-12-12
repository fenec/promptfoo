import type { JSONClient } from 'google-auth-library/build/src/auth/googleauth';
import { getCache, isCacheEnabled } from '../../src/cache';
import { VertexChatProvider } from '../../src/providers/vertex';
import * as vertexUtil from '../../src/providers/vertexUtil';
import type { GeminiFormat, OpenAIMessage } from '../../src/providers/vertexUtil';
import { REQUEST_TIMEOUT_MS } from '../../src/providers/shared';

jest.mock('../../src/cache', () => ({
  getCache: jest.fn().mockReturnValue({
    get: jest.fn(),
    set: jest.fn(),
  }),
  isCacheEnabled: jest.fn(),
}));

jest.mock('../../src/providers/vertexUtil', () => ({
  ...jest.requireActual('../../src/providers/vertexUtil'),
  getGoogleClient: jest.fn(),
}));

jest.mock('../../src/logger');

describe('VertexChatProvider.callGeminiApi', () => {
  let provider: VertexChatProvider;

  beforeEach(() => {
    provider = new VertexChatProvider('gemini-pro', {
      config: {
        context: 'test-context',
        examples: [{ input: 'example input', output: 'example output' }],
        stopSequence: ['\n'],
        temperature: 0.7,
        maxOutputTokens: 100,
        topP: 0.9,
        topK: 40,
      },
    });
    jest.mocked(getCache).mockReturnValue({
      get: jest.fn(),
      set: jest.fn(),
      wrap: jest.fn(),
      del: jest.fn(),
      reset: jest.fn(),
      store: {} as any,
    });

    jest.mocked(isCacheEnabled).mockReturnValue(true);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should call the Gemini API and return the response', async () => {
    const mockResponse = {
      data: [
        {
          candidates: [{ content: { parts: [{ text: 'response text' }] } }],
          usageMetadata: {
            totalTokenCount: 10,
            promptTokenCount: 5,
            candidatesTokenCount: 5,
          },
        },
      ],
    };

    const mockRequest = jest.fn().mockResolvedValue(mockResponse);

    jest.spyOn(vertexUtil, 'getGoogleClient').mockResolvedValue({
      client: {
        request: mockRequest,
      } as unknown as JSONClient,
      projectId: 'test-project-id',
    });

    const response = await provider.callGeminiApi('test prompt');

    expect(response).toEqual({
      cached: false,
      output: 'response text',
      tokenUsage: {
        total: 10,
        prompt: 5,
        completion: 5,
      },
    });

    expect(vertexUtil.getGoogleClient).toHaveBeenCalledWith();
    expect(mockRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        url: expect.stringContaining('streamGenerateContent'),
        method: 'POST',
        data: {
          contents: [
            {
              role: 'user',
              parts: [{ text: 'test prompt' }],
            },
          ],
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 100,
            topP: 0.9,
            topK: 40,
            candidateCount: undefined,
            stopSequences: undefined,
            presencePenalty: undefined,
            frequencyPenalty: undefined,
            responseMimeType: undefined,
            seed: undefined,
            responseLogprobs: undefined,
            logprobs: undefined,
            audioTimestamp: undefined,
          },
        },
        timeout: REQUEST_TIMEOUT_MS,
      }),
    );
  });

  it('should return cached response if available', async () => {
    const mockCachedResponse = {
      cached: true,
      output: 'cached response text',
      tokenUsage: {
        total: 10,
        prompt: 5,
        completion: 5,
      },
    };

    jest.mocked(getCache().get).mockResolvedValue(JSON.stringify(mockCachedResponse));

    const response = await provider.callGeminiApi('test prompt');

    expect(response).toEqual({
      ...mockCachedResponse,
      tokenUsage: {
        ...mockCachedResponse.tokenUsage,
        cached: mockCachedResponse.tokenUsage.total,
      },
    });
  });

  it('should handle API call errors', async () => {
    const mockError = new Error('something went wrong');
    jest.spyOn(vertexUtil, 'getGoogleClient').mockResolvedValue({
      client: {
        request: jest.fn().mockRejectedValue(mockError),
      } as unknown as JSONClient,
      projectId: 'test-project-id',
    });

    const response = await provider.callGeminiApi('test prompt');

    expect(response).toEqual({
      error: `API call error: Error: something went wrong`,
    });
  });

  it('should handle API response errors', async () => {
    const mockResponse = {
      data: [
        {
          error: {
            code: 400,
            message: 'Bad Request',
          },
        },
      ],
    };

    jest.spyOn(vertexUtil, 'getGoogleClient').mockResolvedValue({
      client: {
        request: jest.fn().mockResolvedValue(mockResponse),
      } as unknown as JSONClient,
      projectId: 'test-project-id',
    });

    const response = await provider.callGeminiApi('test prompt');

    expect(response).toEqual({
      error: 'Error 400: Bad Request',
    });
  });
});

describe('maybeCoerceToGeminiFormat', () => {
  it('should handle chat format', () => {
    const input = [
      {
        role: 'user',
        parts: [{ text: 'Hello' }],
      },
    ] as GeminiFormat;
    const result = vertexUtil.maybeCoerceToGeminiFormat(input);
    expect(result.contents).toEqual(input);
  });

  it('should handle OpenAI format', () => {
    const input = [
      {
        role: 'user',
        content: 'Hello',
      },
    ] as OpenAIMessage[];
    const result = vertexUtil.maybeCoerceToGeminiFormat(input);
    expect(result.contents).toEqual([
      {
        role: 'user',
        parts: [{ text: 'Hello' }],
      },
    ]);
  });

  it('should handle string input', () => {
    const input = 'Hello';
    const result = vertexUtil.maybeCoerceToGeminiFormat(input);
    expect(result.contents).toEqual([
      {
        parts: [{ text: 'Hello' }],
      },
    ]);
  });

  it('should handle system messages', () => {
    const input = [
      {
        role: 'system',
        content: 'You are a helpful assistant.',
      },
      {
        role: 'user',
        content: 'Hello!',
      },
    ] as OpenAIMessage[];
    const result = vertexUtil.maybeCoerceToGeminiFormat(input);
    expect(result.contents).toEqual([
      {
        role: 'user',
        parts: [{ text: 'Hello!' }],
      },
    ]);
    expect(result.systemInstruction).toEqual({
      role: 'system',
      parts: [{ text: 'You are a helpful assistant.' }],
    });
  });

  it('should handle unknown formats', () => {
    const input = { unknownFormat: 'test' };
    const result = vertexUtil.maybeCoerceToGeminiFormat(input);
    expect(result.contents).toEqual([
      {
        parts: [{ text: JSON.stringify(input) }],
      },
    ]);
    expect(result.coerced).toBe(true);
  });
});
