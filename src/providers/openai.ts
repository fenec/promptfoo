import type { Cache } from 'cache-manager';
import OpenAI from 'openai';
import type WebSocket from 'ws';
import type { WebSocket as WSType } from 'ws';
import { fetchWithCache, getCache, isCacheEnabled } from '../cache';
import { getEnvString, getEnvFloat, getEnvInt } from '../envars';
import logger from '../logger';
import type {
  ApiModerationProvider,
  ApiProvider,
  CallApiContextParams,
  CallApiOptionsParams,
  EnvOverrides,
  ModerationFlag,
  ProviderEmbeddingResponse,
  ProviderModerationResponse,
  ProviderResponse,
  TokenUsage,
} from '../types';
import { renderVarsInObject } from '../util';
import { maybeLoadFromExternalFile } from '../util';
import { safeJsonStringify } from '../util/json';
import { sleep } from '../util/time';
import type { OpenAiFunction, OpenAiTool } from './openaiUtil';
import { calculateCost, REQUEST_TIMEOUT_MS, parseChatPrompt, toTitleCase } from './shared';

export const OPENAI_AUDIO_MODELS = [
  ...['gpt-4o-audio-preview', 'gpt-4o-realtime-preview-2024-10-01'].map((model) => ({
    id: model,
    cost: {
      input: 15 / 1e6,
      output: 60 / 1e6,
    },
    requiresAudio: true,
  })),
];

// see https://platform.openai.com/docs/models
const OPENAI_CHAT_MODELS = [
  ...OPENAI_AUDIO_MODELS,
  ...['o1-preview', 'o1-preview-2024-09-12'].map((model) => ({
    id: model,
    cost: {
      input: 15 / 1e6,
      output: 60 / 1e6,
    },
  })),
  ...['o1-mini', 'o1-mini-2024-09-12'].map((model) => ({
    id: model,
    cost: {
      input: 3 / 1e6,
      output: 12 / 1e6,
    },
  })),
  ...['gpt-4o', 'gpt-4o-2024-08-06'].map((model) => ({
    id: model,
    cost: {
      input: 2.5 / 1e6,
      output: 10 / 1e6,
    },
  })),
  ...['gpt-4o-2024-05-13'].map((model) => ({
    id: model,
    cost: {
      input: 5 / 1000000,
      output: 15 / 1000000,
    },
  })),
  ...['gpt-4o-mini', 'gpt-4o-mini-2024-07-18'].map((model) => ({
    id: model,
    cost: {
      input: 0.15 / 1000000,
      output: 0.6 / 1000000,
    },
  })),
  ...['gpt-4', 'gpt-4-0613'].map((model) => ({
    id: model,
    cost: {
      input: 30 / 1000000,
      output: 60 / 1000000,
    },
  })),
  ...[
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-turbo-preview',
    'gpt-4-0125-preview',
    'gpt-4-1106-preview',
  ].map((model) => ({
    id: model,
    cost: {
      input: 10 / 1000000,
      output: 30 / 1000000,
    },
  })),
  {
    id: 'gpt-3.5-turbo',
    cost: {
      input: 0.5 / 1000000,
      output: 1.5 / 1000000,
    },
  },
  {
    id: 'gpt-3.5-turbo-0125',
    cost: {
      input: 0.5 / 1000000,
      output: 1.5 / 1000000,
    },
  },
  {
    id: 'gpt-3.5-turbo-1106',
    cost: {
      input: 1 / 1000000,
      output: 2 / 1000000,
    },
  },
  ...['gpt-3.5-turbo-instruct'].map((model) => ({
    id: model,
    cost: {
      input: 1.5 / 1000000,
      output: 2 / 1000000,
    },
  })),
  ...OPENAI_AUDIO_MODELS,
];

// See https://platform.openai.com/docs/models/model-endpoint-compatibility
const OPENAI_COMPLETION_MODELS = [
  {
    id: 'gpt-3.5-turbo-instruct',
    cost: {
      input: 1.5 / 1000000,
      output: 2 / 1000000,
    },
  },
  {
    id: 'text-davinci-002',
  },
  {
    id: 'text-babbage-002',
  },
];

interface OpenAiSharedOptions {
  apiKey?: string;
  apiKeyEnvar?: string;
  apiHost?: string;
  apiBaseUrl?: string;
  organization?: string;
  cost?: number;
  headers?: { [key: string]: string };
  voice?: 'alloy' | 'echo' | 'shimmer' | 'ash' | 'ballad' | 'coral' | 'sage' | 'verse';
  format?: 'wav' | 'mp3' | 'opus' | 'flac' | 'pcm16';
}

export type OpenAiCompletionOptions = OpenAiSharedOptions & {
  temperature?: number;
  max_completion_tokens?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  best_of?: number;
  functions?: OpenAiFunction[];
  function_call?: 'none' | 'auto' | { name: string };
  tools?: OpenAiTool[];
  tool_choice?: 'none' | 'auto' | 'required' | { type: 'function'; function?: { name: string } };
  response_format?:
    | {
        type: 'json_object';
      }
    | {
        type: 'json_schema';
        json_schema: {
          name: string;
          strict: boolean;
          schema: {
            type: 'object';
            properties: Record<string, any>;
            required?: string[];
            additionalProperties: false;
          };
        };
      };
  stop?: string[];
  seed?: number;
  passthrough?: object;

  /**
   * If set, automatically call these functions when the assistant activates
   * these function tools.
   */
  functionToolCallbacks?: Record<
    OpenAI.FunctionDefinition['name'],
    (arg: string) => Promise<string>
  >;
};

function failApiCall(err: any) {
  if (err instanceof OpenAI.APIError) {
    return {
      error: `API error: ${err.type} ${err.message}`,
    };
  }
  return {
    error: `API error: ${String(err)}`,
  };
}

function getTokenUsage(data: any, cached: boolean): Partial<TokenUsage> {
  if (data.usage) {
    if (cached) {
      return { cached: data.usage.total_tokens, total: data.usage.total_tokens };
    } else {
      return {
        total: data.usage.total_tokens,
        prompt: data.usage.prompt_tokens || 0,
        completion: data.usage.completion_tokens || 0,
      };
    }
  }
  return {};
}

export class OpenAiGenericProvider implements ApiProvider {
  modelName: string;

  config: OpenAiSharedOptions;
  env?: EnvOverrides;

  constructor(
    modelName: string,
    options: { config?: OpenAiSharedOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    const { config, id, env } = options;
    this.env = env;
    this.modelName = modelName;
    this.config = config || {};
    this.id = id ? () => id : this.id;
  }

  id(): string {
    return this.config.apiHost || this.config.apiBaseUrl
      ? this.modelName
      : `openai:${this.modelName}`;
  }

  toString(): string {
    return `[OpenAI Provider ${this.modelName}]`;
  }

  getOrganization(): string | undefined {
    return (
      this.config.organization ||
      this.env?.OPENAI_ORGANIZATION ||
      getEnvString('OPENAI_ORGANIZATION')
    );
  }

  getApiUrlDefault(): string {
    return 'https://api.openai.com/v1';
  }

  getApiUrl(): string {
    const apiHost =
      this.config.apiHost || this.env?.OPENAI_API_HOST || getEnvString('OPENAI_API_HOST');
    if (apiHost) {
      return `https://${apiHost}/v1`;
    }
    return (
      this.config.apiBaseUrl ||
      this.env?.OPENAI_API_BASE_URL ||
      this.env?.OPENAI_BASE_URL ||
      getEnvString('OPENAI_API_BASE_URL') ||
      getEnvString('OPENAI_BASE_URL') ||
      this.getApiUrlDefault()
    );
  }

  getApiKey(): string | undefined {
    return (
      this.config.apiKey ||
      (this.config?.apiKeyEnvar
        ? process.env[this.config.apiKeyEnvar] ||
          this.env?.[this.config.apiKeyEnvar as keyof EnvOverrides]
        : undefined) ||
      this.env?.OPENAI_API_KEY ||
      getEnvString('OPENAI_API_KEY')
    );
  }

  // @ts-ignore: Params are not used in this implementation
  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    throw new Error('Not implemented');
  }
}

export class OpenAiEmbeddingProvider extends OpenAiGenericProvider {
  async callEmbeddingApi(text: string): Promise<ProviderEmbeddingResponse> {
    if (!this.getApiKey()) {
      throw new Error('OpenAI API key must be set for similarity comparison');
    }

    const body = {
      input: text,
      model: this.modelName,
    };
    let data,
      cached = false;
    try {
      ({ data, cached } = (await fetchWithCache(
        `${this.getApiUrl()}/embeddings`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.getApiKey()}`,
            ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
            ...this.config.headers,
          },
          body: JSON.stringify(body),
        },
        REQUEST_TIMEOUT_MS,
      )) as unknown as any);
    } catch (err) {
      logger.error(`API call error: ${err}`);
      throw err;
    }
    logger.debug(`\tOpenAI embeddings API response: ${JSON.stringify(data)}`);

    try {
      const embedding = data?.data?.[0]?.embedding;
      if (!embedding) {
        throw new Error('No embedding found in OpenAI embeddings API response');
      }
      return {
        embedding,
        tokenUsage: getTokenUsage(data, cached),
      };
    } catch (err) {
      logger.error(data.error.message);
      throw err;
    }
  }
}

function formatOpenAiError(data: {
  error: { message: string; type?: string; code?: string };
}): string {
  let errorMessage = `API error: ${data.error.message}`;
  if (data.error.type) {
    errorMessage += `, Type: ${data.error.type}`;
  }
  if (data.error.code) {
    errorMessage += `, Code: ${data.error.code}`;
  }
  errorMessage += '\n\n' + safeJsonStringify(data, true /* prettyPrint */);
  return errorMessage;
}

export function calculateOpenAICost(
  modelName: string,
  config: OpenAiSharedOptions,
  promptTokens?: number,
  completionTokens?: number,
): number | undefined {
  return calculateCost(modelName, config, promptTokens, completionTokens, [
    ...OPENAI_CHAT_MODELS,
    ...OPENAI_COMPLETION_MODELS,
  ]);
}

export class OpenAiCompletionProvider extends OpenAiGenericProvider {
  static OPENAI_COMPLETION_MODELS = OPENAI_COMPLETION_MODELS;

  static OPENAI_COMPLETION_MODEL_NAMES = OPENAI_COMPLETION_MODELS.map((model) => model.id);

  config: OpenAiCompletionOptions;

  constructor(
    modelName: string,
    options: { config?: OpenAiCompletionOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    super(modelName, options);
    this.config = options.config || {};
    if (
      !OpenAiCompletionProvider.OPENAI_COMPLETION_MODEL_NAMES.includes(modelName) &&
      this.getApiUrl() === this.getApiUrlDefault()
    ) {
      logger.warn(`FYI: Using unknown OpenAI completion model: ${modelName}`);
    }
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    let stop: string;
    try {
      stop = getEnvString('OPENAI_STOP')
        ? JSON.parse(getEnvString('OPENAI_STOP') || '')
        : this.config?.stop || ['<|im_end|>', '<|endoftext|>'];
    } catch (err) {
      throw new Error(`OPENAI_STOP is not a valid JSON string: ${err}`);
    }
    const body = {
      model: this.modelName,
      prompt,
      seed: this.config.seed,
      max_tokens: this.config.max_tokens ?? getEnvInt('OPENAI_MAX_TOKENS', 1024),
      temperature: this.config.temperature ?? getEnvFloat('OPENAI_TEMPERATURE', 0),
      top_p: this.config.top_p ?? getEnvFloat('OPENAI_TOP_P', 1),
      presence_penalty: this.config.presence_penalty ?? getEnvFloat('OPENAI_PRESENCE_PENALTY', 0),
      frequency_penalty:
        this.config.frequency_penalty ?? getEnvFloat('OPENAI_FREQUENCY_PENALTY', 0),
      best_of: this.config.best_of ?? getEnvInt('OPENAI_BEST_OF', 1),
      ...(callApiOptions?.includeLogProbs ? { logprobs: callApiOptions.includeLogProbs } : {}),
      ...(stop ? { stop } : {}),
      ...(this.config.passthrough || {}),
    };
    logger.debug(`Calling OpenAI API: ${JSON.stringify(body)}`);
    let data,
      cached = false;
    try {
      ({ data, cached } = (await fetchWithCache(
        `${this.getApiUrl()}/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.getApiKey()}`,
            ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
            ...this.config.headers,
          },
          body: JSON.stringify(body),
        },
        REQUEST_TIMEOUT_MS,
      )) as unknown as any);
    } catch (err) {
      logger.error(`API call error: ${String(err)}`);
      return {
        error: `API call error: ${String(err)}`,
      };
    }
    logger.debug(`\tOpenAI completions API response: ${JSON.stringify(data)}`);
    if (data.error) {
      return {
        error: formatOpenAiError(data),
      };
    }
    try {
      return {
        output: data.choices[0].text,
        tokenUsage: getTokenUsage(data, cached),
        cached,
        cost: calculateOpenAICost(
          this.modelName,
          this.config,
          data.usage?.prompt_tokens,
          data.usage?.completion_tokens,
        ),
      };
    } catch (err) {
      return {
        error: `API error: ${String(err)}: ${JSON.stringify(data)}`,
      };
    }
  }
}

export class OpenAiChatCompletionProvider extends OpenAiGenericProvider {
  static OPENAI_CHAT_MODELS = OPENAI_CHAT_MODELS;

  static OPENAI_CHAT_MODEL_NAMES = OPENAI_CHAT_MODELS.map((model) => model.id);

  config: OpenAiCompletionOptions;

  constructor(
    modelName: string,
    options: { config?: OpenAiCompletionOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    if (!OpenAiChatCompletionProvider.OPENAI_CHAT_MODEL_NAMES.includes(modelName)) {
      logger.debug(`Using unknown OpenAI chat model: ${modelName}`);
    }
    super(modelName, options);
    this.config = options.config || {};
  }

  getOpenAiBody(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ) {
    // Merge configs from the provider and the prompt
    const config = {
      ...this.config,
      ...context?.prompt?.config,
    };

    const messages = parseChatPrompt(prompt, [{ role: 'user', content: prompt }]);

    // NOTE: Special handling for o1 models which do not support max_tokens and temperature
    const isO1Model = this.modelName.startsWith('o1-');
    const maxCompletionTokens = isO1Model
      ? (config.max_completion_tokens ?? getEnvInt('OPENAI_MAX_COMPLETION_TOKENS'))
      : undefined;
    const maxTokens = isO1Model
      ? undefined
      : (config.max_tokens ?? getEnvInt('OPENAI_MAX_TOKENS', 1024));
    const temperature = isO1Model
      ? undefined
      : (config.temperature ?? getEnvFloat('OPENAI_TEMPERATURE', 0));

    const body = {
      model: this.modelName,
      messages,
      seed: config.seed,
      ...(maxTokens ? { max_tokens: maxTokens } : {}),
      ...(maxCompletionTokens ? { max_completion_tokens: maxCompletionTokens } : {}),
      ...(temperature ? { temperature } : {}),
      top_p: config.top_p ?? Number.parseFloat(process.env.OPENAI_TOP_P || '1'),
      presence_penalty:
        config.presence_penalty ?? Number.parseFloat(process.env.OPENAI_PRESENCE_PENALTY || '0'),
      frequency_penalty:
        config.frequency_penalty ?? Number.parseFloat(process.env.OPENAI_FREQUENCY_PENALTY || '0'),
      ...(config.functions
        ? {
            functions: maybeLoadFromExternalFile(
              renderVarsInObject(config.functions, context?.vars),
            ),
          }
        : {}),
      ...(config.function_call ? { function_call: config.function_call } : {}),
      ...(config.tools
        ? { tools: maybeLoadFromExternalFile(renderVarsInObject(config.tools, context?.vars)) }
        : {}),
      ...(config.tool_choice ? { tool_choice: config.tool_choice } : {}),
      ...(config.response_format
        ? {
            response_format: maybeLoadFromExternalFile(
              renderVarsInObject(config.response_format, context?.vars),
            ),
          }
        : {}),
      ...(callApiOptions?.includeLogProbs ? { logprobs: callApiOptions.includeLogProbs } : {}),
      ...(config.stop ? { stop: config.stop } : {}),
      ...(config.passthrough || {}),
    };

    return { body, config };
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    const { body, config } = this.getOpenAiBody(prompt, context, callApiOptions);
    logger.debug(`Calling OpenAI API: ${JSON.stringify(body)}`);

    let data, status, statusText;
    let cached = false;
    try {
      ({ data, cached, status, statusText } = await fetchWithCache(
        `${this.getApiUrl()}/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.getApiKey()}`,
            ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
            ...config.headers,
          },
          body: JSON.stringify(body),
        },
        REQUEST_TIMEOUT_MS,
      ));

      if (status < 200 || status >= 300) {
        return {
          error: `API error: ${status} ${statusText}\n${typeof data === 'string' ? data : JSON.stringify(data)}`,
        };
      }
    } catch (err) {
      logger.error(`API call error: ${String(err)}`);
      return {
        error: `API call error: ${String(err)}`,
      };
    }

    logger.debug(`\tOpenAI chat completions API response: ${JSON.stringify(data)}`);
    if (data.error) {
      return {
        error: formatOpenAiError(data),
      };
    }
    try {
      const message = data.choices[0].message;
      if (message.refusal) {
        return {
          output: message.refusal,
          tokenUsage: getTokenUsage(data, cached),
          isRefusal: true,
        };
      }
      let output = '';
      if (message.content && (message.function_call || message.tool_calls)) {
        if (Array.isArray(message.tool_calls) && message.tool_calls.length === 0) {
          output = message.content;
        }
        output = message;
      } else if (message.content === null) {
        output = message.function_call || message.tool_calls;
      } else {
        output = message.content;
      }
      const logProbs = data.choices[0].logprobs?.content?.map(
        (logProbObj: { token: string; logprob: number }) => logProbObj.logprob,
      );

      // Handle structured output
      if (config.response_format?.type === 'json_schema' && typeof output === 'string') {
        try {
          output = JSON.parse(output);
        } catch (error) {
          logger.error(`Failed to parse JSON output: ${error}`);
        }
      }

      // Handle function tool callbacks
      const functionCalls = message.function_call ? [message.function_call] : message.tool_calls;
      if (functionCalls && config.functionToolCallbacks) {
        const results = [];
        for (const functionCall of functionCalls) {
          const functionName = functionCall.name || functionCall.function?.name;
          if (config.functionToolCallbacks[functionName]) {
            try {
              const functionResult = await config.functionToolCallbacks[functionName](
                functionCall.arguments || functionCall.function?.arguments,
              );
              results.push(functionResult);
            } catch (error) {
              logger.error(`Error executing function ${functionName}: ${error}`);
            }
          }
        }
        if (results.length > 0) {
          return {
            output: results.join('\n'),
            tokenUsage: getTokenUsage(data, cached),
            cached,
            logProbs,
            cost: calculateOpenAICost(
              this.modelName,
              config,
              data.usage?.prompt_tokens,
              data.usage?.completion_tokens,
            ),
          };
        }
      }

      return {
        output,
        tokenUsage: getTokenUsage(data, cached),
        cached,
        logProbs,
        cost: calculateOpenAICost(
          this.modelName,
          config,
          data.usage?.prompt_tokens,
          data.usage?.completion_tokens,
        ),
      };
    } catch (err) {
      return {
        error: `API error: ${String(err)}: ${JSON.stringify(data)}`,
      };
    }
  }
}

type OpenAiAssistantOptions = OpenAiSharedOptions & {
  modelName?: string;
  instructions?: string;
  tools?: OpenAI.Beta.Threads.ThreadCreateAndRunParams['tools'];
  /**
   * If set, automatically call these functions when the assistant activates
   * these function tools.
   */
  functionToolCallbacks?: Record<
    OpenAI.FunctionDefinition['name'],
    (arg: string) => Promise<string>
  >;
  metadata?: object[];
  temperature?: number;
  toolChoice?:
    | 'none'
    | 'auto'
    | { type: 'function'; function?: { name: string } }
    | { type: 'file_search' };
  attachments?: OpenAI.Beta.Threads.Message.Attachment[];
};

export class OpenAiAssistantProvider extends OpenAiGenericProvider {
  assistantId: string;
  assistantConfig: OpenAiAssistantOptions;

  constructor(
    assistantId: string,
    options: { config?: OpenAiAssistantOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    super(assistantId, options);
    this.assistantConfig = options.config || {};
    this.assistantId = assistantId;
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    const openai = new OpenAI({
      apiKey: this.getApiKey(),
      organization: this.getOrganization(),
      baseURL: this.getApiUrl(),
      maxRetries: 3,
      timeout: REQUEST_TIMEOUT_MS,
      defaultHeaders: this.assistantConfig.headers,
    });

    const messages = parseChatPrompt(prompt, [
      {
        role: 'user',
        content: prompt,
        ...(this.assistantConfig.attachments
          ? { attachments: this.assistantConfig.attachments }
          : {}),
      },
    ]) as OpenAI.Beta.Threads.ThreadCreateParams.Message[];
    const body: OpenAI.Beta.Threads.ThreadCreateAndRunParams = {
      assistant_id: this.assistantId,
      model: this.assistantConfig.modelName || undefined,
      instructions: this.assistantConfig.instructions || undefined,
      tools:
        maybeLoadFromExternalFile(renderVarsInObject(this.assistantConfig.tools, context?.vars)) ||
        undefined,
      metadata: this.assistantConfig.metadata || undefined,
      temperature: this.assistantConfig.temperature || undefined,
      tool_choice: this.assistantConfig.toolChoice || undefined,
      thread: {
        messages,
      },
    };

    logger.debug(`Calling OpenAI API, creating thread run: ${JSON.stringify(body)}`);
    let run;
    try {
      run = await openai.beta.threads.createAndRun(body);
    } catch (err) {
      return failApiCall(err);
    }

    logger.debug(`\tOpenAI thread run API response: ${JSON.stringify(run)}`);

    while (
      run.status === 'in_progress' ||
      run.status === 'queued' ||
      run.status === 'requires_action'
    ) {
      if (run.status === 'requires_action') {
        const requiredAction: OpenAI.Beta.Threads.Runs.Run.RequiredAction | null =
          run.required_action;
        if (requiredAction === null || requiredAction.type !== 'submit_tool_outputs') {
          break;
        }
        const functionCallsWithCallbacks: OpenAI.Beta.Threads.Runs.RequiredActionFunctionToolCall[] =
          requiredAction.submit_tool_outputs.tool_calls.filter((toolCall) => {
            return (
              toolCall.type === 'function' &&
              toolCall.function.name in (this.assistantConfig.functionToolCallbacks ?? {})
            );
          });
        if (functionCallsWithCallbacks.length === 0) {
          break;
        }
        logger.debug(
          `Calling functionToolCallbacks for functions: ${functionCallsWithCallbacks.map(
            ({ function: { name } }) => name,
          )}`,
        );
        const toolOutputs = await Promise.all(
          functionCallsWithCallbacks.map(async (toolCall) => {
            logger.debug(
              `Calling functionToolCallbacks[${toolCall.function.name}]('${toolCall.function.arguments}')`,
            );
            const result = await this.assistantConfig.functionToolCallbacks![
              toolCall.function.name
            ](toolCall.function.arguments);
            return {
              tool_call_id: toolCall.id,
              output: result,
            };
          }),
        );
        logger.debug(
          `Calling OpenAI API, submitting tool outputs for ${run.thread_id}: ${JSON.stringify(
            toolOutputs,
          )}`,
        );
        try {
          run = await openai.beta.threads.runs.submitToolOutputs(run.thread_id, run.id, {
            tool_outputs: toolOutputs,
          });
        } catch (err) {
          return failApiCall(err);
        }
        continue;
      }

      await sleep(1000);

      logger.debug(`Calling OpenAI API, getting thread run ${run.id} status`);
      try {
        run = await openai.beta.threads.runs.retrieve(run.thread_id, run.id);
      } catch (err) {
        return failApiCall(err);
      }
      logger.debug(`\tOpenAI thread run API response: ${JSON.stringify(run)}`);
    }

    if (run.status !== 'completed' && run.status !== 'requires_action') {
      if (run.last_error) {
        return {
          error: `Thread run failed: ${run.last_error.message}`,
        };
      }
      return {
        error: `Thread run failed: ${run.status}`,
      };
    }

    // Get run steps
    logger.debug(`Calling OpenAI API, getting thread run steps for ${run.thread_id}`);
    let steps;
    try {
      steps = await openai.beta.threads.runs.steps.list(run.thread_id, run.id, {
        order: 'asc',
      });
    } catch (err) {
      return failApiCall(err);
    }
    logger.debug(`\tOpenAI thread run steps API response: ${JSON.stringify(steps)}`);

    const outputBlocks = [];
    for (const step of steps.data) {
      if (step.step_details.type === 'message_creation') {
        logger.debug(`Calling OpenAI API, getting message ${step.id}`);
        let message;
        try {
          message = await openai.beta.threads.messages.retrieve(
            run.thread_id,
            step.step_details.message_creation.message_id,
          );
        } catch (err) {
          return failApiCall(err);
        }
        logger.debug(`\tOpenAI thread run step message API response: ${JSON.stringify(message)}`);

        const content = message.content
          .map((content) =>
            content.type === 'text' ? content.text.value : `<${content.type} output>`,
          )
          .join('\n');
        outputBlocks.push(`[${toTitleCase(message.role)}] ${content}`);
      } else if (step.step_details.type === 'tool_calls') {
        for (const toolCall of step.step_details.tool_calls) {
          if (toolCall.type === 'function') {
            outputBlocks.push(
              `[Call function ${toolCall.function.name} with arguments ${toolCall.function.arguments}]`,
            );
            outputBlocks.push(`[Function output: ${toolCall.function.output}]`);
          } else if (toolCall.type === 'file_search') {
            outputBlocks.push(`[Ran file search]`);
          } else if (toolCall.type === 'code_interpreter') {
            const output = toolCall.code_interpreter.outputs
              .map((output) => (output.type === 'logs' ? output.logs : `<${output.type} output>`))
              .join('\n');
            outputBlocks.push(`[Code interpreter input]`);
            outputBlocks.push(toolCall.code_interpreter.input);
            outputBlocks.push(`[Code interpreter output]`);
            outputBlocks.push(output);
          } else {
            outputBlocks.push(`[Unknown tool call type: ${(toolCall as any).type}]`);
          }
        }
      } else {
        outputBlocks.push(`[Unknown step type: ${(step.step_details as any).type}]`);
      }
    }

    return {
      output: outputBlocks.join('\n\n').trim(),
      tokenUsage: getTokenUsage(run, false),
    };
  }
}

type OpenAiImageOptions = OpenAiSharedOptions & {
  size?: string;
};

export class OpenAiImageProvider extends OpenAiGenericProvider {
  config: OpenAiImageOptions;

  constructor(
    modelName: string,
    options: { config?: OpenAiImageOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    super(modelName, options);
    this.config = options.config || {};
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    const cache = getCache();
    const cacheKey = `openai:image:${safeJsonStringify({ context, prompt })}`;

    if (!this.getApiKey()) {
      throw new Error(
        'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    const openai = new OpenAI({
      apiKey: this.getApiKey(),
      organization: this.getOrganization(),
      baseURL: this.getApiUrl(),
      maxRetries: 3,
      timeout: REQUEST_TIMEOUT_MS,
      defaultHeaders: this.config.headers,
    });

    let response: OpenAI.Images.ImagesResponse | undefined;
    let cached = false;
    if (isCacheEnabled()) {
      // Try to get the cached response
      const cachedResponse = await cache.get(cacheKey);
      if (cachedResponse) {
        logger.debug(`Retrieved cached response for ${prompt}: ${cachedResponse}`);
        response = JSON.parse(cachedResponse as string) as OpenAI.Images.ImagesResponse;
        cached = true;
      }
    }

    if (!response) {
      try {
        response = await openai.images.generate({
          model: this.modelName,
          prompt,
          n: 1,
          size:
            ((this.config.size || process.env.OPENAI_IMAGE_SIZE) as
              | '1024x1024'
              | '256x256'
              | '512x512'
              | '1792x1024'
              | '1024x1792'
              | undefined) || '1024x1024',
        });
      } catch (error) {
        return {
          error: `OpenAI threw error: ${(error as Error).message}`,
        };
      }
    }
    const url = response.data[0].url;
    if (!url) {
      return {
        error: `No image URL found in response: ${JSON.stringify(response)}`,
      };
    }

    if (!cached && isCacheEnabled()) {
      try {
        await cache.set(cacheKey, JSON.stringify(response));
      } catch (err) {
        logger.error(`Failed to cache response: ${String(err)}`);
      }
    }

    const sanitizedPrompt = prompt
      .replace(/\r?\n|\r/g, ' ')
      .replace(/\[/g, '(')
      .replace(/\]/g, ')');
    const ellipsizedPrompt =
      sanitizedPrompt.length > 50 ? `${sanitizedPrompt.substring(0, 47)}...` : sanitizedPrompt;
    return {
      output: `![${ellipsizedPrompt}](${url})`,
      cached,
    };
  }
}

export class OpenAiModerationProvider
  extends OpenAiGenericProvider
  implements ApiModerationProvider
{
  async callModerationApi(
    userPrompt: string, // userPrompt is not supported by OpenAI moderation API
    assistantResponse: string,
  ): Promise<ProviderModerationResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    const openai = new OpenAI({
      apiKey: this.getApiKey(),
      organization: this.getOrganization(),
      baseURL: this.getApiUrl(),
      maxRetries: 3,
      timeout: REQUEST_TIMEOUT_MS,
    });

    let cache: Cache | undefined;
    let cacheKey: string | undefined;
    if (isCacheEnabled()) {
      cache = await getCache();
      cacheKey = `openai:${this.modelName}:${JSON.stringify(
        this.config,
      )}:${userPrompt}:${assistantResponse}`;

      // Try to get the cached response
      const cachedResponse = await cache.get(cacheKey);

      if (cachedResponse) {
        logger.debug(`Returning cached response for ${userPrompt}: ${cachedResponse}`);
        return JSON.parse(cachedResponse as string);
      }
    }

    logger.debug(
      `Calling OpenAI moderation API: prompt [${userPrompt}] assistant [${assistantResponse}]`,
    );
    let moderation: OpenAI.Moderations.ModerationCreateResponse | undefined;
    try {
      moderation = await openai.moderations.create({
        model: this.modelName,
        input: assistantResponse,
      });
    } catch (err) {
      logger.error(`API call error: ${String(err)}`);
      return {
        error: `API call error: ${String(err)}`,
      };
    }

    logger.debug(`\tOpenAI moderation API response: ${JSON.stringify(moderation)}`);
    try {
      const { results } = moderation;

      const flags: ModerationFlag[] = [];
      if (!results) {
        throw new Error('API response error: no results');
      }

      if (cache && cacheKey) {
        await cache.set(cacheKey, JSON.stringify(moderation));
      }

      if (results.length === 0) {
        return { flags };
      }

      for (const result of results) {
        if (result.flagged) {
          for (const [category, flagged] of Object.entries(result.categories)) {
            if (flagged) {
              flags.push({
                code: category,
                description: category,
                confidence:
                  result.category_scores[category as keyof OpenAI.Moderation.CategoryScores],
              });
            }
          }
        }
      }
      return { flags };
    } catch (err) {
      return {
        error: `API response error: ${String(err)}: ${JSON.stringify(moderation)}`,
      };
    }
  }
}

// Update the type definition for voice and format options
type OpenAiAudioOptions = OpenAiSharedOptions & {
  voice?: 'alloy' | 'echo' | 'shimmer' | 'ash' | 'ballad' | 'coral' | 'sage' | 'verse';
  format?: 'wav' | 'mp3' | 'opus' | 'flac' | 'pcm16';
};

// First, let's extend the ProviderResponse type to include audio-related fields
declare module '../types' {
  interface ProviderResponse {
    audio?: {
      data: string;
      id: string;
      expires_at: number;
      transcript?: string;
    };
  }
}

export class OpenAiAudioProvider extends OpenAiGenericProvider {
  config: OpenAiAudioOptions;
  private ws: WSType | null = null;

  static OPENAI_AUDIO_MODELS = OPENAI_AUDIO_MODELS;

  constructor(
    modelName: string,
    options: { config?: OpenAiAudioOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    super(modelName, options);
    this.config = options.config || {};
  }

  private async initWebSocket(): Promise<WSType> {
    const WebSocket = (await import('ws')).default;
    logger.debug('Initializing WebSocket connection...');

    const ws = new WebSocket(`wss://api.openai.com/v1/realtime?model=${this.modelName}`, {
      headers: {
        Authorization: `Bearer ${this.getApiKey()}`,
        'OpenAI-Beta': 'realtime=v1',
        ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
        ...this.config.headers,
      },
    });

    return new Promise((resolve, reject) => {
      const connectionTimeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
        ws.close();
      }, 10000); // 10 second connection timeout

      ws.on('open', () => {
        logger.debug('WebSocket connection established');
        clearTimeout(connectionTimeout);
        resolve(ws);
      });

      ws.on('error', (error) => {
        logger.error(`WebSocket connection error: ${error}`);
        clearTimeout(connectionTimeout);
        reject(error);
      });

      ws.on('close', (code, reason) => {
        logger.debug(
          `WebSocket closed during initialization with code ${code} and reason: ${reason}`,
        );
        clearTimeout(connectionTimeout);
        reject(
          new Error(
            `WebSocket closed during initialization (${code}${reason ? ': ' + reason : ''})`,
          ),
        );
      });
    });
  }

  private async waitForEvent(
    ws: WSType,
    expectedType: string,
    timeoutMs: number = 5000,
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Timeout waiting for event ${expectedType}`));
      }, timeoutMs);

      const messageHandler = (data: WebSocket.RawData) => {
        try {
          const event = JSON.parse(data.toString());
          if (event.type === expectedType) {
            clearTimeout(timeout);
            ws.removeListener('message', messageHandler);
            resolve(event);
          } else if (event.type === 'error') {
            clearTimeout(timeout);
            ws.removeListener('message', messageHandler);
            reject(new Error(`API error: ${event.error.message}`));
          }
        } catch (err) {
          // Continue waiting if we can't parse the message
          logger.debug(`Error parsing message while waiting for ${expectedType}: ${err}`);
        }
      };

      ws.on('message', messageHandler);
    });
  }

  private async sendMessage(ws: WSType, message: any): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        logger.debug(`Sending message: ${JSON.stringify(message)}`);
        ws.send(JSON.stringify(message), (error) => {
          if (error) {
            reject(error);
          } else {
            resolve();
          }
        });
      } catch (err) {
        reject(err);
      }
    });
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    let ws: WSType | null = null;
    try {
      logger.debug('Starting API call...');
      ws = await this.initWebSocket();
      this.ws = ws;

      logger.debug('Initializing session...');
      await this.sendMessage(ws, {
        type: 'session.update',
        session: {
          voice: this.config.voice || 'alloy',
        },
      });
      // Wait for session acknowledgment
      await this.waitForEvent(ws, 'session.updated');

      logger.debug('Creating response...');
      await this.sendMessage(ws, {
        type: 'response.create',
        response: {
          modalities: ['audio', 'text'],
          instructions:
            'You are a helpful AI assistant. Please provide clear and concise responses.',
        },
      });
      // Wait for response creation acknowledgment
      await this.waitForEvent(ws, 'response.created');

      logger.debug('Sending user message...');
      await this.sendMessage(ws, {
        type: 'conversation.item.create',
        item: {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'text',
              text: prompt,
            },
          ],
        },
      });

      return new Promise((resolve, reject) => {
        if (!ws) {
          reject(new Error('WebSocket connection not initialized'));
          return;
        }

        let output = '';
        let audioData = '';
        let audioId = '';
        let expiresAt = 0;
        let isDone = false;
        let hasStartedReceiving = false;

        const timeout = setTimeout(() => {
          if (!isDone) {
            logger.error('Request timed out');
            ws?.close();
            reject(new Error('Request timed out'));
          }
        }, REQUEST_TIMEOUT_MS);

        ws.on('close', (code, reason) => {
          logger.debug(`WebSocket closed during response with code ${code} and reason: ${reason}`);
          clearTimeout(timeout);
          if (!isDone && hasStartedReceiving) {
            resolve({
              output,
              audio: audioData
                ? {
                    data: audioData,
                    id: audioId,
                    expires_at: expiresAt,
                    transcript: output,
                  }
                : undefined,
              cached: false,
            });
          } else if (!isDone) {
            reject(
              new Error(`WebSocket closed unexpectedly (${code}${reason ? ': ' + reason : ''})`),
            );
          }
        });

        ws.on('error', (error) => {
          logger.error(`WebSocket error during response: ${error}`);
          clearTimeout(timeout);
          reject(error);
        });

        ws.on('message', (data: WebSocket.RawData) => {
          try {
            const event = JSON.parse(data.toString());
            logger.debug(`Received event: ${JSON.stringify(event)}`);
            hasStartedReceiving = true;

            if (event.type === 'error') {
              clearTimeout(timeout);
              reject(new Error(`API error: ${event.error.message}`));
              return;
            }

            switch (event.type) {
              case 'response.output.done':
              case 'response.completed':
              case 'done':
                isDone = true;
                clearTimeout(timeout);
                resolve({
                  output,
                  audio: audioData
                    ? {
                        data: audioData,
                        id: audioId,
                        expires_at: expiresAt,
                        transcript: output,
                      }
                    : undefined,
                  cached: false,
                });
                ws?.close();
                return;

              case 'response.output.text':
                output += event.text;
                break;

              case 'response.output.audio':
                audioData = event.audio;
                audioId = event.id;
                expiresAt = event.expires_at;
                break;

              case 'response.failed':
              case 'error':
                clearTimeout(timeout);
                reject(new Error(`Response failed: ${event.error?.message || 'Unknown error'}`));
                return;

              default:
                logger.debug(`Unknown event type: ${event.type}`);
            }
          } catch (err) {
            logger.error(`Error processing message: ${err}`);
            clearTimeout(timeout);
            reject(err);
          }
        });
      });
    } catch (err) {
      logger.error(`API call error: ${String(err)}`);
      return {
        error: `API call error: ${String(err)}`,
      };
    } finally {
      if (ws) {
        try {
          ws.close();
        } catch (err) {
          logger.error(`Error closing WebSocket: ${String(err)}`);
        }
      }
      this.ws = null;
    }
  }
}

// Update the default provider with valid voice and format options
export const DefaultAudioProvider = new OpenAiAudioProvider('gpt-4o-audio-preview', {
  config: {
    voice: 'alloy',
    format: 'wav',
  },
});

export const DefaultEmbeddingProvider = new OpenAiEmbeddingProvider('text-embedding-3-large');
export const DefaultGradingProvider = new OpenAiChatCompletionProvider('gpt-4o-2024-05-13');
export const DefaultGradingJsonProvider = new OpenAiChatCompletionProvider('gpt-4o-2024-05-13', {
  config: {
    response_format: { type: 'json_object' },
  },
});
export const DefaultSuggestionsProvider = new OpenAiChatCompletionProvider('gpt-4o-2024-05-13');
export const DefaultModerationProvider = new OpenAiModerationProvider('omni-moderation-latest');
