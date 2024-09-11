import packageJson from '../package.json';

export const VERSION = packageJson.version;

// We use process.env in this file because it's imported by the next.js server as well.

// This is used for sharing evals.
export const SHARE_API_BASE_URL =
  // TODO(ian): Backwards compatibility, 2024-04-01
  process.env.NEXT_PUBLIC_PROMPTFOO_REMOTE_API_BASE_URL ||
  process.env.NEXT_PUBLIC_PROMPTFOO_BASE_URL ||
  process.env.PROMPTFOO_REMOTE_API_BASE_URL ||
  `https://api.promptfoo.dev`;

// This is used for creating shared eval links.
export const SHARE_VIEW_BASE_URL =
  process.env.NEXT_PUBLIC_PROMPTFOO_BASE_URL ||
  process.env.PROMPTFOO_REMOTE_APP_BASE_URL ||
  `https://app.promptfoo.dev`;

// Maximum width for terminal outputs.
export const TERMINAL_MAX_WIDTH =
  process?.stdout?.columns && process?.stdout?.columns > 10 ? process?.stdout?.columns - 10 : 120;

export enum SERVER_MODE {
  OPEN = 'open',
  SAAS = 'saas',
}
