// The client wires and the three upstream wires, through one normalized shape.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. A provider speaks one of three
// wires (openai, anthropic, gemini). This file normalizes a client request once — system text, messages
// (text plus images), tool definitions, the output cap, stop sequences, JSON mode, streaming — builds any
// provider's request from it, reads any provider's answer (JSON or SSE) back into one shape, and renders
// that answer in the client's wire (JSON or SSE).
//
// What the provider echoes is what the client compares: the system text, the user text, the tool names,
// max_tokens, the stop sequences, the image (by the sha of its bytes) and JSON mode. Every conversion
// below preserves exactly those fields, in the spelling each wire's parser reads. The provider's answer
// text is returned whole, never rewritten: it carries the echo.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The provider's stream cuts its text into pieces this long; the renderers cut the same way. */
const SSE_DELTA_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Normalized request
// ---------------------------------------------------------------------------

export interface NormalizedImage {
  kind: 'base64' | 'url';
  mediaType: string;
  data: string;
  url: string;
}

export interface NormalizedMessage {
  role: 'user' | 'assistant';
  text: string;
  images: NormalizedImage[];
}

export interface NormalizedTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface NormalizedRequest {
  systemText: string;
  messages: NormalizedMessage[];
  tools: NormalizedTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  temperature: number | null;
  topP: number | null;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A string, or every string of a string array, joined the way the providers join stop lists. */
function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.flatMap((v) => (typeof v === 'string' ? [v] : []));
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** A `data:<mime>;base64,<data>` URL as its media type and data; null when it is not one. */
function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** The images of an OpenAI content array: image_url parts, data URL or plain URL. */
function openAIImages(content: unknown): NormalizedImage[] {
  if (!Array.isArray(content)) return [];
  const out: NormalizedImage[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
    const url = typeof block.image_url.url === 'string' ? block.image_url.url : '';
    if (url === '') continue;
    const data = parseDataUrl(url);
    if (data !== null) out.push({ kind: 'base64', mediaType: data.mediaType, data: data.data, url: '' });
    else out.push({ kind: 'url', mediaType: '', data: '', url });
  }
  return out;
}

/** The images of an Anthropic content array: image blocks, base64 or URL. */
function anthropicImages(content: unknown): NormalizedImage[] {
  if (!Array.isArray(content)) return [];
  const out: NormalizedImage[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      out.push({ kind: 'base64', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: '' });
    } else if (source.type === 'url' && typeof source.url === 'string') {
      out.push({ kind: 'url', mediaType: '', data: '', url: source.url });
    }
  }
  return out;
}

/** An OpenAI Chat Completions body as the normalized request. */
export function normalizeOpenAI(body: Json): NormalizedRequest {
  const systemParts: string[] = [];
  const messages: NormalizedMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role !== 'user' && raw.role !== 'assistant') continue;
    messages.push({ role: raw.role, text: textOf(raw.content), images: openAIImages(raw.content) });
  }
  const tools: NormalizedTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
  };
}

/** An Anthropic Messages body as the normalized request. */
export function normalizeAnthropic(body: Json): NormalizedRequest {
  const messages: NormalizedMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role !== 'user' && raw.role !== 'assistant') continue;
    messages.push({ role: raw.role, text: textOf(raw.content), images: anthropicImages(raw.content) });
  }
  const tools: NormalizedTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    systemText: textOf(body.system),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
  };
}

// ---------------------------------------------------------------------------
// Provider requests, built from the normalized request
// ---------------------------------------------------------------------------

/** One normalized image as an OpenAI image_url part. */
function openAIImagePart(image: NormalizedImage): Json {
  const url = image.kind === 'base64' ? `data:${image.mediaType};base64,${image.data}` : image.url;
  return { type: 'image_url', image_url: { url } };
}

/** One normalized image as an Anthropic image block. */
function anthropicImageBlock(image: NormalizedImage): Json {
  if (image.kind === 'base64') return { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } };
  return { type: 'image', source: { type: 'url', url: image.url } };
}

/** One normalized image as a Gemini part. */
function geminiImagePart(image: NormalizedImage): Json {
  if (image.kind === 'base64') return { inlineData: { mimeType: image.mediaType, data: image.data } };
  return { fileData: { fileUri: image.url } };
}

/** The normalized request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAIRequest(norm: NormalizedRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const message of norm.messages) {
    if (message.images.length === 0) {
      messages.push({ role: message.role, content: message.text });
      continue;
    }
    const parts: Json[] = [];
    if (message.text !== '') parts.push({ type: 'text', text: message.text });
    for (const image of message.images) parts.push(openAIImagePart(image));
    messages.push({ role: message.role, content: parts });
  }
  const out: Json = { model, messages };
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stream) {
    out.stream = true;
    // The client needs usage in the stream, and the provider only sends its usage chunk when asked.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The normalized request as an Anthropic Messages body naming `model`. */
export function buildAnthropicRequest(norm: NormalizedRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of norm.messages) {
    if (message.images.length === 0) {
      messages.push({ role: message.role, content: message.text });
      continue;
    }
    const blocks: Json[] = [];
    if (message.text !== '') blocks.push({ type: 'text', text: message.text });
    for (const image of message.images) blocks.push(anthropicImageBlock(image));
    messages.push({ role: message.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (norm.systemText !== '') out.system = norm.systemText;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stream) out.stream = true;
  return out;
}

/** The normalized request as a Gemini generateContent body. */
export function buildGeminiRequest(norm: NormalizedRequest): Json {
  const contents: Json[] = [];
  for (const message of norm.messages) {
    const parts: Json[] = [];
    if (message.text !== '') parts.push({ text: message.text });
    for (const image of message.images) parts.push(geminiImagePart(image));
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (norm.systemText !== '') out.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers, read back into one shape
// ---------------------------------------------------------------------------

/** A provider's answer, whole text and usage, in the client's terms. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason, in its own spelling. */
  stop: string;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOf(frame: SseFrame): Json | null {
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI Chat Completions answer (JSON or SSE) as one answer. */
export function parseOpenAIAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const answerText = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: answerText,
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    };
  }
  let answerText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOf(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') answerText += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: answerText, inputTokens, outputTokens, stop };
}

/** An Anthropic Messages answer (JSON or SSE) as one answer. */
export function parseAnthropicAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: textOf(value.content),
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
      stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
    };
  }
  let answerText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    const data = jsonOf(frame);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      answerText += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: answerText, inputTokens, outputTokens, stop };
}

/** A Gemini generateContent answer (JSON or SSE) as one answer. */
export function parseGeminiAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const answerText = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text: answerText,
      inputTokens: numberOrZero(meta.promptTokenCount),
      outputTokens: numberOrZero(meta.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    };
  }
  let answerText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  for (const frame of parseSseFrames(text)) {
    const data = jsonOf(frame);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') answerText += part.text;
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: answerText, inputTokens, outputTokens, stop };
}

// ---------------------------------------------------------------------------
// Client answers, rendered from one provider answer
// ---------------------------------------------------------------------------

/** A provider's stop reason as an OpenAI finish_reason. */
function openAIStopOf(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider's stop reason as an Anthropic stop_reason. */
function anthropicStopOf(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** One provider answer as an OpenAI Chat Completions object. */
export function renderOpenAIJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** One provider answer as an OpenAI event stream: role, text pieces, finish, usage, [DONE]. */
export function renderOpenAISse(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStopOf(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One provider answer as an Anthropic Messages object. */
export function renderAnthropicJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** One provider answer as the Anthropic event sequence: start, block, deltas, stop, delta, stop. */
export function renderAnthropicSse(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
