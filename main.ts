// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// It reads the brief (the provider directory, the quality table, the workload manifest), probes each
// provider's catalog route, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING (quality is 60% of the score, served 15%, wire 10%, failover 10%, catalog 5%)
//   Each request is routed to the best free model that meets its requirements, by the quality table's
//   rating_milli, with one best model per provider. The load is spread: a provider that has already
//   taken MAX_CALLS_PER_PROVIDER calls yields to a lower-rated provider that has not, because every
//   provider has a small per-trial budget (the smallest is 4) and a provider that is out of budget
//   takes every request routed to it down with it. A JSON-mode request on the OpenAI wire never rides
//   an anthropic-wire provider: that wire echoes json_mode false, and the echo would not match.
//   On a refusal (402, 429, 5xx) the request is retried on the next candidate. The refusal's message
//   names the kind: "budget is spent" and "is out" are permanent (that provider serves no more of this
//   trial), "refused this call" is transient (the provider may answer a later call number). A provider
//   that answers a refusal is never retried for the same request: the next call number may be another
//   refusal, and a different provider is the cheaper recovery. One refusal spends nothing at all: a
//   429 with the concurrent_requests code means the provider was busy serving another call, and its own
//   call count never moved — so it is not counted toward the load-spreading tally either, or the
//   provider would be capped a real call early for a call it never took.
//   One provider call answers each request: the first success is used, and no further provider is tried.
//
// THE WIRES (lib/wire.ts)
//   A client request is normalized once (system text, messages with text and images, tools, the output
//   cap, stop sequences, JSON mode, streaming) and any provider's request is built from it, in the
//   provider's own wire. A provider's answer (JSON or SSE, on any of the three wires) is read back into
//   one shape — the whole text, the usage counts, the stop reason — and rendered in the client's wire.
//   The provider's text is returned whole, never rewritten: it carries the echo the client compares.
//   An OpenAI stream asks the provider for its usage chunk (stream_options.include_usage), because the
//   client needs the usage in the stream and the provider only sends it when asked.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  buildAnthropicRequest,
  buildGeminiRequest,
  buildOpenAIRequest,
  normalizeAnthropic,
  normalizeOpenAI,
  openAIError,
  openAIModelList,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
  type NormalizedRequest,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most calls one provider takes as a first pick: every provider's per-trial budget is at least 4. */
const MAX_CALLS_PER_PROVIDER = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Calls made to each provider, refused ones included: a refusal spends a call number too. */
const providerCalls = new Map<string, number>();
/** Providers a permanent refusal has closed for this trial. */
const providerDead = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * One best free model per provider that meets the request, best first. A provider that has already
 * taken its share of calls yields to a lower-rated provider that has not (its budget is the scarcer
 * resource); within a share, the higher rating wins, then the fewer calls, then the provider id, so the
 * order is deterministic and the load spreads instead of piling onto one provider.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const byProvider = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (providerDead.has(probe.provider.id)) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) {
        best = { probe, model, rating };
      }
    }
    if (best !== null) byProvider.set(probe.provider.id, best);
  }
  const list = [...byProvider.values()];
  list.sort((a, b) => {
    const aCalls = providerCalls.get(a.probe.provider.id) ?? 0;
    const bCalls = providerCalls.get(b.probe.provider.id) ?? 0;
    const aOver = aCalls >= MAX_CALLS_PER_PROVIDER ? 1 : 0;
    const bOver = bCalls >= MAX_CALLS_PER_PROVIDER ? 1 : 0;
    if (aOver !== bOver) return aOver - bOver;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (aCalls !== bCalls) return aCalls - bCalls;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return list;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface CallOutcome {
  ok: boolean;
  answer: ProviderAnswer | null;
  /** True when the refusal is permanent: the provider serves no more of this trial. */
  permanent: boolean;
  status: number;
  message: string;
}

/** The error message and the error code of a provider's refusal body, in any of the three wires' shapes. */
function refusalDetail(text: string, wire: 'openai' | 'anthropic' | 'gemini'): { message: string; code: string } {
  const fallback = { message: text.slice(0, 200), code: '' };
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return fallback;
  }
  if (!isRecord(value) || !isRecord(value.error)) return fallback;
  const error = value.error;
  const message =
    typeof error.message === 'string'
      ? error.message
      : isRecord(error.error) && typeof error.error.message === 'string'
        ? error.error.message
        : text.slice(0, 200);
  // The code rides a different member on each wire: error.code on OpenAI, error.type on Anthropic
  // (its envelope's type carries the provider's code), error.status on Gemini.
  const code = wire === 'openai' ? error.code : wire === 'anthropic' ? error.type : error.status;
  return { message, code: typeof code === 'string' ? code : '' };
}

/** Whether a refusal's message names a permanent end: the budget spent, or an outage. */
function isPermanentRefusal(message: string): boolean {
  return message.includes('budget is spent') || message.includes('is out');
}

/**
 * One call to one provider, on its own wire, read back into one answer. A refusal is classified by
 * its message; a body that is not the wire's answer is a refusal too, so a bad answer never reaches
 * the client as a success.
 */
async function callProvider(candidate: Candidate, norm: NormalizedRequest): Promise<CallOutcome> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const key = process.env[provider.key_env] ?? provider.key;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, key),
  };
  if (norm.stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let url: string;
  let requestBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    requestBody = buildOpenAIRequest(norm, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    requestBody = buildAnthropicRequest(norm, candidate.model.id);
  } else {
    const action = norm.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    requestBody = buildGeminiRequest(norm);
  }
  providerCalls.set(provider.id, (providerCalls.get(provider.id) ?? 0) + 1);
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(requestBody), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, answer: null, permanent: false, status: 502, message: (err as Error).message };
  }
  const text = await upstream.text();
  if (!upstream.ok) {
    const { message, code } = refusalDetail(text, provider.wire);
    if (code === 'concurrent_requests') {
      // A provider that is busy refuses with 429 and the concurrent_requests code, and that refusal
      // spends none of its budget: its own call count never moved. Undo the increment this call made
      // before the fetch, so the load-spreading tally counts only calls the provider really took.
      providerCalls.set(provider.id, Math.max(0, (providerCalls.get(provider.id) ?? 0) - 1));
    }
    return { ok: false, answer: null, permanent: isPermanentRefusal(message), status: upstream.status, message };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, norm.stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, norm.stream)
        : parseGeminiAnswer(text, norm.stream);
  if (answer === null) {
    return { ok: false, answer: null, permanent: false, status: 502, message: `the ${provider.id} answer did not parse on the ${provider.wire} wire` };
  }
  return { ok: true, answer, permanent: false, status: 200, message: '' };
}

/** One client request, served by the best free model that meets it, failing over on a refusal. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const norm = clientApi === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, clientApi, norm.jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    if (providerDead.has(candidate.probe.provider.id)) continue;
    const outcome = await callProvider(candidate, norm);
    if (outcome.ok && outcome.answer !== null) {
      const answer = outcome.answer;
      log(`served ${clientApi}${norm.stream ? ' stream' : ''} on ${candidate.probe.provider.id}/${candidate.model.id} rating ${candidate.rating}`);
      if (clientApi === 'openai') {
        if (norm.stream) sendSse(res, renderOpenAISse(answer, candidate.model.id));
        else sendJson(res, 200, renderOpenAIJson(answer, candidate.model.id));
      } else {
        if (norm.stream) sendSse(res, renderAnthropicSse(answer, candidate.model.id));
        else sendJson(res, 200, renderAnthropicJson(answer, candidate.model.id));
      }
      return;
    }
    if (outcome.permanent) {
      providerDead.add(candidate.probe.provider.id);
      log(`provider ${candidate.probe.provider.id} is out: ${outcome.message}`);
    } else {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} refused (${outcome.status}): ${outcome.message}`);
    }
  }
  sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
