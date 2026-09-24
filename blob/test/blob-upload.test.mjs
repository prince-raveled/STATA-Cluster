// Executes api/blob-upload.js -- it does not read it.
//
// The route failed in production on every request, GET or POST, as
// FUNCTION_INVOCATION_FAILED, and the only tests it had inspected its source text.
// These call the real handlers with real Request objects and the real @vercel/blob
// SDK. MicroVerse and the Blob control API are played by one local HTTP server, so
// every request the route makes leaves the process and comes back.
//
// The last group loads the file through Vercel's own Node launcher
// (@vercel/node dist/bundling-handler.js), which is where the fault actually was: it
// replaces a module with its default export, and ran `export default POST` as a
// Node (req, res) handler. Point VERCEL_NODE_LAUNCHER at that file to run it; CI does,
// and sets MICROVERSE_REQUIRE_LAUNCHER=1 so a missing launcher fails instead of skipping.
//
//     cd blob && npm test

import assert from 'node:assert/strict';
import { execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import http from 'node:http';
import { createRequire } from 'node:module';
import path from 'node:path';
import { after, before, beforeEach, describe, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SERVICE_ROOT = path.resolve(HERE, '..');
const ENTRY = pathToFileURL(path.join(SERVICE_ROOT, 'api', 'blob-upload.js')).href;

// Stand-in values, not credentials: nothing here can reach a real store.
const SECRET = 'worker-secret-for-tests';
const STORE = 'teststore';
const RW_TOKEN = `vercel_blob_rw_${STORE}_notarealsecret`;
const TOKEN = '0123456789abcdef01234567';
const PATHNAME = `microverse/${TOKEN}/abundance`;
const GOOD_TICKET = 'a-ticket-microverse-signed';

// --- one server playing MicroVerse and the Blob control API ------------------
const seen = [];
let behaviour = {};

function readJson(req) {
  return new Promise((resolve) => {
    let raw = '';
    req.on('data', (chunk) => { raw += chunk; });
    req.on('end', () => {
      try { resolve(JSON.parse(raw || 'null')); } catch { resolve(undefined); }
    });
  });
}

function reply(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

const b64url = (text) => Buffer.from(text).toString('base64url');

const backend = http.createServer(async (req, res) => {
  const body = await readJson(req);
  seen.push({ path: req.url, headers: req.headers, body });
  const forced = behaviour[req.url];
  if (forced) return reply(res, forced.status, forced.body);

  if (req.url === '/upload/ticket' || req.url === '/download/grant') {
    if (req.headers['x-microverse-worker'] !== SECRET) {
      return reply(res, 403, { error: 'Not authorised.' });
    }
  }
  if (req.url === '/upload/ticket') {
    if (body.ticket !== GOOD_TICKET || body.pathname !== PATHNAME) {
      return reply(res, 403, { error: 'That upload has expired.' });
    }
    return reply(res, 200, {
      maximum_size_in_bytes: 67108864,
      allowed_content_types: ['text/*', 'application/octet-stream'],
      valid_until: Date.now() + 60_000,
    });
  }
  if (req.url === '/download/grant') {
    if (body.token !== TOKEN || body.name !== 'bundle.zip') {
      return reply(res, 404, { error: 'There is no such download.' });
    }
    return reply(res, 200, {
      pathname: `microverse/${TOKEN}/bundle.zip`,
      access: 'private',
      valid_until: Date.now() + 900_000,
    });
  }
  if (req.url.startsWith('/signed-token')) {
    if (req.headers.authorization !== `Bearer ${RW_TOKEN}`) {
      return reply(res, 403, { error: { code: 'forbidden', message: 'bad token' } });
    }
    const scope = { storeId: STORE, pathname: body.pathname, operations: body.operations,
                    validUntil: body.validUntil };
    return reply(res, 200, {
      delegationToken: `${b64url(JSON.stringify(scope))}.fake-delegation-signature`,
      clientSigningToken: 'fake-client-signing-token',
      validUntil: body.validUntil,
    });
  }
  return reply(res, 404, { error: 'unexpected route in test' });
});

let base;
let handler;

function configure(overrides = {}) {
  const values = {
    MICROVERSE_BACKEND_URL: base,
    MICROVERSE_WORKER_SECRET: SECRET,
    BLOB_READ_WRITE_TOKEN: RW_TOKEN,
    VERCEL_BLOB_API_URL: base,
    ...overrides,
  };
  for (const [name, value] of Object.entries(values)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
}

before(async () => {
  await new Promise((resolve) => backend.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${backend.address().port}`;
  configure();
  handler = await import(ENTRY);
});

after(() => new Promise((resolve) => backend.close(resolve)));

beforeEach(() => {
  seen.length = 0;
  behaviour = {};
  configure();
});

const post = (body, raw = false) => handler.POST(new Request('https://example.test/api/blob-upload', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: raw ? body : JSON.stringify(body),
}));

const generate = (overrides = {}) => post({
  type: 'blob.generate-client-token',
  payload: { pathname: PATHNAME, clientPayload: GOOD_TICKET, multipart: false },
  ...overrides,
});

const download = (query = `token=${TOKEN}&name=bundle.zip`, route = '/api/blob-download') =>
  handler.GET(new Request(`https://example.test${route}?${query}`));

function decodeClientToken(clientToken) {
  const prefix = `vercel_blob_client_${STORE}_`;
  assert.ok(clientToken.startsWith(prefix), clientToken);
  const inner = Buffer.from(clientToken.slice(prefix.length), 'base64').toString();
  const payload = inner.slice(inner.indexOf('.') + 1);
  return JSON.parse(Buffer.from(payload, 'base64').toString());
}

function assertNoSecretIn(text) {
  assert.ok(!text.includes(SECRET), 'the worker secret reached a response');
  assert.ok(!text.includes('notarealsecret'), 'the store token reached a response');
}

// --- the module's shape ---------------------------------------------------------
describe('module shape', () => {
  test('has no default export, so the launcher cannot mistake it for (req, res)', async () => {
    const mod = await import(ENTRY);
    assert.equal(mod.default, undefined);
    assert.equal(typeof mod.POST, 'function');
    assert.equal(typeof mod.GET, 'function');
  });
});

// --- upload: minting a client token ---------------------------------------------
describe('POST /api/blob-upload', () => {
  test('mints a token bound to the pathname MicroVerse approved', async () => {
    const response = await generate();
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.type, 'blob.generate-client-token');

    const claims = decodeClientToken(body.clientToken);
    assert.equal(claims.pathname, PATHNAME);
    assert.equal(claims.maximumSizeInBytes, 67108864, 'the size limit comes from MicroVerse');
    assert.deepEqual(claims.allowedContentTypes, ['text/*', 'application/octet-stream']);
    assert.equal(claims.addRandomSuffix, false);
    assert.equal(claims.allowOverwrite, true);
    assert.ok(claims.validUntil > Date.now());

    const asked = seen.find((r) => r.path === '/upload/ticket');
    assert.equal(asked.headers['x-microverse-worker'], SECRET);
    assert.deepEqual(asked.body, { ticket: GOOD_TICKET, pathname: PATHNAME });
    assertNoSecretIn(JSON.stringify(body));
  });

  test('a body that is not JSON is 400, not a crash', async () => {
    const response = await post('{not json', true);
    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: 'Malformed request.' });
  });

  for (const [label, body] of [['null', null], ['an array', []], ['no type', { payload: {} }],
                               ['no payload', { type: 'blob.generate-client-token' }]]) {
    test(`a malformed event (${label}) is 400`, async () => {
      const response = await post(body);
      assert.equal(response.status, 400);
      assert.equal(seen.length, 0, 'nothing should be asked of MicroVerse');
    });
  }

  test('an unknown event type is 400', async () => {
    const response = await post({ type: 'blob.something-else', payload: {} });
    assert.equal(response.status, 400);
  });

  test('a completion callback without a signature is refused', async () => {
    const response = await post({ type: 'blob.upload-completed', payload: { blob: {} } });
    assert.equal(response.status, 400);
  });

  test('a ticket MicroVerse refuses mints nothing', async () => {
    const response = await generate({
      payload: { pathname: PATHNAME, clientPayload: 'forged', multipart: false } });
    assert.equal(response.status, 403);
    const body = await response.json();
    assert.equal(body.error, 'That upload has expired.');
    assert.equal(body.clientToken, undefined);
  });

  test('a pathname MicroVerse did not approve mints nothing', async () => {
    const response = await generate({
      payload: { pathname: 'microverse/elsewhere/abundance', clientPayload: GOOD_TICKET,
                 multipart: false } });
    assert.equal(response.status, 403);
  });

  test('a secret MicroVerse does not share is refused', async () => {
    configure({ MICROVERSE_WORKER_SECRET: 'some-other-secret' });
    const response = await generate();
    assert.equal(response.status, 403);
  });

  test('an unreachable MicroVerse is a gateway failure, not the caller\'s', async () => {
    configure({ MICROVERSE_BACKEND_URL: 'http://127.0.0.1:9' });
    const response = await generate();
    assert.equal(response.status, 502);
  });

  test('MicroVerse failing is a gateway failure too', async () => {
    behaviour['/upload/ticket'] = { status: 500, body: { error: 'boom' } };
    const response = await generate();
    assert.equal(response.status, 502);
  });

  for (const missing of ['MICROVERSE_WORKER_SECRET', 'MICROVERSE_BACKEND_URL',
                         'BLOB_READ_WRITE_TOKEN']) {
    test(`a deployment without ${missing} is a 500 that names it`, async () => {
      configure({ [missing]: undefined });
      const response = await generate();
      assert.equal(response.status, 500);
      const text = JSON.stringify(await response.json());
      assert.ok(text.includes(missing), text);
      assertNoSecretIn(text);
      assert.equal(seen.length, 0, 'an unconfigured deployment must ask nothing and mint nothing');
    });
  }
});

// --- download: signing one read ----------------------------------------------------
describe('GET /api/blob-download', () => {
  test('redirects to a signed URL for exactly the granted object', async () => {
    const response = await download();
    assert.equal(response.status, 302);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    const location = new URL(response.headers.get('location'));
    assert.equal(location.host, `${STORE}.private.blob.vercel-storage.com`);
    assert.equal(location.pathname, `/microverse/${TOKEN}/bundle.zip`);
    assert.equal(location.searchParams.get('cache'), '0', 'a rerun must not be served stale');
    assert.ok([...location.searchParams.keys()].length > 1, 'the URL carries no signature');

    const grant = seen.find((r) => r.path === '/download/grant');
    assert.equal(grant.headers['x-microverse-worker'], SECRET);
    assert.deepEqual(grant.body, { token: TOKEN, name: 'bundle.zip' });
    const signed = seen.find((r) => r.path.startsWith('/signed-token'));
    assert.equal(signed.body.pathname, `microverse/${TOKEN}/bundle.zip`);
    assert.deepEqual(signed.body.operations, ['get'], 'a download token must only read');
    assertNoSecretIn(location.toString());
  });

  test('what MicroVerse refuses is refused with its status', async () => {
    for (const [status, error] of [[404, 'That job no longer exists.'],
                                   [409, 'That run has not finished yet.'],
                                   [403, 'Not authorised.']]) {
      behaviour['/download/grant'] = { status, body: { error } };
      const response = await download();
      assert.equal(response.status, status);
      assert.equal((await response.json()).error, error);
    }
    assert.ok(!seen.some((r) => r.path.startsWith('/signed-token')), 'nothing may be signed');
  });

  test('an unknown download is passed through as 404', async () => {
    const response = await download(`token=${TOKEN}&name=run.pkl.gz`);
    assert.equal(response.status, 404);
  });

  test('the store refusing to sign is a gateway failure', async () => {
    configure({ BLOB_READ_WRITE_TOKEN: `vercel_blob_rw_${STORE}_adifferenttoken` });
    const response = await download();
    assert.equal(response.status, 502);
  });

  test('an unconfigured deployment signs nothing', async () => {
    configure({ MICROVERSE_WORKER_SECRET: undefined });
    const response = await download();
    assert.equal(response.status, 500);
    assert.equal(seen.length, 0);
  });

  test('GET on any other path is 404', async () => {
    const response = await download('', '/api/blob-upload');
    assert.equal(response.status, 404);
  });
});

// --- the way Vercel actually invokes it ---------------------------------------------
function findLauncher() {
  if (process.env.VERCEL_NODE_LAUNCHER) return process.env.VERCEL_NODE_LAUNCHER;
  try {
    const globalRoot = execSync('npm root -g', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    const candidate = path.join(globalRoot, 'vercel', 'node_modules', '@vercel', 'node',
                                'dist', 'bundling-handler.js');
    return existsSync(candidate) ? candidate : null;
  } catch {
    return null;
  }
}

const launcherPath = findLauncher();
const requireLauncher = process.env.MICROVERSE_REQUIRE_LAUNCHER === '1';

describe('through Vercel\'s Node launcher', { skip: !launcherPath && !requireLauncher
  ? 'set VERCEL_NODE_LAUNCHER to @vercel/node/dist/bundling-handler.js' : false }, () => {
  let server;
  let url;

  before(async () => {
    assert.ok(launcherPath && existsSync(launcherPath),
              `the launcher is required but was not found (${launcherPath})`);
    // The launcher resolves entrypoints against the working directory, as the
    // function's own file tree is on Vercel.
    process.chdir(SERVICE_ROOT);
    const launch = createRequire(import.meta.url)(launcherPath);
    server = http.createServer((req, res) => {
      req.headers['x-matched-path'] = '/api/blob-upload';
      launch(req, res).catch((error) => {
        // On Vercel an invocation that ends like this is FUNCTION_INVOCATION_FAILED.
        res.statusCode = 599;
        res.end(`crashed: ${error && error.message}`);
      });
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    url = `http://127.0.0.1:${server.address().port}`;
  });

  after(() => server && new Promise((resolve) => server.close(resolve)));

  test('a POST is handled, not crashed', async () => {
    const response = await fetch(`${url}/api/blob-upload`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        type: 'blob.generate-client-token',
        payload: { pathname: PATHNAME, clientPayload: GOOD_TICKET, multipart: false },
      }),
    });
    assert.equal(response.status, 200, await response.clone().text());
    assert.ok((await response.json()).clientToken);
  });

  test('a malformed POST is 400 through the launcher too', async () => {
    const response = await fetch(`${url}/api/blob-upload`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: '{not json',
    });
    assert.equal(response.status, 400);
  });

  test('a download redirect survives the launcher', async () => {
    const response = await fetch(`${url}/api/blob-download?token=${TOKEN}&name=bundle.zip`,
                                 { redirect: 'manual' });
    assert.equal(response.status, 302);
    assert.ok(response.headers.get('location').includes('.private.blob.vercel-storage.com/'));
  });

  test('a method with no handler is 405, the launcher\'s own answer', async () => {
    const response = await fetch(`${url}/api/blob-upload`, { method: 'PUT' });
    assert.equal(response.status, 405);
  });
});
