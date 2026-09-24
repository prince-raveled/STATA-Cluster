// Executes api/blob-upload.js -- it does not read it.
//
// The route failed in production on every request, GET or POST, as
// FUNCTION_INVOCATION_FAILED, and the only tests it had inspected its source text.
// These call the real handlers with real Request objects and the real @vercel/blob
// SDK. The Blob control API is played by a local HTTP server, so every request the
// route makes leaves the process and comes back. MicroVerse is not played by anything:
// the route acts on grants it signed (app/grants.py), which these tests sign the same
// way. tests/test_serverless_flow.py runs the route on grants the Python side signed.
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
import { createHmac } from 'node:crypto';
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
const BUNDLE = `microverse/${TOKEN}/bundle.zip`;

// --- grants, signed as app/grants.py signs them --------------------------------
const b64url = (text) => Buffer.from(text).toString('base64url');

function keyFor({ secret = SECRET, store = RW_TOKEN } = {}) {
  const base = secret
    ? Buffer.from(secret, 'utf8')
    : createHmac('sha256', store).update('microverse/key/v1').digest();
  return createHmac('sha256', base).update('microverse/grant/v1').digest();
}

function sign(claims, key = keyFor()) {
  const body = b64url(JSON.stringify(claims));
  return `${body}.${createHmac('sha256', key).update(body).digest('base64url')}`;
}

const uploadGrant = (overrides = {}, key) => sign({
  use: 'upload',
  pathname: PATHNAME,
  maximum_size_in_bytes: 67108864,
  allowed_content_types: ['text/*', 'application/octet-stream'],
  valid_until: Date.now() + 60_000,
  ...overrides,
}, key);

const downloadGrant = (overrides = {}, key) => sign({
  use: 'download',
  pathname: BUNDLE,
  access: 'private',
  valid_until: Date.now() + 900_000,
  ...overrides,
}, key);

// --- one server playing the Blob control API ------------------------------------
const seen = [];

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

const blobApi = http.createServer(async (req, res) => {
  const body = await readJson(req);
  seen.push({ path: req.url, headers: req.headers, body });
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
    MICROVERSE_WORKER_SECRET: SECRET,
    BLOB_READ_WRITE_TOKEN: RW_TOKEN,
    VERCEL_BLOB_API_URL: base,
    MICROVERSE_BACKEND_URL: undefined,
    ...overrides,
  };
  for (const [name, value] of Object.entries(values)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
}

before(async () => {
  await new Promise((resolve) => blobApi.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${blobApi.address().port}`;
  configure();
  handler = await import(ENTRY);
});

after(() => new Promise((resolve) => blobApi.close(resolve)));

beforeEach(() => {
  seen.length = 0;
  configure();
});

const post = (body, raw = false) => handler.POST(new Request('https://example.test/api/blob-upload', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: raw ? body : JSON.stringify(body),
}));

const generate = ({ grant = uploadGrant(), pathname = PATHNAME } = {}) => post({
  type: 'blob.generate-client-token',
  payload: { pathname, clientPayload: grant, multipart: false },
});

const download = (query = `grant=${encodeURIComponent(downloadGrant())}`, route = '/api/blob-download') =>
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
  test('mints a token bound to the pathname and limits MicroVerse signed', async () => {
    const response = await generate();
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.type, 'blob.generate-client-token');

    const claims = decodeClientToken(body.clientToken);
    assert.equal(claims.pathname, PATHNAME);
    assert.equal(claims.maximumSizeInBytes, 67108864, 'the size limit comes from the grant');
    assert.deepEqual(claims.allowedContentTypes, ['text/*', 'application/octet-stream']);
    assert.equal(claims.addRandomSuffix, false);
    assert.equal(claims.allowOverwrite, true);
    assert.ok(claims.validUntil > Date.now());
    assert.equal(claims.onUploadCompleted, undefined, 'the store must not call the deployment back');
    assertNoSecretIn(JSON.stringify(body));
  });

  test('asks nothing of anyone to decide', async () => {
    await generate();
    assert.equal(seen.length, 0, 'minting an upload token must not call out');
  });

  test('works with only the store token configured', async () => {
    configure({ MICROVERSE_WORKER_SECRET: undefined });
    const response = await generate({ grant: uploadGrant({}, keyFor({ secret: '' })) });
    assert.equal(response.status, 200, await response.clone().text());
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

  for (const [label, grant] of [
    ['forged', 'not.a-grant'],
    ['empty', ''],
    ['three parts', `${uploadGrant()}.extra`],
    ['signed with another key', uploadGrant({}, keyFor({ secret: 'some-other-secret' }))],
    ['signed with the store-derived key while a secret is set', uploadGrant({}, keyFor({ secret: '' }))],
    ['expired', uploadGrant({ valid_until: Date.now() - 1 })],
    ['for downloading', downloadGrant({ pathname: PATHNAME })],
    ['with no pathname', uploadGrant({ pathname: '' })],
  ]) {
    test(`a grant that is ${label} mints nothing`, async () => {
      const response = await generate({ grant });
      assert.equal(response.status, 403);
      const body = await response.json();
      assert.equal(body.error, 'That upload has expired.');
      assert.equal(body.clientToken, undefined);
    });
  }

  test('a tampered grant mints nothing', async () => {
    const [body, signature] = uploadGrant().split('.');
    const claims = JSON.parse(Buffer.from(body, 'base64url').toString());
    claims.maximum_size_in_bytes = 1e12;
    const response = await generate({ grant: `${b64url(JSON.stringify(claims))}.${signature}` });
    assert.equal(response.status, 403);
  });

  test('a pathname the grant does not name mints nothing', async () => {
    const response = await generate({ pathname: 'microverse/elsewhere/abundance' });
    assert.equal(response.status, 403);
    assert.equal((await response.json()).error, 'That upload is not authorised for this location.');
  });

  test('a deployment without BLOB_READ_WRITE_TOKEN is a 500 that names it', async () => {
    configure({ BLOB_READ_WRITE_TOKEN: undefined });
    const response = await generate();
    assert.equal(response.status, 500);
    const text = JSON.stringify(await response.json());
    assert.ok(text.includes('BLOB_READ_WRITE_TOKEN'), text);
    assertNoSecretIn(text);
    assert.equal(seen.length, 0, 'an unconfigured deployment must ask nothing and mint nothing');
  });

  test('MICROVERSE_BACKEND_URL is no longer needed', async () => {
    configure({ MICROVERSE_BACKEND_URL: undefined });
    assert.equal((await generate()).status, 200);
  });
});

// --- download: signing one read ----------------------------------------------------
describe('GET /api/blob-download', () => {
  test('redirects to a signed URL for exactly the granted object', async () => {
    const response = await download();
    assert.equal(response.status, 302);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    const location = new URL(response.headers.get('location'));
    assert.equal(location.host, `${STORE}.private.blob.vercel-storage.com`);
    assert.equal(location.pathname, `/${BUNDLE}`);
    assert.equal(location.searchParams.get('cache'), '0', 'a rerun must not be served stale');
    assert.ok([...location.searchParams.keys()].length > 1, 'the URL carries no signature');

    const signed = seen.find((r) => r.path.startsWith('/signed-token'));
    assert.equal(signed.body.pathname, BUNDLE);
    assert.deepEqual(signed.body.operations, ['get'], 'a download token must only read');
    assertNoSecretIn(location.toString());
  });

  for (const [label, grant] of [
    ['missing', ''],
    ['forged', 'not.a-grant'],
    ['expired', downloadGrant({ valid_until: Date.now() - 1 })],
    ['for uploading', uploadGrant({ pathname: BUNDLE })],
    ['signed with another key', downloadGrant({}, keyFor({ secret: 'some-other-secret' }))],
  ]) {
    test(`a grant that is ${label} signs nothing`, async () => {
      const response = await download(`grant=${encodeURIComponent(grant)}`);
      assert.equal(response.status, 403);
      assert.ok(!seen.some((r) => r.path.startsWith('/signed-token')), 'nothing may be signed');
    });
  }

  test('the old token-and-name link signs nothing', async () => {
    const response = await download(`token=${TOKEN}&name=bundle.zip`);
    assert.equal(response.status, 403);
  });

  test('the store refusing to sign is a gateway failure', async () => {
    const other = `vercel_blob_rw_${STORE}_adifferenttoken`;
    configure({ BLOB_READ_WRITE_TOKEN: other });
    const response = await download(`grant=${encodeURIComponent(downloadGrant())}`);
    assert.equal(response.status, 502);
  });

  test('an unconfigured deployment signs nothing', async () => {
    configure({ BLOB_READ_WRITE_TOKEN: undefined });
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
        payload: { pathname: PATHNAME, clientPayload: uploadGrant(), multipart: false },
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
    const response = await fetch(`${url}/api/blob-download?grant=${encodeURIComponent(downloadGrant())}`,
                                 { redirect: 'manual' });
    assert.equal(response.status, 302);
    assert.ok(response.headers.get('location').includes('.private.blob.vercel-storage.com/'));
  });

  test('a method with no handler is 405, the launcher\'s own answer', async () => {
    const response = await fetch(`${url}/api/blob-upload`, { method: 'PUT' });
    assert.equal(response.status, 405);
  });
});
