// The things MicroVerse cannot do in Python: sign for the Blob store.
//
// Vercel's Blob SDK can issue a credential scoped to a single pathname, so a browser
// can write one object and nothing else, and it can sign a short-lived URL that reads
// one private object and nothing else. Both exist only in the JavaScript SDK --
// `vercel.blob` in Python can use a client token but not create one, and has no URL
// signing at all -- and without them a 64 MB abundance table has to travel through a
// function whose request body caps at 4.5 MB, and a 10 MB results bundle through one
// whose response does.
//
// So this file exists, and does nothing else. It decides nothing about who may
// upload or download, what a valid filename is, or how large a file may be. Those
// rules live in `app/uploads.py` and `app/routers/results.py` and stay there:
// MicroVerse writes its decision into a signed grant (`app/grants.py`), and this route
// checks the signature and signs exactly what the grant names, under the limits the
// grant carries. Adding a rule here would mean two places to keep in agreement, and one
// of them would eventually be wrong.
//
// The grant travels with the request rather than being fetched from MicroVerse. An
// earlier version asked FastAPI over HTTP, which needed the application's public URL
// and a second shared secret on every deployment, and on a preview deployment behind
// Vercel's login the call to the deployment's own URL was stopped at the login wall.
//
// BLOB_READ_WRITE_TOKEN is read here and never leaves: what reaches the browser is
// derived from it, scoped to one pathname, and short-lived. It is also where the grant
// key comes from when MICROVERSE_WORKER_SECRET is not set, derived exactly as
// app/grants.py derives it.
//
// Named method exports only -- there must be no default export. Vercel's Node
// launcher replaces a module with its default export when it has one, and a default
// export that is a function is then invoked as a Node `(req, res)` handler, with an
// IncomingMessage that has no `.json()`. That is how every request to this route,
// GET or POST, ended as FUNCTION_INVOCATION_FAILED. blob/test/ loads this file
// through the launcher itself so the shape cannot regress.

import { createHmac, timingSafeEqual } from 'node:crypto';

import { BlobError, issueSignedToken, presignUrl } from '@vercel/blob';
import { handleUpload } from '@vercel/blob/client';

// Must match app/grants.py. Changing either invalidates every grant in flight.
const BASE_LABEL = 'microverse/key/v1';
const GRANT_LABEL = 'microverse/grant/v1';

/** An answer with a status, as opposed to an error nobody anticipated. */
class Refusal extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/**
 * The key grants are signed with, read per request.
 *
 * The store token is the one setting this route cannot work without. Missing it is
 * the deployment's fault, never the caller's, so it is a 500 that names it -- the
 * name only, never a value.
 */
function grantKey() {
  const store = process.env.BLOB_READ_WRITE_TOKEN || '';
  if (!store) {
    // An unconfigured deployment must not be an open signing endpoint.
    throw new Refusal(500, 'Blob access is not configured on this deployment (missing: BLOB_READ_WRITE_TOKEN).');
  }
  const shared = process.env.MICROVERSE_WORKER_SECRET || '';
  const base = shared
    ? Buffer.from(shared, 'utf8')
    : createHmac('sha256', store).update(BASE_LABEL).digest();
  return createHmac('sha256', base).update(GRANT_LABEL).digest();
}

/**
 * The claims of a grant MicroVerse signed for `use`, or null.
 *
 * Null for anything forged, malformed, meant for the other use, or expired: the
 * caller refuses all of them the same way, because none of them is an approval.
 */
function approved(raw, use) {
  const key = grantKey();
  const [body, signature, ...rest] = String(raw || '').split('.');
  if (!body || !signature || rest.length) return null;
  const expected = createHmac('sha256', key).update(body).digest();
  const given = Buffer.from(signature, 'base64url');
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) return null;
  let claims;
  try {
    claims = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
  } catch {
    return null;
  }
  if (!claims || typeof claims !== 'object' || claims.use !== use) return null;
  if (typeof claims.pathname !== 'string' || !claims.pathname) return null;
  if (!(Number(claims.valid_until) > Date.now())) return null;
  return claims;
}

function refuse(error) {
  if (error instanceof Refusal) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  // Not a refusal anyone wrote, so a fault in this route rather than in the request.
  console.error('[blob] unexpected failure:', error && error.name, error && error.message);
  return Response.json({ error: 'The Blob store could not complete this request.' }, { status: 500 });
}

/** Upload: mint a client token for one pathname MicroVerse approved. */
export async function POST(request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'Malformed request.' }, { status: 400 });
  }
  if (!body || typeof body !== 'object' || typeof body.type !== 'string'
      || !body.payload || typeof body.payload !== 'object') {
    return Response.json({ error: 'Malformed request.' }, { status: 400 });
  }

  try {
    grantKey();
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const grant = approved(clientPayload, 'upload');
        if (!grant) {
          throw new Refusal(403, 'That upload has expired.');
        }
        // `upload()` sends whatever pathname the browser passed it, so it is checked
        // against the one MicroVerse signed rather than trusted -- otherwise a valid
        // grant would authorise writing anywhere in the store.
        if (grant.pathname !== pathname) {
          throw new Refusal(403, 'That upload is not authorised for this location.');
        }
        return {
          // A random suffix would move the object somewhere neither side can find.
          addRandomSuffix: false,
          allowOverwrite: true,
          // The application's own limits, enforced by the store itself rather than
          // trusted from the browser's declaration.
          maximumSizeInBytes: grant.maximum_size_in_bytes,
          allowedContentTypes: grant.allowed_content_types,
          validUntil: grant.valid_until,
          tokenPayload: JSON.stringify({ pathname }),
        };
      },
      // No onUploadCompleted, deliberately. MicroVerse reads the object when the
      // browser calls /upload/complete, which is also what validates it and creates
      // the job, so there is nothing for a completion callback to do -- and giving one
      // makes the store call this deployment back, which on a preview behind Vercel's
      // login is stopped at the login wall.
    });
    return Response.json(result);
  } catch (error) {
    if (error instanceof Refusal) {
      return refuse(error);
    }
    if (error instanceof BlobError) {
      // Configuration was checked above, so what the SDK rejects here is the event
      // itself: an unknown type, or a completion callback without a valid signature.
      // (Its errors are named plain "Error"; the class is the only reliable mark.)
      return Response.json({ error: error.message }, { status: 400 });
    }
    return refuse(error);
  }
}

/** Download: redirect to a short-lived URL for one object MicroVerse approved. */
export async function GET(request) {
  const url = new URL(request.url);
  if (!url.pathname.endsWith('/api/blob-download')) {
    return Response.json({ error: 'Not found.' }, { status: 404 });
  }
  try {
    const grant = approved(url.searchParams.get('grant'), 'download');
    if (!grant) {
      throw new Refusal(403, 'That download link has expired. Open the results page and download again.');
    }
    let presignedUrl;
    try {
      const signed = await issueSignedToken({
        pathname: grant.pathname,
        operations: ['get'],
        validUntil: grant.valid_until,
      });
      ({ presignedUrl } = await presignUrl(signed, {
        operation: 'get',
        pathname: grant.pathname,
        access: grant.access,
        validUntil: grant.valid_until,
        // A rerun rewrites the same pathname; the CDN may hold the old copy for a
        // minute, and a results file must never be the previous run's.
        useCache: false,
      }));
    } catch (error) {
      console.error('[blob] signing failed:', error && error.name, error && error.message);
      throw new Refusal(502, 'The Blob store could not sign this download.');
    }
    return new Response(null, {
      status: 302,
      headers: { Location: presignedUrl, 'Cache-Control': 'no-store' },
    });
  } catch (error) {
    return refuse(error);
  }
}
