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
// rules live in `app/uploads.py` and `app/routers/results.py` and stay there: this
// route asks MicroVerse whether a request is allowed, and signs only what MicroVerse
// approved. Adding a rule here would mean two places to keep in agreement, and one of
// them would eventually be wrong.
//
// BLOB_READ_WRITE_TOKEN is read here and never leaves: what reaches the browser is
// derived from it, scoped to one pathname, and short-lived.
//
// Named method exports only -- there must be no default export. Vercel's Node
// launcher replaces a module with its default export when it has one, and a default
// export that is a function is then invoked as a Node `(req, res)` handler, with an
// IncomingMessage that has no `.json()`. That is how every request to this route,
// GET or POST, ended as FUNCTION_INVOCATION_FAILED. blob/test/ loads this file
// through the launcher itself so the shape cannot regress.

import { BlobError, issueSignedToken, presignUrl } from '@vercel/blob';
import { handleUpload } from '@vercel/blob/client';

/** An answer with a status, as opposed to an error nobody anticipated. */
class Refusal extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/**
 * The three settings this route cannot work without, read per request.
 *
 * Missing any of them is the deployment's fault, never the caller's, so it is a 500
 * that names what is missing -- names only, never a value.
 */
function settings() {
  const env = {
    MICROVERSE_BACKEND_URL: (process.env.MICROVERSE_BACKEND_URL || '').replace(/\/+$/, ''),
    MICROVERSE_WORKER_SECRET: process.env.MICROVERSE_WORKER_SECRET || '',
    BLOB_READ_WRITE_TOKEN: process.env.BLOB_READ_WRITE_TOKEN || '',
  };
  const missing = Object.keys(env).filter((name) => !env[name]);
  if (missing.length) {
    // An unconfigured deployment must not be an open signing endpoint.
    throw new Refusal(500, `Blob access is not configured on this deployment (missing: ${missing.join(', ')}).`);
  }
  return { backend: env.MICROVERSE_BACKEND_URL, secret: env.MICROVERSE_WORKER_SECRET };
}

/**
 * Ask MicroVerse whether a request is allowed. Returns its answer, or throws.
 *
 * MicroVerse refusing (403, 404, 409) is passed on as it was given; anything else
 * going wrong between the two services is a gateway failure, not the caller's.
 */
async function ask(route, body) {
  const { backend, secret } = settings();
  let response;
  try {
    response = await fetch(`${backend}${route}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Microverse-Worker': secret,
      },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Refusal(502, 'MicroVerse could not be reached to authorise this request.');
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const status = [403, 404, 409].includes(response.status) ? response.status : 502;
    throw new Refusal(status, payload.error || 'That request is not authorised.');
  }
  return payload;
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
    settings();
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const granted = await ask('/upload/ticket', { ticket: clientPayload, pathname });
        return {
          // The pathname the token is bound to is the one MicroVerse approved, and
          // a random suffix would move it somewhere neither side can find again.
          addRandomSuffix: false,
          allowOverwrite: true,
          // The application's own 64 MB limit, enforced by the store itself rather
          // than trusted from the browser's declaration.
          maximumSizeInBytes: granted.maximum_size_in_bytes,
          allowedContentTypes: granted.allowed_content_types,
          validUntil: granted.valid_until,
          tokenPayload: JSON.stringify({ pathname }),
        };
      },
      onUploadCompleted: async () => {
        // Nothing to do. MicroVerse reads the object when the browser calls
        // /upload/complete, which is also what validates it and creates the job,
        // so there is no state here that a missed callback could leave behind.
      },
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
    const grant = await ask('/download/grant', {
      token: url.searchParams.get('token') || '',
      name: url.searchParams.get('name') || '',
    });
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
