// The one thing MicroVerse cannot do in Python: mint a browser upload token.
//
// Vercel's Blob SDK can issue a credential scoped to a single pathname, so a browser
// can write one object and nothing else. That function exists only in the JavaScript
// SDK -- `vercel.blob` in Python can use a client token but not create one -- and
// without it a 64 MB abundance table has to travel through a function whose request
// body caps at 4.5 MB.
//
// So this file exists, and does nothing else. It decides nothing about who may
// upload, what a valid filename is, or how large a file may be. Those rules live in
// `app/uploads.py` and stay there: this route asks MicroVerse whether the signed
// ticket the browser presented really authorises the pathname it is asking for, and
// mints a token only if MicroVerse says yes. Adding a rule here would mean two
// places to keep in agreement, and one of them would eventually be wrong.
//
// BLOB_READ_WRITE_TOKEN is read here and never leaves: what reaches the browser is
// derived from it, scoped to one pathname, size-capped, and short-lived.

import { handleUpload } from '@vercel/blob/client';

const BACKEND = process.env.MICROVERSE_BACKEND_URL || '';
const SECRET = process.env.MICROVERSE_WORKER_SECRET || '';

/**
 * Ask MicroVerse whether this ticket authorises this pathname.
 * Returns the constraints to put on the token, or throws.
 */
async function authorise(pathname, clientPayload) {
  if (!SECRET) {
    // An unconfigured deployment must not be an open upload endpoint.
    throw new Error('Uploads are not configured on this deployment.');
  }
  const response = await fetch(`${BACKEND}/upload/ticket`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Microverse-Worker': SECRET,
    },
    body: JSON.stringify({ ticket: clientPayload, pathname }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || 'That upload is not authorised.');
  }
  return response.json();
}

export async function POST(request) {
  const body = await request.json();
  try {
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const granted = await authorise(pathname, clientPayload);
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
    return Response.json({ error: error.message }, { status: 400 });
  }
}

export default POST;
