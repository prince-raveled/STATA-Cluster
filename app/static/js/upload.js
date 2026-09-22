/* Send the files to storage instead of through the application.

   The form this enhances works on its own. Without JavaScript, with an old browser,
   or if any step here fails before bytes are committed, the browser posts the form
   the way it always has and the server answers the way it always has. Nothing below
   changes what is uploaded, what is accepted, or what any error says — it changes
   only the route the bytes take, because some hosts cap a request body well below
   MicroVerse's upload limit.

   Progress is reported on the existing button rather than in new furniture. */
(function () {
  "use strict";

  var form = document.querySelector('form[action="/upload"]');
  if (!form || !window.fetch || !window.FormData || !window.AbortController) return;

  var button = form.querySelector('button[type="submit"]');
  var original = button ? button.innerHTML : "";
  var inFlight = null;
  var ticket = null;

  function field(name) {
    return form.querySelector('input[type="file"][name="' + name + '"]');
  }

  function chosen() {
    var files = {};
    ["abundance", "metadata", "taxonomy"].forEach(function (name) {
      var input = field(name);
      if (input && input.files && input.files[0]) files[name] = input.files[0];
    });
    return files;
  }

  function say(text) {
    if (button) button.innerHTML = text;
  }

  function restore() {
    if (button) {
      button.innerHTML = original;
      button.disabled = false;
    }
  }

  /* The server's own words. Never invented here, so a rejected file reads exactly
     as it does on the error page. */
  function showError(payload) {
    var box = document.getElementById("upload-error");
    if (!box) {
      box = document.createElement("div");
      box.id = "upload-error";
      box.className = "alert";          // the same style a rejected dataset already uses
      box.style.margin = "0 0 20px";
      box.setAttribute("role", "alert");
      form.insertBefore(box, form.firstElementChild.nextSibling);
    }
    var message = (payload && payload.error) || "That upload could not be completed.";
    var hint = (payload && payload.hint) || "";
    box.textContent = "";
    var strong = document.createElement("strong");
    strong.textContent = message;
    box.appendChild(strong);
    if (hint) {
      var p = document.createElement("p");
      p.className = "muted";
      p.style.margin = "6px 0 0";
      p.textContent = hint;
      box.appendChild(p);
    }
    box.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function clearError() {
    var box = document.getElementById("upload-error");
    if (box) box.remove();
  }

  function asJson(response) {
    return response.json().catch(function () { return {}; });
  }

  /* Two ways to put one file where the server said it may go.

     `staged` is this application taking the bytes itself, which is what a host with
     a disk does and what the test suite exercises. `vercel-blob` is the browser
     writing straight to object storage, for hosts that cap a request body below
     MicroVerse's upload limit — the whole reason any of this exists.

     Only the second needs the Blob SDK, so it is imported at the moment it is
     needed rather than on every page load, and a failure to load it lands in the
     same fallback as any other pre-commit failure. */
  function toStagingRoute(target, issued, file) {
    var headers = Object.assign({}, target.headers || {});
    headers["X-Microverse-Ticket"] = issued;
    return fetch(target.url, {
      method: target.method || "PUT",
      headers: headers,
      body: file,
      signal: inFlight.signal,
    }).then(function (response) {
      if (response.ok) return;
      return asJson(response).then(function (payload) {
        throw { handled: true, payload: payload };
      });
    });
  }

  var blobClient = null;
  function loadBlobClient() {
    if (!blobClient) {
      blobClient = import(
        "https://cdn.jsdelivr.net/npm/@vercel/blob@2.8.0/client/+esm"
      );
    }
    return blobClient;
  }

  function toBlobStore(target, issued, file) {
    return loadBlobClient().then(function (client) {
      // The pathname is the server's, not ours, and /upload/ticket checks it again
      // against the signed ticket before any token is issued.
      return client.upload(target.pathname, file, {
        access: target.access || "private",
        handleUploadUrl: target.handler,
        clientPayload: issued,
        contentType: file.type || "application/octet-stream",
        abortSignal: inFlight.signal,
        onUploadProgress: function (progress) {
          if (progress && typeof progress.percentage === "number") {
            say("Uploading… " + Math.round(progress.percentage) + "%");
          }
        },
      });
    }).catch(function (error) {
      // A rejection from the store is about this file, so it is worth showing.
      // Anything else -- the CDN blocked, the route missing -- is not, and falls
      // through to the form.
      if (error && error.name && String(error.name).indexOf("Blob") === 0) {
        throw { handled: true, payload: { error: error.message, hint: "" } };
      }
      throw error;
    });
  }

  /* Tell the server to drop anything already staged. Best effort: a cancelled
     upload that leaves bytes behind is a quota problem, not a correctness one, and
     retention sweeps them regardless. */
  function abandon() {
    if (!ticket) return;
    var body = JSON.stringify({ ticket: ticket });
    ticket = null;
    if (navigator.sendBeacon) {
      navigator.sendBeacon("/upload/abandon", new Blob([body], { type: "application/json" }));
    } else {
      fetch("/upload/abandon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        keepalive: true,
      }).catch(function () {});
    }
  }

  window.addEventListener("pagehide", abandon);

  form.addEventListener("submit", function (event) {
    var files = chosen();
    if (!files.abundance || !files.metadata) return;   // let the form's own `required` speak

    event.preventDefault();
    clearError();
    if (button) button.disabled = true;
    inFlight = new AbortController();

    var declared = { files: {} };
    Object.keys(files).forEach(function (name) {
      declared.files[name] = { filename: files[name].name, size: files[name].size };
    });

    say("Preparing…");
    fetch("/upload/authorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(declared),
      signal: inFlight.signal,
    })
      .then(function (response) {
        return asJson(response).then(function (payload) {
          // 501 means this deployment cannot delegate the upload. That is not a
          // problem with the file, so the form posts it the ordinary way.
          if (response.status === 501 || payload.fallback === "form") {
            throw { fallback: true };
          }
          if (!response.ok) throw { handled: true, payload: payload };
          return payload;
        });
      })
      .then(function (auth) {
        ticket = auth.ticket;
        var names = Object.keys(auth.uploads);
        var done = 0;
        return names.reduce(function (chain, name) {
          return chain.then(function () {
            var target = auth.uploads[name];
            say("Uploading " + (done + 1) + " of " + names.length + "…");
            var sent = target.strategy === "vercel-blob"
              ? toBlobStore(target, auth.ticket, files[name])
              : toStagingRoute(target, auth.ticket, files[name]);
            return sent.then(function () { done += 1; });
          });
        }, Promise.resolve());
      })
      .then(function () {
        say("Validating…");
        var groupInput = form.querySelector('input[name="group_column"]');
        return fetch("/upload/complete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticket: ticket,
            group_column: groupInput ? groupInput.value : "",
          }),
          signal: inFlight.signal,
        }).then(function (response) {
          return asJson(response).then(function (payload) {
            if (!response.ok) throw { handled: true, payload: payload };
            return payload;
          });
        });
      })
      .then(function (result) {
        ticket = null;                      // the bytes are a dataset now
        window.location.assign(result.next);
      })
      .catch(function (error) {
        if (error && error.handled) {
          // The server rejected the data. Show what it said, in its words.
          abandon();
          restore();
          showError(error.payload);
          return;
        }
        // Something went wrong before the server had an opinion — offline, blocked,
        // a host without this route. Fall back to the plain form post.
        abandon();
        say("Uploading…");
        form.submit();
      });
  });
})();
