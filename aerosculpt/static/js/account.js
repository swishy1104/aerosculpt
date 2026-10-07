import { api, appTop, busy, esc, h, handleError, modal, requireAuth, toast, $ } from "/static/js/core.js";

let me = await requireAuth();
appTop(me, [{ label: "Account" }]);

const pf = $("#profile");
pf.name.value = me.name || "";
pf.email.value = me.email;
pf.addEventListener("submit", async (e) => {
  e.preventDefault();
  const b = pf.querySelector("button");
  busy(b, true);
  try { me = await api("/api/me", { method: "PATCH", body: { name: pf.name.value } }); toast("Profile saved.", "good"); }
  catch (err) { handleError(err); } finally { busy(b, false); }
});

const pw = $("#password");
pw.addEventListener("submit", async (e) => {
  e.preventDefault();
  const b = pw.querySelector("button");
  busy(b, true);
  try {
    await api("/api/me/password", { method: "POST", body: { current: pw.current.value, new: pw.new.value } });
    pw.reset();
    toast("Password changed.", "good");
  } catch (err) { handleError(err); } finally { busy(b, false); }
});

function renderApi() {
  const box = $("#api-body");
  if (!me.limits.api) {
    box.innerHTML = `<a class="btn" href="/app/billing?upgrade=team">Upgrade to Team for API access</a>`;
    return;
  }
  box.innerHTML = me.has_api_key
    ? `<div class="row wrap"><code>${esc(me.api_key_prefix)}…</code><button class="btn sm" id="rot" type="button">Generate a new key</button><button class="btn sm danger" id="rev" type="button">Revoke</button></div>`
    : `<button class="btn" id="rot" type="button">Generate API key</button>`;
  $("#rot").addEventListener("click", async () => {
    try {
      const r = await api("/api/me/api-key", { method: "POST" });
      me = await api("/api/me");
      renderApi();
      const body = h("div");
      body.innerHTML = `<p>Copy this key now — it is shown only once. Generating a new key revokes the old one.</p><pre style="white-space:pre-wrap;word-break:break-all">${esc(r.key)}</pre>`;
      modal({ title: "Your API key", body, actions: [{ label: "Copy", kind: "primary", onClick: () => { navigator.clipboard.writeText(r.key); toast("Copied."); return true; } }, { label: "Done" }] });
    } catch (err) { handleError(err); }
  });
  const rev = $("#rev");
  if (rev) rev.addEventListener("click", async () => {
    try { await api("/api/me/api-key", { method: "DELETE" }); me = await api("/api/me"); renderApi(); toast("API key revoked."); }
    catch (err) { handleError(err); }
  });
}
renderApi();

$("#delete").addEventListener("click", () => {
  const body = h("div");
  body.innerHTML = `<p>Type your password to confirm. Everything is deleted immediately.</p>
    <label class="field"><span>Password</span><input type="password" id="del-pw" autocomplete="current-password"></label>`;
  modal({
    title: "Delete your account?", body,
    actions: [{ label: "Cancel", kind: "ghost" }, {
      label: "Delete everything", kind: "danger",
      onClick: async (btn) => {
        busy(btn, true);
        try {
          await api("/api/me/delete", { method: "POST", body: { password: $("#del-pw").value } });
          location.href = "/";
        } catch (err) { handleError(err); busy(btn, false); return true; }
      },
    }],
  });
});
