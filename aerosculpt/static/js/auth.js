import { api, ApiError, busy, qs, siteNav, $ } from "/static/js/core.js";

siteNav();
const page = document.body.dataset.page;
const form = $("form");
const errBox = $("#form-error");

function showError(e) {
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
  form.querySelectorAll(".field-error").forEach((el) => el.remove());
  const msg = e instanceof ApiError ? e.message : "Network error — please try again.";
  const field = e instanceof ApiError && e.data.field;
  const input = field && form.querySelector(`[name="${field}"]`);
  if (input && input.type !== "checkbox") {
    input.classList.add("invalid");
    const fe = document.createElement("div");
    fe.className = "field-error";
    fe.textContent = msg;
    input.after(fe);
    input.focus();
  } else {
    errBox.textContent = msg;
    errBox.classList.remove("hidden");
  }
}

function nextUrl() {
  const n = qs("next");
  return n && n.startsWith("/") && !n.startsWith("//") ? n : "/app";
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  errBox.classList.add("hidden");
  const btn = form.querySelector("button[type=submit]");
  const data = Object.fromEntries(new FormData(form).entries());
  busy(btn, true);
  try {
    if (page === "signup") {
      await api("/api/auth/signup", { method: "POST", body: { ...data, accept_terms: !!data.accept_terms } });
      const plan = qs("plan");
      location.href = plan === "pro" || plan === "team" ? `/app/billing?upgrade=${plan}` : "/app?welcome=1";
    } else if (page === "login") {
      await api("/api/auth/login", { method: "POST", body: data });
      location.href = nextUrl();
    } else if (page === "forgot") {
      await api("/api/auth/forgot", { method: "POST", body: data });
      form.innerHTML = `<div class="notice info">If an account exists for that address, a reset link is on its way. It expires in one hour.</div>
        <p style="margin-top:16px"><a href="/login">Back to sign in</a></p>`;
    } else if (page === "reset") {
      if (data.password !== data.password2) throw new ApiError(400, { error: "Passwords do not match.", field: "password2" });
      await api("/api/auth/reset", { method: "POST", body: { token: qs("token"), password: data.password } });
      location.href = "/app";
    }
  } catch (e) {
    showError(e);
  } finally {
    if (document.contains(btn)) busy(btn, false);
  }
});

if (page === "login") {
  // single sign-on (OpenID Connect), when the server has it configured
  api("/api/auth/sso").then((r) => {
    if (!r.enabled) return;
    $("#sso-btn").textContent = `Sign in with ${r.name}`;
    $("#sso-box").classList.remove("hidden");
  }).catch(() => {});
  if (qs("sso_error")) {
    const fe = $("#form-error");
    fe.textContent = `Single sign-on failed: ${qs("sso_error")}`;
    fe.classList.remove("hidden");
  }
}
if (page === "login") {
  api("/api/plans").then((cat) => {
    if (cat.guest_demo) $("#guest-box").classList.remove("hidden");
    if (!cat.signup_open) $("#signup-link").classList.add("hidden");
  }).catch(() => {});
  $("#guest-btn").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    busy(btn, true);
    try {
      await api("/api/auth/guest", { method: "POST" });
      location.href = "/app?welcome=1";
    } catch (err) {
      showError(err);
      busy(btn, false);
    }
  });
}

if (page === "reset" && !qs("token")) {
  errBox.textContent = "This page needs the link from your reset email.";
  errBox.classList.remove("hidden");
}
