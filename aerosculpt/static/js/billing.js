import { api, appTop, busy, confirmDialog, esc, fmtDate, fmtInt, h, handleError, icon, qs, requireAuth, toast, $ } from "/static/js/core.js";

let me = await requireAuth();
const top = appTop(me, [{ label: "Plan & billing" }]);
const catalog = await api("/api/plans");

function flash() {
  const box = $("#flash");
  const c = qs("checkout");
  if (c === "success") {
    box.innerHTML = `<div class="notice info" style="margin-top:12px">${icon("check")}<div>${qs("demo")
      ? "Demo mode: your plan was switched instantly without payment."
      : "Thanks! Your subscription is being confirmed with Stripe — this page updates within a few seconds."}</div></div>`;
  } else if (c === "cancel") {
    box.innerHTML = `<div class="notice" style="margin-top:12px">${icon("info")}<div>Checkout was cancelled. Nothing was charged.</div></div>`;
  }
  if (me.billing_mode === "demo") {
    box.insertAdjacentHTML("beforeend", `<div class="notice warn" style="margin-top:12px">${icon("alert")}<div><b style="color:var(--text)">Demo billing.</b>
      This server has no Stripe keys, so upgrades are granted without payment. Set <code>STRIPE_SECRET_KEY</code>,
      <code>STRIPE_WEBHOOK_SECRET</code> and the price IDs before taking real customers (see the README).</div></div>`);
  }
}

function current() {
  const box = $("#current");
  const status = { trialing: "Free trial", active: "Active", past_due: "Payment overdue", canceled: "Cancelled" }[me.plan_status] || me.plan_status;
  let line = "";
  if (me.plan !== "lite") {
    if (me.plan_status === "trialing" && me.trial_ends_at) line = `Trial ends ${fmtDate(me.trial_ends_at)}${me.billing_source === "stripe" ? " — your card is charged then unless you cancel." : "."}`;
    else if (me.cancel_at_period_end && me.current_period_end) line = `Cancels on ${fmtDate(me.current_period_end)}; you keep ${esc(me.plan_name)} until then.`;
    else if (me.current_period_end) line = `Renews ${fmtDate(me.current_period_end)}.`;
  } else line = "Free forever. Upgrade any time — Pro includes a free trial.";
  box.innerHTML = `<div class="label">Current plan</div>
    <div class="row" style="gap:10px;margin:4px 0 6px"><span style="font-size:26px;font-weight:750">${esc(me.plan_name)}</span>
      <span class="badge ${me.plan_status === "past_due" ? "bad" : "accent"}">${esc(status)}</span></div>
    <p class="small" style="margin:0 0 14px">${line}</p><div class="row wrap" id="manage"></div>`;
  const manage = $("#manage");
  if (me.billing_mode === "stripe" && me.billing_source === "stripe") {
    const b = h("button", { class: "btn", type: "button", html: `${icon("ext")}Manage subscription & invoices` });
    b.addEventListener("click", async () => {
      busy(b, true);
      try { location.href = (await api("/api/billing/portal", { method: "POST" })).url; } catch (e) { handleError(e); busy(b, false); }
    });
    manage.append(b);
  } else if (me.billing_mode === "demo" && me.plan !== "lite") {
    const act = me.cancel_at_period_end ? "resume" : "cancel";
    const b = h("button", { class: "btn " + (act === "cancel" ? "danger" : ""), type: "button" }, act === "cancel" ? "Cancel subscription" : "Resume subscription");
    b.addEventListener("click", async () => {
      if (act === "cancel" && !(await confirmDialog("Cancel subscription?", "You keep your plan until the end of the period, then move to Lite.", "Cancel subscription", true))) return;
      try { me = await api("/api/billing/demo", { method: "POST", body: { action: act } }); render(); } catch (e) { handleError(e); }
    });
    const d = h("button", { class: "btn ghost", type: "button" }, "Downgrade to Lite now");
    d.addEventListener("click", async () => {
      if (!(await confirmDialog("Downgrade now?", "Your plan changes to Lite immediately.", "Downgrade", true))) return;
      try { me = await api("/api/billing/demo", { method: "POST", body: { action: "downgrade_now" } }); render(); } catch (e) { handleError(e); }
    });
    manage.append(b, d);
  }
}

function usage() {
  const used = me.usage.credits, total = me.limits.monthly_credits;
  const frac = Math.min(1, used / Math.max(1, total));
  $("#usage").innerHTML = `<div class="label">This month</div>
    <div class="row spread" style="margin:6px 0"><span style="font-size:26px;font-weight:750">${fmtInt(me.credits_left)}</span>
      <span class="muted small">of ${fmtInt(total)} credits left</span></div>
    <div class="meter ${frac > 0.9 ? "bad" : frac > 0.7 ? "warn" : ""}"><div style="width:${(1 - frac) * 100}%"></div></div>
    <div class="kv" style="margin-top:14px"><span class="k">CFD runs</span><span class="v">${fmtInt(me.usage.cfd_runs)}</span>
      <span class="k">Jobs</span><span class="v">${fmtInt(me.usage.jobs)}</span>
      <span class="k">Credits per run</span><span class="v">draft ${catalog.credits_per_run.draft} · standard ${catalog.credits_per_run.standard} · high ${catalog.credits_per_run.high}</span></div>`;
}

function plans() {
  const box = $("#plans");
  box.innerHTML = "";
  for (const p of catalog.plans) {
    const isCur = me.plan === p.key;
    const card = h("div", { class: "card price-card" + (p.key === "pro" ? " featured" : "") });
    const trial = p.trial && !me.trial_used;
    card.innerHTML = `<div class="row spread"><h3 style="margin:0">${esc(p.name)}</h3>${p.key === "pro" ? '<span class="badge accent">Most popular</span>' : ""}</div>
      <p class="small" style="margin:4px 0 12px">${esc(p.tagline)}</p>
      <div class="price">$${p.price_month}<small> / month</small></div>
      <ul>${p.features.map((f) => `<li>${icon("check")}<span>${esc(f)}</span></li>`).join("")}</ul>`;
    const btn = h("button", { class: "btn block " + (p.key === "pro" ? "primary" : ""), type: "button" });
    if (isCur) { btn.textContent = "Current plan"; btn.disabled = true; }
    else if (p.key === "lite") {
      btn.textContent = me.billing_source === "stripe" ? "Downgrade in the billing portal" : "Downgrade";
      btn.disabled = me.plan === "lite";
      btn.addEventListener("click", async () => {
        if (me.billing_mode === "stripe") {
          try { location.href = (await api("/api/billing/portal", { method: "POST" })).url; } catch (e) { handleError(e); }
        } else if (await confirmDialog("Downgrade to Lite?", "Your plan changes to Lite immediately.", "Downgrade", true)) {
          try { me = await api("/api/billing/demo", { method: "POST", body: { action: "downgrade_now" } }); render(); } catch (e) { handleError(e); }
        }
      });
    } else {
      btn.textContent = trial ? `Start ${catalog.trial_days}-day free trial` : `Switch to ${p.name}`;
      btn.addEventListener("click", async () => {
        busy(btn, true, "Opening checkout…");
        try {
          const r = await api("/api/billing/checkout", { method: "POST", body: { plan: p.key } });
          if (r.mode === "demo") {
            me = await api("/api/me");
            toast(`${p.name} activated (demo mode).`, "good");
            history.replaceState(null, "", "/app/billing?checkout=success&demo=1");
            render();
          } else location.href = r.url;
        } catch (e) { handleError(e); busy(btn, false); }
      });
    }
    card.append(btn);
    if (trial && !isCur && p.key !== "lite") card.append(h("p", { class: "small muted", style: { margin: "10px 0 0", textAlign: "center" } }, "No charge until the trial ends. Cancel any time."));
    box.append(card);
  }
}

function render() {
  top.update(me);
  flash();
  current();
  usage();
  plans();
}
render();

// after Stripe checkout the webhook may land a moment after the redirect: poll briefly
if (qs("checkout") === "success" && !qs("demo") && me.plan === "lite") {
  let n = 0;
  const t = setInterval(async () => {
    me = await api("/api/me");
    if (me.plan !== "lite" || ++n > 15) { clearInterval(t); render(); }
  }, 2000);
}
const up = qs("upgrade");
if (up && me.plan === "lite") document.querySelectorAll(".price-card")[up === "team" ? 2 : 1]?.scrollIntoView({ behavior: "smooth" });
