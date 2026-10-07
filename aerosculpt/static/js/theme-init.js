// Runs before first paint (plain script in <head>): apply the saved theme so pages never
// flash the wrong colours. No saved choice = follow the operating system.
(function () {
  try {
    var t = localStorage.getItem("as-theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage unavailable: follow the system */ }
})();
