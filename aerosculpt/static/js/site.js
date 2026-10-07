// Marketing/legal pages: navigation + footer.
import { siteNav, siteFooter } from "/static/js/core.js";
siteNav();
if (!document.body.dataset.nofooter) siteFooter();
