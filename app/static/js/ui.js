/* Small behaviours shared by every page. The pages work without any of this:
   each block below only adds to markup that already stands on its own. */
(function () {
  "use strict";

  /* ---- the header's real height, for anything that sticks below it --------
     It wraps to two rows on a phone, so no single constant is right. */
  const header = document.querySelector(".site-header");
  function measureHeader() {
    if (header) document.documentElement.style.setProperty("--header-live", header.offsetHeight + "px");
  }
  measureHeader();
  window.addEventListener("resize", measureHeader);

  /* ---- inline definitions and [?] help --------------------------------------
     CSS opens them on hover and focus. Two things CSS cannot do: close one with
     Escape (WCAG 1.4.13 asks that extra content be dismissible without moving
     focus or the pointer), and toggle on tap where a tap does not focus. */
  function hosts() {
    return document.querySelectorAll(".dfn, .help");
  }
  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;
    hosts().forEach(function (el) {
      if (el.matches(":hover") || el.contains(document.activeElement)) {
        el.classList.add("is-dismissed");
      }
    });
  });
  document.addEventListener("pointerover", function (event) {
    const host = event.target.closest && event.target.closest(".dfn, .help");
    if (host) host.classList.remove("is-dismissed");
  });
  document.addEventListener("focusin", function (event) {
    const host = event.target.closest && event.target.closest(".dfn, .help");
    if (host) host.classList.remove("is-dismissed");
  });
  document.addEventListener("click", function (event) {
    const trigger = event.target.closest && event.target.closest(".help-trigger");
    if (trigger) {
      event.preventDefault();
      trigger.focus();
    }
  });

  /* ---- copy buttons: <button data-copy="#id"> copies that element's text ---- */
  document.addEventListener("click", function (event) {
    const button = event.target.closest && event.target.closest("[data-copy]");
    if (!button || !navigator.clipboard) return;
    const source = document.querySelector(button.getAttribute("data-copy"));
    if (!source) return;
    const text = source.value !== undefined && source.tagName === "INPUT"
      ? source.value : source.textContent.trim();
    const label = button.textContent;
    navigator.clipboard.writeText(text).then(function () {
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = label; }, 1600);
    }).catch(function () {});
  });

  /* ---- section navigation: highlight the section being read --------------- */
  const subnav = document.querySelector("[data-subnav]");
  if (subnav && "IntersectionObserver" in window) {
    const links = Array.prototype.slice.call(subnav.querySelectorAll("a[href^='#']"));
    const byId = {};
    links.forEach(function (link) { byId[link.getAttribute("href").slice(1)] = link; });
    const visible = new Map();
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { visible.set(entry.target.id, entry.isIntersecting); });
      const first = links.find(function (link) {
        return visible.get(link.getAttribute("href").slice(1));
      });
      if (!first) return;
      links.forEach(function (link) {
        const on = link === first;
        link.classList.toggle("is-active", on);
        if (on) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
      // On a narrow screen the menu scrolls sideways; keep the current section in it.
      const list = first.closest("ol");
      if (list && list.scrollWidth > list.clientWidth) {
        list.scrollTo({ left: Math.max(0, first.offsetLeft - 16), behavior: "smooth" });
      }
    }, { rootMargin: "-140px 0px -55% 0px" });
    Object.keys(byId).forEach(function (id) {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  }
})();
