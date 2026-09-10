// Cleveris Research — minimal interactions
(function () {
  "use strict";

  // Scroll reveal via IntersectionObserver
  var revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && revealEls.length) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            e.target.classList.add("in-view");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("in-view"); });
  }

  // Subtle parallax on the banner image (respect reduced motion)
  var bannerImg = document.querySelector(".banner-img");
  var banner = document.querySelector(".banner");
  var prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (bannerImg && banner && !prefersReduced) {
    var ticking = false;
    var onScroll = function () {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () {
        var r = banner.getBoundingClientRect();
        var vh = window.innerHeight;
        // only while banner is in view
        if (r.bottom > 0 && r.top < vh) {
          var progress = Math.min(Math.max(-r.top / vh, 0), 1);
          var shift = progress * 36; // px
          bannerImg.style.transform = "scale(1.04) translateY(" + shift + "px)";
        }
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // Top-nav active state on scroll
  var navLinks = document.querySelectorAll(".topnav a");
  var sections = ["features", "tools", "commands", "download", "colophon"]
    .map(function (id) { return document.getElementById(id); })
    .filter(Boolean);
  if (sections.length && navLinks.length && "IntersectionObserver" in window) {
    var spy = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            navLinks.forEach(function (a) {
              a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id);
            });
          }
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    sections.forEach(function (s) { spy.observe(s); });
  }
})();