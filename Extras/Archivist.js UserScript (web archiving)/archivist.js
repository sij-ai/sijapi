// ==UserScript==
// @name         Archivist
// @version      0.5
// @description  archivist userscript posts to sij.ai/archive
// @author       sij
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// ==/UserScript==

(function () {
  "use strict";

  // Function to check if the URL is likely an ad, tracker, or unwanted resource
  function isUnwantedURL(url) {
    const unwantedPatterns = [
      /doubleclick\.net/,
      /googlesyndication\.com/,
      /adservice\./,
      /analytics\./,
      /tracker\./,
      /pixel\./,
      /ad\d*\./,
      /\.ad\./,
      /ads\./,
      /\/ads\//,
      /url=http/,
      /safeframe/,
      /adsystem/,
      /adserver/,
      /adnetwork/,
      /sync\./,
      /beacon\./,
      /optimizely/,
      /outbrain/,
      /widgets\./,
      /cdn\./,
      /pixel\?/,
      /recaptcha/,
      /accounts\.google\.com\/gsi/,
      /imasdk\.googleapis\.com/,
      /amplify-imp/,
      /zemanta/,
      /monitor\.html/,
      /widgetMonitor/,
      /nanoWidget/,
      /client_storage/,
    ];
    return unwantedPatterns.some((pattern) => pattern.test(url));
  }

  // Function to archive the page
  function archivePage() {
    var currentUrl = window.location.href;

    if (isUnwantedURL(currentUrl)) {
      console.log("Skipping unwanted URL:", currentUrl);
      return;
    }

    var data = new URLSearchParams({
      title: document.title,
      url: currentUrl,
      referrer: document.referrer || "",
      width: window.innerWidth ? window.innerWidth.toString() : "",
      encoding: document.characterSet,
      source: document.documentElement.outerHTML,
    });

    GM_xmlhttpRequest({
      method: "POST",
      url: "https://api.sij.ai/archive?api_key=sk-NhrtQwCHNdK5sRZC",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Authorization: "bearer sk-NhrtQwCHNdK5sRZC",
      },
      data: data.toString(),
      onload: function (response) {
        console.log("Archive request sent for:", currentUrl);
      },
      onerror: function (error) {
        console.error("Error sending archive request:", error);
      },
    });
  }

  // Debounce function to limit how often archivePage can be called
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Debounced version of archivePage
  const debouncedArchivePage = debounce(archivePage, 2000);

  // Listen for navigation events
  window.addEventListener("popstate", debouncedArchivePage);

  // Intercept link clicks
  document.addEventListener(
    "click",
    function (e) {
      var link = e.target.closest("a");
      if (link && !isUnwantedURL(link.href)) {
        setTimeout(debouncedArchivePage, 1000); // Delay to allow page to load
      }
    },
    true
  );

  // Initial page load
  setTimeout(archivePage, 5000);
})();
