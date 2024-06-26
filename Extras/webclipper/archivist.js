// ==UserScript==
// @name         Archivist
// @version      0.1
// @description  archivist userscript posts to sij.ai/clip
// @author       sij.ai
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// ==/UserScript==

(function() {
    'use strict';
    
    window.addEventListener('load', function() {
        setTimeout(function() {
            var data = new URLSearchParams({
                title: document.title,
                url: window.location.href,
                referrer: document.referrer || '',
                width: window.innerWidth ? window.innerWidth.toString() : '',
                encoding: document.characterSet,
                source: document.documentElement.outerHTML
            });

            GM_xmlhttpRequest({
                method: 'POST',
                url: 'https://!{!{ YOUR DOMAIN HERE }!}!/clip?api_key=!{!{ YOUR API KEY HERE }!}!',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Authorization': 'bearer !{!{ GLOBAL_API_KEY HERE }!}!'
                },
                data: data.toString(),
                onload: function(response) {
                    console.log('Data sent to server');
                },
                onerror: function(error) {
                    console.error('Error sending data:', error);
                }
            });
        }, 5000);
    });
})();
