// "30-second tour" modal: native <dialog>, never autoplays, opens on ?tour=1.
(function () {
  function dialog() { return document.getElementById('tour-dialog'); }

  function open() {
    var d = dialog();
    if (!d || d.open) return;
    d.querySelector('video').setAttribute('playsinline', '');
    d.showModal();
  }

  function close() {
    var d = dialog();
    if (d && d.open) d.close();
  }

  document.addEventListener('click', function (e) {
    if (e.target.closest('#tour-open')) open();
    else if (e.target.closest('#tour-close') || e.target === dialog()) close();
  });

  // Esc and backdrop both end in the close event; pause and hand focus back.
  document.addEventListener('close', function (e) {
    if (e.target !== dialog()) return;
    e.target.querySelector('video').pause();
    var trigger = document.getElementById('tour-open');
    if (trigger) trigger.focus();
  }, true);

  if (new URLSearchParams(location.search).get('tour') === '1') {
    // Dash renders the layout after load; wait for the dialog to exist.
    var observer = new MutationObserver(function () {
      if (!dialog()) return;
      observer.disconnect();
      open();
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }
})();
