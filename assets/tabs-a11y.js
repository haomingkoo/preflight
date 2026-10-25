// dcc.Tabs renders plain divs. Give them tab semantics, focus and arrow-key navigation.
(function () {
  function sync(container) {
    container.setAttribute('role', 'tablist');
    container.setAttribute('aria-label', 'Workflow steps');
    container.querySelectorAll('.tab').forEach(function (tab) {
      var selected = tab.classList.contains('tab--selected');
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
  }

  function onKey(e) {
    var tab = e.target.closest && e.target.closest('.tab-container .tab');
    if (!tab) return;
    var tabs = Array.prototype.slice.call(tab.parentNode.querySelectorAll('.tab'));
    var i = tabs.indexOf(tab);
    var next = null;
    if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
    else if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
    else if (e.key === 'Home') next = tabs[0];
    else if (e.key === 'End') next = tabs[tabs.length - 1];
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); tab.click(); return; }
    if (next) { e.preventDefault(); next.click(); next.focus(); }
  }

  var observer = new MutationObserver(function () {
    document.querySelectorAll('.tab-container').forEach(sync);
  });
  observer.observe(document.documentElement, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['class'],
  });
  document.addEventListener('keydown', onKey);
})();
