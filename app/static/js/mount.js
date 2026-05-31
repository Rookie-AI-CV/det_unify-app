(function () {
  var el = document.documentElement;
  var prefix = el.getAttribute('data-mount-prefix') || window.__DETUNIFY_MOUNT__ || '';
  if (prefix === '/') prefix = '';
  window.__DETUNIFY_MOUNT__ = prefix;
  window.duPath = function (path) {
    if (!path) return prefix || '/';
    if (path.charAt(0) !== '/') path = '/' + path;
    return prefix + path;
  };
})();
