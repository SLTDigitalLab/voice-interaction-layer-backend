/* Number, currency, percent, date, time and duration formatting that follows
 * the language chosen on the first screen (en, si, ta).
 *
 *   SltFormat.number(1234567.5)        // "12,34,567.5" in ta, "1,234,567.5" in si and en
 *   SltFormat.currency(1250)           // LKR, "රු. 1,250.00" in si
 *   SltFormat.percent(0.25)            // "25%"
 *   SltFormat.date(new Date())         // long date in the page language
 *   SltFormat.time(new Date())         // short time
 *   SltFormat.duration(105)            // "1:45" (for the OTP and confirmation countdowns)
 *   SltFormat.forLanguage("ta").number(5)   // explicit language, used by tests
 *
 * The page language comes from SltI18n.language() when i18n.js is loaded, else
 * the <html lang> attribute, else English. The locales tried are si-LK, ta-LK
 * and en-LK, then the language alone, then en-US; the first one the browser
 * supports wins. NOTHING here throws: a bad value, a missing Intl or an
 * unsupported locale gives the plain String(value) (or "" for a bad date).
 *
 * Plain script, no dependencies. Loads in the browser and in Node (tests).
 */
(function (root) {
  "use strict";

  var CHAINS = {
    en: ["en-LK", "en-US", "en"],
    si: ["si-LK", "si", "en-LK", "en-US"],
    ta: ["ta-LK", "ta", "en-LK", "en-US"],
  };
  var SI_LUNAR_TO_GREGORIAN = {
    "දුරුතු": "ජනවාරි",
    "නවම්": "පෙබරවාරි",
    "මැදින්": "මාර්තු",
    "බක්": "අප්‍රේල්",
    "වෙසක්": "මැයි",
    "පොසොන්": "ජූනි",
    "ඇසළ": "ජූලි",
    "නිකිණි": "අගෝස්තු",
    "බිනර": "සැප්තැම්බර්",
    "වප්": "ඔක්තෝබර්",
    "ඉල්": "නොවැම්බර්",
    "උඳුවප්": "දෙසැම්බර්",
  };
  var cache = {};

  function normaliseSinhalaMonths(text, lang) {
    if (!text || normalise(lang) !== "si") return text;
    var out = String(text);
    for (var lunar in SI_LUNAR_TO_GREGORIAN) {
      if (Object.prototype.hasOwnProperty.call(SI_LUNAR_TO_GREGORIAN, lunar) && out.indexOf(lunar) !== -1) {
        out = out.split(lunar).join(SI_LUNAR_TO_GREGORIAN[lunar]);
      }
    }
    return out;
  }

  function normalise(lang) {
    var short = String(lang || "").toLowerCase().slice(0, 2);
    return CHAINS[short] ? short : "en";
  }

  function pageLanguage() {
    try {
      if (root.SltI18n && typeof root.SltI18n.language === "function") return root.SltI18n.language();
    } catch (e) { /* fall through */ }
    try {
      if (root.document && root.document.documentElement) return root.document.documentElement.lang || "en";
    } catch (e2) { /* fall through */ }
    return "en";
  }

  function supported(kind, locale) {
    try {
      var ctor = root.Intl && root.Intl[kind];
      return !!(ctor && typeof ctor.supportedLocalesOf === "function" && ctor.supportedLocalesOf([locale]).length);
    } catch (e) {
      return false;
    }
  }

  function localeFor(lang, kind) {
    var code = normalise(lang);
    var key = kind + ":" + code;
    if (cache[key]) return cache[key];
    var chain = CHAINS[code];
    var chosen = null;
    for (var i = 0; i < chain.length; i++) {
      if (supported(kind, chain[i])) { chosen = chain[i]; break; }
    }
    cache[key] = chosen;
    return chosen;
  }

  function build(kind, lang, options) {
    var locale = localeFor(lang, kind);
    if (!locale) return null;
    try {
      return new root.Intl[kind](locale, options || {});
    } catch (e) {
      try { return new root.Intl[kind](locale); } catch (e2) { return null; }
    }
  }

  function toNumber(value) {
    if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
    var n = typeof value === "number" ? value : Number(value);
    return isFinite(n) ? n : null;
  }

  function toDate(value) {
    var d = value instanceof Date ? value : new Date(value === undefined ? Date.now() : value);
    return isNaN(d.getTime()) ? null : d;
  }

  function make(getLang) {
    function numberFormat(value, options) {
      var n = toNumber(value);
      if (n === null) return value === null || value === undefined ? "" : String(value);
      var fmt = build("NumberFormat", getLang(), options);
      try { return fmt ? fmt.format(n) : String(n); } catch (e) { return String(n); }
    }

    return {
      language: function () { return normalise(getLang()); },
      locale: function () { return localeFor(getLang(), "NumberFormat") || "en-US"; },
      number: function (value, options) { return numberFormat(value, options); },
      integer: function (value) { return numberFormat(value, { maximumFractionDigits: 0 }); },
      percent: function (value, options) {
        return numberFormat(value, Object.assign({ style: "percent", maximumFractionDigits: 0 }, options || {}));
      },
      currency: function (value, code, options) {
        return numberFormat(value, Object.assign({ style: "currency", currency: code || "LKR" }, options || {}));
      },
      date: function (value, options) {
        var d = toDate(value);
        if (!d) return "";
        var lang = getLang();
        var fmt = build("DateTimeFormat", lang, options || { dateStyle: "long" });
        try { return normaliseSinhalaMonths(fmt ? fmt.format(d) : d.toDateString(), lang); } catch (e) { return ""; }
      },
      time: function (value, options) {
        var d = toDate(value);
        if (!d) return "";
        var fmt = build("DateTimeFormat", getLang(), options || { timeStyle: "short" });
        try { return fmt ? fmt.format(d) : d.toTimeString().slice(0, 5); } catch (e) { return ""; }
      },
      dateTime: function (value, options) {
        var d = toDate(value);
        if (!d) return "";
        var lang = getLang();
        var fmt = build("DateTimeFormat", lang, options || { dateStyle: "medium", timeStyle: "short" });
        try { return normaliseSinhalaMonths(fmt ? fmt.format(d) : d.toString(), lang); } catch (e) { return ""; }
      },
      // Seconds as m:ss (or h:mm:ss from an hour up), the way the countdowns always read.
      // Digits follow the locale
      // numbering system, which is Latin for si-LK, ta-LK and en-LK.
      duration: function (seconds) {
        var s = toNumber(seconds);
        if (s === null) return "";
        s = Math.max(0, Math.floor(s));
        var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
        var two = function (n) { return (n < 10 ? numberFormat(0) : "") + numberFormat(n, { useGrouping: false }); };
        return (h ? numberFormat(h, { useGrouping: false }) + ":" + two(m) : numberFormat(m, { useGrouping: false })) + ":" + two(r);
      },
    };
  }

  var live = make(pageLanguage);
  live.forLanguage = function (lang) { return make(function () { return lang; }); };
  live._reset = function () { cache = {}; };

  root.SltFormat = live;
  if (typeof module !== "undefined" && module.exports) module.exports = live;
})(typeof window !== "undefined" ? window : globalThis);
