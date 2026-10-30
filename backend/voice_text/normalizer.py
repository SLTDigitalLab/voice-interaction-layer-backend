"""Text normalizer for the speech synthesis tiers (edge-tts, LiveKit Inference, SAPI).

    normalize("Your bill is Rs. 4,590.50, due 25 October 2026.", "en")
    -> "Your bill is four thousand five hundred ninety rupees and fifty cents, due twenty fifth of October twenty twenty six."

Rules, in this order: dates, phone numbers and short codes, times, currency, units,
remaining numbers, abbreviations, English telecom words (si and ta only). All words live in
lexicon.yaml. Properties the tests check: never raises (returns the input on any internal
error), idempotent, the text never grows beyond 8 times plus 64 characters, and the sentence
punctuation (. , ? ! ; :) is kept.

English numbers become words. Sinhala and Tamil numbers do NOT (see lexicon.yaml): integers
stay as digits, but phone numbers, short codes, codes after "OTP" / "PIN", numbers of 7 or
more digits, leading zero strings and decimal fractions are read digit by digit with the digit
words of the language.
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional

LANGS = ("en", "si", "ta")
_MAX_GROWTH = 8
_MAX_EXTRA = 64
_LEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicon.yaml")

_EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_FALLBACK = {"en": {"digits": _EN_DIGITS, "plus": "plus", "point": "point"}}

_lex_cache: Optional[dict] = None


def _lexicon() -> dict:
    global _lex_cache
    if _lex_cache is None:
        data: dict = {}
        try:
            import yaml

            with open(_LEX_PATH, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
        langs = data.get("languages") if isinstance(data.get("languages"), dict) else {}
        out: Dict[str, dict] = {}
        for code in LANGS:
            entry = dict(_FALLBACK["en"])
            entry.update(langs.get(code) or {})
            entry["_abbr"] = _compile_table(entry.get("abbreviations"), ignore_case=False)
            entry["_words"] = _compile_table(entry.get("words"), ignore_case=True)
            units = entry.get("units") or {}
            entry["_unit_re"] = _unit_regex(units)
            out[code] = entry
        out["_short_codes"] = [str(c) for c in (data.get("short_codes") or []) if re.fullmatch(r"[0-9]{3,5}", str(c))]
        _lex_cache = out
    return _lex_cache


def _compile_table(table, ignore_case: bool):
    if not isinstance(table, dict) or not table:
        return None
    lowered = {str(k).lower() if ignore_case else str(k): str(v) for k, v in table.items()}
    keys = sorted(lowered, key=len, reverse=True)
    pattern = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in keys)
    rx = re.compile(r"(?<![A-Za-z0-9])(?:" + pattern + r")(?![A-Za-z0-9])", re.I if ignore_case else 0)
    return rx, lowered, ignore_case


def _unit_regex(units: dict) -> Optional[str]:
    names = sorted((re.escape(str(k)) for k in units), key=len, reverse=True)
    return "|".join(names) if names else None


# === English numbers

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ["", "thousand", "million", "billion", "trillion"]
_ORD_SPECIAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                "nine": "ninth", "twelve": "twelfth"}


def en_int(n: int) -> str:
    """Whole number below 10**15 as English words."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + en_int(rest))
    parts: List[str] = []
    idx = 0
    while n > 0:
        n, chunk = divmod(n, 1000)
        if chunk:
            parts.append(en_int(chunk) + (" " + _SCALES[idx] if _SCALES[idx] else ""))
        idx += 1
    return " ".join(reversed(parts))


def en_ordinal(n: int) -> str:
    words = en_int(n).split(" ")
    last = words[-1]
    if last in _ORD_SPECIAL:
        last = _ORD_SPECIAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return " ".join(words[:-1] + [last])


def en_year(y: int) -> str:
    if 2000 <= y <= 2009:
        return "two thousand" + ("" if y == 2000 else " " + en_int(y % 100))
    hi, lo = divmod(y, 100)
    if lo == 0:
        return en_int(hi) + " hundred"
    if lo < 10:
        return en_int(hi) + " oh " + en_int(lo)
    return en_int(hi) + " " + en_int(lo)


# === helpers

def _digit_words(digits: str, L: dict) -> str:
    d = L["digits"]
    return " ".join(d[int(c)] for c in digits if "0" <= c <= "9")


def _say_number(int_part: str, frac: Optional[str], lang: str, L: dict, force_digits: bool = False, year_ok: bool = False, quantity: bool = False) -> str:
    """int_part: digits only (commas removed). frac: digits after the point or None."""
    spelled = force_digits or (len(int_part) >= 7 and not quantity) or (len(int_part) > 1 and int_part[0] == "0")
    if spelled:
        head = _digit_words(int_part, L)
    elif lang == "en":
        n = int(int_part)
        head = en_year(n) if (year_ok and frac is None and 1900 <= n <= 2099) else en_int(n)
    elif quantity and len(int_part) >= 7:
        head = f"{int(int_part):,}"  # keep the commas so a second pass does not spell it out
    else:
        head = str(int(int_part)) if int_part else ""
    if frac:
        return f"{head} {L['point']} {_digit_words(frac, L)}"
    return head


def _clean_num(raw: str):
    int_part, _, frac = raw.partition(".")
    return int_part.replace(",", ""), (frac or None)


def _ends_sentence(rest: str) -> bool:
    if rest == "" or rest[0] == "\n":
        return True
    if rest[0].isspace():
        s = rest.lstrip()
        return s == "" or not (s[0].islower() or s[0].isdigit())
    return False


_NUM = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
_SEP = r"[  -]"

# === stages: dates

_MONTH_EN = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december"]
_MONTH_RX = "(" + "|".join(m.capitalize() for m in _MONTH_EN) + ")"
_ORD_SUFFIX = r"(?:st|nd|rd|th)?"


def _valid_date(d: int, m: int) -> bool:
    return 1 <= m <= 12 and 1 <= d <= 31


def _speak_date(day: int, month: int, year: Optional[int], lang: str, L: dict) -> str:
    mname = L["months"][month - 1]
    if lang == "en":
        out = f"{en_ordinal(day)} of {mname}"
        if year:
            out += " " + en_year(year)
        return out
    out = f"{mname} {day}"
    return f"{year} {out}" if year else out


def _dates(text: str, lang: str, L: dict) -> str:
    def iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _speak_date(d, mo, y, lang, L) if _valid_date(d, mo) and 1000 <= y <= 2999 else m.group(0)

    def dmy(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _speak_date(d, mo, y, lang, L) if _valid_date(d, mo) and 1000 <= y <= 2999 else m.group(0)

    def dmonthy(m):
        d = int(m.group(1))
        mo = _MONTH_EN.index(m.group(2).lower()) + 1
        y = int(m.group(3)) if m.group(3) else None
        return _speak_date(d, mo, y, lang, L) if 1 <= d <= 31 else m.group(0)

    def monthdy(m):
        mo = _MONTH_EN.index(m.group(1).lower()) + 1
        d = int(m.group(2))
        y = int(m.group(3)) if m.group(3) else None
        return _speak_date(d, mo, y, lang, L) if 1 <= d <= 31 else m.group(0)

    text = re.sub(r"(?<![0-9])([0-9]{4})-([0-9]{2})-([0-9]{2})(?![0-9])", iso, text)
    text = re.sub(r"(?<![0-9/.])([0-9]{1,2})[/.]([0-9]{1,2})[/.]([0-9]{4})(?![0-9])", dmy, text)
    text = re.sub(
        r"(?<![0-9])([0-9]{1,2})" + _ORD_SUFFIX + r"\s+(?:of\s+)?(?i:" + _MONTH_RX + r")(?![A-Za-z])(?:,?\s+([0-9]{4})(?![0-9]))?",
        dmonthy, text)
    text = re.sub(
        r"(?<![A-Za-z])(?i:" + _MONTH_RX + r")\s+([0-9]{1,2})" + _ORD_SUFFIX + r"(?![0-9A-Za-z])(?:,?\s+([0-9]{4})(?![0-9]))?",
        monthdy, text)
    return text


# === stages: phones

def _phone_words(raw: str, lang: str, L: dict) -> str:
    plus = False
    s = raw.strip()
    if s.startswith("+"):
        plus = True
        s = s[1:]
    elif s.startswith("0094"):
        plus = True
        s = s[2:]
    groups = [g for g in re.split(_SEP + "+", s.strip()) if g]
    spoken = [_digit_words(g, L) for g in groups]
    out = ", ".join(spoken)
    return (L["plus"] + " " + out) if plus else out


def _phones(text: str, lang: str, L: dict, codes: List[str]) -> str:
    rx = re.compile(
        r"(?<![0-9A-Za-z.,/+-])(?:\+\s?94|0094|0)(?:" + _SEP + r"?[0-9]){9}(?![0-9A-Za-z]|[.,][0-9])"
    )

    def phone(m):
        return _phone_words(m.group(0), lang, L)

    text = rx.sub(phone, text)
    if codes:
        crx = re.compile(r"(?<![0-9A-Za-z.,])(" + "|".join(re.escape(c) for c in codes) + r")(?![0-9A-Za-z]|[.,][0-9])")
        text = crx.sub(lambda m: _digit_words(m.group(1), L), text)
    return text


# === stages: times

def _time_en(h: int, mi: int, has_mer: bool) -> str:
    if mi == 0:
        return en_int(h) + (" o'clock" if (has_mer or h <= 12) else " hundred") if h != 0 else "zero hundred"
    return en_int(h) + " " + (("oh " + en_int(mi)) if mi < 10 else en_int(mi))


def _times(text: str, lang: str, L: dict) -> str:
    ctx = [re.escape(str(w)) for w in (L.get("time_context") or [])]
    ctx_rx = re.compile(r"(?:^|[\s(])(?:" + "|".join(ctx) + r")\s*$", re.I) if ctx else None
    rx = re.compile(
        r"(?<![0-9.:,/])(?P<h>[0-9]{1,2})(?P<sep>[.:])(?P<m>[0-9]{2})(?![0-9])(?![.:][0-9])"
        r"(?:\s*(?P<mer>[aApP]\.?[mM]\.?)(?![A-Za-z]))?"
    )

    def sub(m):
        h, mi, sep, mer = int(m.group("h")), int(m.group("m")), m.group("sep"), m.group("mer")
        rest = m.string[m.end():]
        if h > 24 or mi > 59 or (mer and not (1 <= h <= 12)):
            return m.group(0)
        if sep == "." and not mer:
            before = m.string[max(0, m.start() - 24): m.start()]
            if not (ctx_rx and ctx_rx.search(before)):
                return m.group(0)
            if re.match(r"\s*[A-Za-z%]", rest):
                return m.group(0)
        tail = ""
        mer_word = ""
        if mer:
            key = "am" if mer[0].lower() == "a" else "pm"
            mer_word = (L.get("meridiem") or {}).get(key, "")
            if mer.endswith(".") and _ends_sentence(rest):
                tail = "."
        if lang == "en":
            out = _time_en(h, mi, bool(mer))
        else:
            out = f"{h}:{mi:02d}"
        if mer_word:
            out += " " + mer_word
        return out + tail

    return rx.sub(sub, text)


# === stages: currency

def _money(int_part: str, frac: Optional[str], lang: str, L: dict) -> str:
    cents = None
    if frac is not None and frac.strip("0") != "":
        if len(frac) <= 2:
            cents = int(frac.ljust(2, "0"))
        else:
            return _say_number(int_part, frac, lang, L) + " " + _rupee_word(int_part, lang, L, plural=True)
    if lang == "en":
        n = int(int_part) if len(int_part) < 16 else None
        if n is None or (len(int_part) > 1 and int_part[0] == "0"):
            return _digit_words(int_part, L) + " rupees"
        out = en_int(n) + (" rupee" if n == 1 and cents is None else " rupees")
        if cents is not None:
            out += " and " + en_int(cents) + (" cent" if cents == 1 else " cents")
        return out
    num = _say_number(int_part, None, lang, L, quantity=True)
    out = L["rupee_fmt"].format(n=num)
    if cents is not None:
        out += " " + L["cent_fmt"].format(n=cents)
    return out


def _rupee_word(int_part: str, lang: str, L: dict, plural: bool) -> str:
    if lang == "en":
        return "rupees"
    return L["rupee_fmt"].format(n="").strip()


def _currency(text: str, lang: str, L: dict) -> str:
    letter = "A-Za-z඀-෿஀-௿"
    prefix = re.compile(
        r"(?<![" + letter + r"0-9])(?:(?i:rs|lkr)|රු|ரூ|₨)\.?[  ]?(" + _NUM + r")(?![0-9A-Za-z])(?!\.[0-9])(?:[  ]?/[=-])?"
    )
    suffix = re.compile(r"(?<![0-9A-Za-z,.])(" + _NUM + r")[  ]?(?i:lkr)(?![A-Za-z])")

    def sub(m):
        i, f = _clean_num(m.group(1))
        return _money(i, f, lang, L)

    return suffix.sub(sub, prefix.sub(sub, text))


# === stages: units and numbers

def _units(text: str, lang: str, L: dict) -> str:
    units = L.get("units") or {}
    urx = L.get("_unit_re")
    if not urx:
        return text
    rx = re.compile(r"(?<![0-9A-Za-z.,])(" + _NUM + r")[  ]?(" + urx + r"|%)(?![A-Za-z])(?!\.[0-9])")

    def sub(m):
        i, f = _clean_num(m.group(1))
        num = _say_number(i, f, lang, L, quantity=True)
        unit = m.group(2)
        if unit == "%":
            if L.get("percent_position") == "before" and m.string[:m.start()].rstrip().endswith(str(L.get("percent"))):
                return m.group(0)
            word = L.get("percent", "percent")
            return f"{word} {num}" if L.get("percent_position") == "before" else f"{num} {word}"
        spoken = units.get(unit)
        if spoken is None:
            return m.group(0)
        if isinstance(spoken, list):
            spoken = spoken[0] if (i == "1" and f is None) else spoken[1]
        return f"{num} {spoken}"

    return rx.sub(sub, text)


_QTY_AFTER_RX = re.compile(
    r"^\s*(?:rupees?|cents?|gigabytes?|megabytes?|kilobytes?|percent|units?|රුපියල්|ශත|ගිගාබයිට්|මෙගාබයිට්|ரூபாய்|சதம்|ஜிகாபைட்|மெகாபைட்|சதவீதம்)\b",
    re.I,
)
_QTY_BEFORE_RX = re.compile(r"(?:රුපියල්|ශත|සියයට|ரூபாய்|சதம்)\s*$", re.I)


def _numbers(text: str, lang: str, L: dict) -> str:
    ctx_words = [re.escape(str(w)) for w in (L.get("digit_context") or [])]
    ctx_rx = re.compile(r"(?:^|[^A-Za-z])(?:" + "|".join(ctx_words) + r")[\s:#-]*$", re.I) if ctx_words else None

    if lang == "en":
        def ordinal(m):
            return en_ordinal(int(m.group(1)))

        text = re.sub(r"(?<![0-9A-Za-z.,])([0-9]{1,9})(?:st|nd|rd|th)(?![A-Za-z])", ordinal, text)

    rx = re.compile(r"(?<![0-9A-Za-z_.,])(" + _NUM + r")(?![0-9A-Za-z])(?!\.[0-9])(?!,[0-9])")

    def sub(m):
        raw = m.group(1)
        i, f = _clean_num(raw)
        if len(i) > 15:
            return _digit_words(i, L) + ((" " + L["point"] + " " + _digit_words(f, L)) if f else "")
        force = False
        before = m.string[max(0, m.start() - 24): m.start()]
        after = m.string[m.end(): m.end() + 24]
        if ctx_rx and "," not in raw and f is None:
            force = bool(ctx_rx.search(before))
        is_qty = ("," in raw) or bool(_QTY_AFTER_RX.search(after)) or bool(_QTY_BEFORE_RX.search(before))
        return _say_number(i, f, lang, L, force_digits=force, year_ok=not is_qty, quantity=is_qty)

    return rx.sub(sub, text)


# === stages: words

def _table(text: str, compiled, lang_en: bool) -> str:
    if not compiled or lang_en:
        return text
    rx, mapping, ignore = compiled

    def sub(m):
        key = re.sub(r"\s+", " ", m.group(0))
        key = key.lower() if ignore else key
        return mapping.get(key, m.group(0))

    return rx.sub(sub, text)


def _abbreviations(text: str, compiled) -> str:
    if not compiled:
        return text
    rx, mapping, ignore = compiled

    def sub(m):
        key = re.sub(r"\s+", " ", m.group(0))
        return mapping.get(key, m.group(0))

    return rx.sub(sub, text)


# === entry point

def _run(text: str, lang: str) -> str:
    """Runs the rules until the text stops changing (at most 5 passes), which is what
    makes the result idempotent even for odd input such as 'Rs. Rs. 10'."""
    current = text
    for _ in range(5):
        nxt = _pass(current, lang)
        if nxt == current:
            break
        current = nxt
        if len(current) > _MAX_GROWTH * len(text) + _MAX_EXTRA:
            break
    return current


def _pass(text: str, lang: str) -> str:
    lex = _lexicon()
    L = lex[lang]
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ 	 ]{2,}", " ", text).strip()
    text = _dates(text, lang, L)
    text = _phones(text, lang, L, lex["_short_codes"])
    text = _times(text, lang, L)
    text = _currency(text, lang, L)
    text = _units(text, lang, L)
    text = _numbers(text, lang, L)
    text = _abbreviations(text, L.get("_abbr"))
    if lang != "en":
        text = _table(text, L.get("_words"), False)
    text = re.sub(r"[ \t ]{2,}", " ", text)
    return text.strip()


def normalize(text, language: str = "en") -> str:
    """Spoken form of text for a TTS voice. Never raises."""
    if not isinstance(text, str):
        return ""
    if not text:
        return text
    try:
        lang = (language or "en").strip().lower()[:2]
        if lang not in LANGS:
            lang = "en"
        out = _run(text, lang)
        if len(out) > _MAX_GROWTH * len(text) + _MAX_EXTRA:
            return text
        return out
    except Exception:
        return text
