"""Persian OCR-weighted lexicon spellchecker with visual-confusion costs and ZWNJ/split-join."""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import SpellcheckMode
from src.postprocess.normalizer import ZWNJ, normalize_persian
from src.schemas.document import SpellcheckCorrection

logger = logging.getLogger(__name__)

_PERSIAN_LETTER_RE = re.compile(r"[\u0621-\u065f\u0670-\u06d3\u06f0-\u06f9\u200c]")
_PURE_PERSIAN_WORD_RE = re.compile(r"^[\u0621-\u065f\u0670-\u06d3\u200c]+$")
_HAS_DIGIT_OR_LATIN_RE = re.compile(r"[0-9\u06f0-\u06f9\u0660-\u0669A-Za-z:/@._]")
_TOKEN_EDGE_PUNCT_RE = re.compile(
    r'^([()\[\]{}«»"\'،؛:.!?؟\-–—/\\*]*)(.*?)([()\[\]{}«»"\'،؛:.!?؟\-–—/\\*]*)$'
)
_WS_SPLIT_RE = re.compile(r"(\s+)")

# Same-base dot/stroke groups -> very low OCR confusion cost
_DOT_CONFUSION_GROUPS: tuple[str, ...] = (
    "کگ",
    "رزژ",
    "دذ",
    "سش",
    "صض",
    "طظ",
    "عغ",
    "فق",
    "بپتثنی",
    "جچحخ",
    "آا",
    "ئی",
    "هة",
)

# Near-shape character pairs with slightly higher confusion cost
_SHAPE_CONFUSION_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("ل", "ن", 0.35),
    ("ل", "ا", 0.45),
    ("ه", "و", 0.40),
    ("و", "د", 0.45),
    ("م", "ه", 0.45),
    ("ی", "ئ", 0.25),
)

_THIN_CHARS = frozenset({"ا", "ل", "ر", "ی", "و", "ه", ZWNJ})


def _build_confusion_maps() -> tuple[dict[tuple[str, str], float], dict[str, tuple[str, ...]]]:
    costs: dict[tuple[str, str], float] = {}
    neighbors: dict[str, set[str]] = {}

    for group in _DOT_CONFUSION_GROUPS:
        for ch_a in group:
            for ch_b in group:
                if ch_a != ch_b:
                    costs[(ch_a, ch_b)] = min(costs.get((ch_a, ch_b), 1.0), 0.25)
                    neighbors.setdefault(ch_a, set()).add(ch_b)

    for ch_a, ch_b, cost in _SHAPE_CONFUSION_PAIRS:
        costs[(ch_a, ch_b)] = min(costs.get((ch_a, ch_b), 1.0), cost)
        costs[(ch_b, ch_a)] = min(costs.get((ch_b, ch_a), 1.0), cost)
        neighbors.setdefault(ch_a, set()).add(ch_b)
        neighbors.setdefault(ch_b, set()).add(ch_a)

    return costs, {k: tuple(sorted(v)) for k, v in neighbors.items()}


_CONFUSION_COSTS, _CONFUSION_NEIGHBORS = _build_confusion_maps()

# Curated Iranian geography & official administrative Persian vocabulary
_DOMAIN_SUPPLEMENT: dict[str, float] = {
    # Provinces & major cities / counties
    "آذربایجان": 8e-5,
    "اردبیل": 6e-5,
    "اردستان": 2e-5,
    "ارومیه": 5e-5,
    "اصفهان": 1e-4,
    "البرز": 6e-5,
    "ایذه": 2e-5,
    "ایلام": 5e-5,
    "بجنورد": 4e-5,
    "بختیاری": 5e-5,
    "بلوچستان": 5e-5,
    "بندرعباس": 5e-5,
    "بوانات": 2e-5,
    "بوشهر": 6e-5,
    "بویراحمد": 4e-5,
    "بیرجند": 4e-5,
    "بیله‌سوار": 2e-5,
    "تبریز": 1e-4,
    "تربت": 3e-5,
    "تهران": 2e-4,
    "تیران": 2e-5,
    "جیرفت": 3e-5,
    "چهارمحال": 4e-5,
    "حیدریه": 2e-5,
    "خراسان": 1e-4,
    "خرم‌آباد": 4e-5,
    "خلخال": 2e-5,
    "خوانسار": 2e-5,
    "خوزستان": 8e-5,
    "دماوند": 3e-5,
    "دنا": 2e-5,
    "رباط‌کریم": 2e-5,
    "رشت": 8e-5,
    "رودهن": 2e-5,
    "زاهدان": 5e-5,
    "زنجان": 6e-5,
    "ساری": 6e-5,
    "سمیرم": 2e-5,
    "سمنان": 5e-5,
    "سنندج": 5e-5,
    "سیستان": 5e-5,
    "شاهین‌شهر": 2e-5,
    "شمیرانات": 3e-5,
    "شهرکرد": 4e-5,
    "شهریار": 4e-5,
    "شیراز": 1e-4,
    "طبس": 3e-5,
    "فارس": 8e-5,
    "فریمان": 2e-5,
    "فشافویه": 2e-5,
    "قرچک": 2e-5,
    "قزوین": 6e-5,
    "قم": 8e-5,
    "کازرون": 3e-5,
    "کرج": 8e-5,
    "کرمان": 8e-5,
    "کرمانشاه": 7e-5,
    "کرون": 2e-5,
    "کنگان": 2e-5,
    "کوهبنان": 2e-5,
    "کهگیلویه": 4e-5,
    "گرگان": 6e-5,
    "گلستان": 6e-5,
    "گیلان": 7e-5,
    "لرستان": 6e-5,
    "لواسان": 3e-5,
    "مازندران": 8e-5,
    "مرکزی": 8e-5,
    "مشهد": 1e-4,
    "ملارد": 2e-5,
    "ملایر": 3e-5,
    "مهریز": 2e-5,
    "میناب": 2e-5,
    "میمه": 2e-5,
    "نمین": 2e-5,
    "نهاوند": 3e-5,
    "ورامین": 4e-5,
    "هرمزگان": 6e-5,
    "همدان": 7e-5,
    "یزد": 8e-5,
    # Common Persian administrative / legal / OCR ZWNJ terms
    "آزمون‌ها": 5e-5,
    "آزمون‌های": 5e-5,
    "استخدامی": 8e-5,
    "اسناد": 1e-4,
    "الحاقیه": 4e-5,
    "املاک": 1e-4,
    "ایثارگران": 5e-5,
    "باسمه‌تعالی": 4e-5,
    "بخشنامه": 6e-5,
    "بدوی": 4e-5,
    "پیمانی": 4e-5,
    "تجدیدنظر": 5e-5,
    "ثبت‌نام": 8e-5,
    "داوطلبان": 7e-5,
    "دستورالعمل": 6e-5,
    "دفترچه": 5e-5,
    "شغل‌محل": 3e-5,
    "شغل‌محل‌ها": 3e-5,
    "شغل‌محل‌های": 3e-5,
    "قراردادی": 5e-5,
    "کارمعین": 3e-5,
    "کشور": 5e-4,
    "مأموران": 4e-5,
    "مالیاتی": 7e-5,
    "معلولان": 4e-5,
    "می‌گردد": 8e-5,
    "می‌نمایند": 6e-5,
    "می‌شود": 2e-4,
    "نقشه‌برداری": 5e-5,
    "همسان‌سازی": 3e-5,
    "هیأت": 6e-5,
    "هیأت‌های": 4e-5,
}

_ZWNJ_PREFIXES: tuple[str, ...] = ("می", "نمی", "بی")
_ZWNJ_SUFFIXES: tuple[str, ...] = (
    "ها",
    "های",
    "هایی",
    "تر",
    "ترین",
    "ام",
    "ات",
    "اش",
    "ای",
    "مان",
    "تان",
    "شان",
)
_ATTACHED_SUFFIXES: tuple[str, ...] = (
    "هایی",
    "های",
    "ترین",
    "ها",
    "تر",
    "ان",
    "گان",
    "ات",
    "ی",
)


def _sub_cost(ch_a: str, ch_b: str) -> float:
    if ch_a == ch_b:
        return 0.0
    return _CONFUSION_COSTS.get((ch_a, ch_b), 1.0)


def _indel_cost(ch: str) -> float:
    return 0.65 if ch in _THIN_CHARS else 0.85


def weighted_ocr_distance(a: str, b: str, max_cost: float = 1.5) -> float:
    """Compute OCR visual-confusion weighted Damerau-Levenshtein distance."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if abs(la - lb) * 0.65 > max_cost:
        return max_cost + 1.0

    inf = max_cost + 10.0
    dp = [[inf] * (lb + 1) for _ in range(la + 1)]
    dp[0][0] = 0.0
    for i in range(1, la + 1):
        dp[i][0] = dp[i - 1][0] + _indel_cost(a[i - 1])
    for j in range(1, lb + 1):
        dp[0][j] = dp[0][j - 1] + _indel_cost(b[j - 1])

    for i in range(1, la + 1):
        ca = a[i - 1]
        row_min = inf
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = min(
                dp[i - 1][j] + _indel_cost(ca),
                dp[i][j - 1] + _indel_cost(cb),
                dp[i - 1][j - 1] + _sub_cost(ca, cb),
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cost = min(cost, dp[i - 2][j - 2] + 0.75)
            dp[i][j] = cost
            if cost < row_min:
                row_min = cost
        if row_min > max_cost + 0.85:
            return max_cost + 1.0

    return dp[la][lb]


class PersianLexicon:
    """Normalized Persian frequency lexicon with SymSpell 1-delete index."""

    def __init__(
        self,
        freq_map: dict[str, float],
        delete_index: dict[str, list[str]],
    ) -> None:
        self.freq_map = freq_map
        self.delete_index = delete_index

    @classmethod
    def build(cls, custom_dict_path: Path | None = None) -> "PersianLexicon":
        """Build lexicon from wordfreq 'fa' (best), domain supplement, and optional custom dict."""
        import wordfreq

        raw_dict = wordfreq.get_frequency_dict("fa", wordlist="best")
        freq_map: dict[str, float] = {}

        for word, freq in raw_dict.items():
            norm = normalize_persian(word, digit_policy="preserve")
            if not norm or not _PURE_PERSIAN_WORD_RE.match(norm):
                continue
            if norm not in freq_map or freq > freq_map[norm]:
                freq_map[norm] = float(freq)

        for word, freq in _DOMAIN_SUPPLEMENT.items():
            norm = normalize_persian(word, digit_policy="preserve")
            if norm:
                freq_map[norm] = max(freq_map.get(norm, 0.0), freq)

        if custom_dict_path is not None and custom_dict_path.is_file():
            for raw_line in custom_dict_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                word = normalize_persian(parts[0], digit_policy="preserve")
                if not word:
                    continue
                try:
                    custom_freq = float(parts[1]) if len(parts) > 1 else 1e-4
                except ValueError:
                    custom_freq = 1e-4
                freq_map[word] = max(freq_map.get(word, 0.0), custom_freq)

        # Build SymSpell 1-delete index for fast O(1) edit-distance candidate lookup
        raw_index: dict[str, list[str]] = {}
        for word, freq in freq_map.items():
            if len(word) < 3 or freq < 1e-7:
                continue
            for idx in range(len(word)):
                del_key = word[:idx] + word[idx + 1 :]
                if len(del_key) >= 2:
                    raw_index.setdefault(del_key, []).append(word)

        delete_index: dict[str, list[str]] = {}
        for del_key, cands in raw_index.items():
            if len(cands) > 6:
                cands.sort(key=lambda w: freq_map[w], reverse=True)
                delete_index[del_key] = cands[:6]
            else:
                delete_index[del_key] = cands

        logger.info("Loaded Persian lexicon: %d words, %d SymSpell keys.", len(freq_map), len(delete_index))
        return cls(freq_map=freq_map, delete_index=delete_index)

    def get_freq(self, word: str) -> float:
        """Return lexicon frequency for word (0.0 if absent)."""
        return self.freq_map.get(word, 0.0)

    def is_known(self, word: str) -> bool:
        """Return True if word is valid via direct lookup, ZWNJ morphology, or suffix."""
        if not word:
            return False
        if word in self.freq_map:
            return True

        if ZWNJ in word:
            parts = [p for p in word.split(ZWNJ) if p]
            if len(parts) >= 2:
                if parts[0] in _ZWNJ_PREFIXES and ZWNJ.join(parts[1:]) in self.freq_map:
                    return True
                if parts[-1] in _ZWNJ_SUFFIXES and ZWNJ.join(parts[:-1]) in self.freq_map:
                    return True
                if all(p in self.freq_map and len(p) >= 2 for p in parts):
                    return True

        for suffix in _ATTACHED_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                stem = word[: -len(suffix)]
                if self.freq_map.get(stem, 0.0) >= 1e-6:
                    return True

        return False


def _is_protected_core(core: str) -> bool:
    """Return True if token must never be altered by spellcheck."""
    if not core:
        return True
    if _HAS_DIGIT_OR_LATIN_RE.search(core):
        return True
    if not _PURE_PERSIAN_WORD_RE.match(core):
        return True
    persian_len = len(core.replace(ZWNJ, ""))
    return persian_len < 3


def _split_edge_punct(token: str) -> tuple[str, str, str]:
    match = _TOKEN_EDGE_PUNCT_RE.match(token)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


class PersianSpellchecker:
    """OCR-weighted Persian lexicon spellchecker with optional KenLM rescoring hook."""

    def __init__(
        self,
        lexicon: PersianLexicon,
        mode: SpellcheckMode = "lexicon",
        kenlm_model: Any | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.mode = mode
        self._kenlm = kenlm_model

    def _generate_candidates(self, word: str) -> set[str]:
        """Generate in-lexicon candidates via visual swaps and SymSpell 1-delete lookups."""
        cands: set[str] = set()
        freq_map = self.lexicon.freq_map
        del_index = self.lexicon.delete_index

        # 1. Direct 1-character visual-confusion swaps
        one_char_variants: list[str] = []
        for idx, ch in enumerate(word):
            for alt in _CONFUSION_NEIGHBORS.get(ch, ()):
                cand = word[:idx] + alt + word[idx + 1 :]
                one_char_variants.append(cand)
                if cand in freq_map:
                    cands.add(cand)

        # 2. SymSpell 1-delete lookup (word itself -> insertions; 1-delete -> substitutions/deletions)
        if word in del_index:
            cands.update(del_index[word])
        for idx in range(len(word)):
            del_key = word[:idx] + word[idx + 1 :]
            if del_key in freq_map:
                cands.add(del_key)
            if del_key in del_index:
                cands.update(del_index[del_key])

        # 3. 2-char visual-confusion swaps for longer words when no tight 1-char match exists
        if len(word) >= 5 and not any(
            weighted_ocr_distance(word, c, max_cost=0.5) <= 0.45 for c in cands
        ):
            for idx1, ch1 in enumerate(word):
                alts1 = _CONFUSION_NEIGHBORS.get(ch1, ())
                if not alts1:
                    continue
                for idx2 in range(idx1 + 1, len(word)):
                    alts2 = _CONFUSION_NEIGHBORS.get(word[idx2], ())
                    if not alts2:
                        continue
                    for a1 in alts1:
                        for a2 in alts2:
                            cand = (
                                word[:idx1]
                                + a1
                                + word[idx1 + 1 : idx2]
                                + a2
                                + word[idx2 + 1 :]
                            )
                            if cand in freq_map:
                                cands.add(cand)

        cands.discard(word)
        return cands

    def _try_zwnj_compound(self, word: str) -> str | None:
        """Check if a word missing an internal ZWNJ matches a known ZWNJ-joined form."""
        if len(word) < 5:
            return None
        best: tuple[float, str] | None = None
        for split_idx in range(2, len(word) - 1):
            if word[split_idx - 1] == ZWNJ or word[split_idx] == ZWNJ:
                continue
            candidate = word[:split_idx] + ZWNJ + word[split_idx:]
            freq = self.lexicon.get_freq(candidate)
            if freq >= 5e-6:
                if best is None or freq > best[0]:
                    best = (freq, candidate)
        return best[1] if best is not None else None

    def _select_best_candidate(
        self,
        core: str,
        left_ctx: str = "",
        right_ctx: str = "",
    ) -> tuple[str, float, str] | None:
        """Return (best_candidate, cost, rule) for an OOV Persian word, or None."""
        zwnj_hit = self._try_zwnj_compound(core)
        if zwnj_hit is not None:
            return zwnj_hit, 0.20, "zwnj_join"

        word_len = len(core.replace(ZWNJ, ""))
        if word_len == 3:
            max_cost, min_freq = 0.45, 1e-5
        elif word_len == 4:
            max_cost, min_freq = 0.70, 2e-6
        else:
            max_cost, min_freq = 0.90, 5e-7

        scored: list[tuple[float, float, str]] = []
        for cand in self._generate_candidates(core):
            freq = self.lexicon.get_freq(cand)
            if freq < min_freq:
                continue
            dist = weighted_ocr_distance(core, cand, max_cost=max_cost)
            if dist > max_cost:
                continue
            # Lower objective is better: prioritize OCR visual closeness, break ties with log freq
            lm_bonus = 0.0
            if self.mode == "kenlm" and self._kenlm is not None:
                phrase = " ".join(p for p in (left_ctx, cand, right_ctx) if p)
                lm_bonus = float(self._kenlm.score(phrase, bos=False, eos=False)) * 0.04
            objective = dist - 0.03 * math.log10(freq + 1e-12) - lm_bonus
            scored.append((objective, dist, cand))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        _, best_dist, best_cand = scored[0]
        return best_cand, round(best_dist, 3), "visual_lexicon"

    def _join_adjacent_splits(
        self,
        tokens: list[str],
        page_number: int,
    ) -> tuple[list[str], list[SpellcheckCorrection]]:
        """Merge accidentally split OOV word fragments (e.g. 'برد اری' -> 'برداری')."""
        if len(tokens) < 2:
            return tokens, []

        out: list[str] = []
        corrections: list[SpellcheckCorrection] = []
        i = 0
        while i < len(tokens):
            merged = False
            # Try 3-token ZWNJ/plain join first (e.g. "نقشه برد اری" -> "نقشه‌برداری"), then 2-token
            for span in (3, 2):
                if i + span > len(tokens):
                    continue
                slice_toks = tokens[i : i + span]
                parts_info = [_split_edge_punct(t) for t in slice_toks]
                # Require no inner edge punctuation between merged tokens
                if any(parts_info[k][2] for k in range(span - 1)) or any(
                    parts_info[k][0] for k in range(1, span)
                ):
                    continue
                cores = [p[1] for p in parts_info]
                if any(_HAS_DIGIT_OR_LATIN_RE.search(c) or not _PURE_PERSIAN_WORD_RE.match(c) for c in cores):
                    continue
                all_known = all(self.lexicon.is_known(c) for c in cores)
                has_rare_frag = any(
                    len(c.replace(ZWNJ, "")) <= 3 and self.lexicon.get_freq(c) < 5e-6
                    for c in cores
                )
                if all_known and not has_rare_frag:
                    continue

                candidates = ["".join(cores)]
                if span == 2:
                    candidates.append(ZWNJ.join(cores))
                elif span == 3:
                    candidates.append(cores[0] + ZWNJ + cores[1] + cores[2])
                    candidates.append(cores[0] + cores[1] + ZWNJ + cores[2])

                min_join_freq = 1e-5 if all_known else 1e-6
                best_join: tuple[float, str] | None = None
                for cand in candidates:
                    freq = self.lexicon.get_freq(cand)
                    if freq >= min_join_freq and (best_join is None or freq > best_join[0]):
                        best_join = (freq, cand)

                if best_join is not None:
                    lead = parts_info[0][0]
                    trail = parts_info[-1][2]
                    orig_phrase = " ".join(slice_toks)
                    fixed_tok = f"{lead}{best_join[1]}{trail}"
                    out.append(fixed_tok)
                    corrections.append(
                        SpellcheckCorrection(
                            page_number=page_number,
                            original=orig_phrase,
                            corrected=fixed_tok,
                            cost=0.30,
                            rule="split_join",
                        )
                    )
                    i += span
                    merged = True
                    break

            if not merged:
                out.append(tokens[i])
                i += 1
        return out, corrections

    def correct_line(
        self,
        line: str,
        page_number: int = 0,
    ) -> tuple[str, list[SpellcheckCorrection]]:
        """Apply split-join and OOV visual-confusion spellcheck on a single line."""
        if not line or not line.strip() or not _PERSIAN_LETTER_RE.search(line):
            return line, []

        chunks = _WS_SPLIT_RE.split(line)
        word_positions = [idx for idx, ch in enumerate(chunks) if ch and not ch.isspace()]
        if not word_positions:
            return line, []

        words = [chunks[idx] for idx in word_positions]
        corrections: list[SpellcheckCorrection] = []

        words, join_corrs = self._join_adjacent_splits(words, page_number=page_number)
        corrections.extend(join_corrs)

        # Pass 2: single-token OOV correction
        final_words: list[str] = []
        for idx, tok in enumerate(words):
            lead, core, trail = _split_edge_punct(tok)
            if _is_protected_core(core) or self.lexicon.is_known(core):
                final_words.append(tok)
                continue

            left_ctx = _split_edge_punct(words[idx - 1])[1] if idx > 0 else ""
            right_ctx = _split_edge_punct(words[idx + 1])[1] if idx + 1 < len(words) else ""
            match = self._select_best_candidate(core, left_ctx=left_ctx, right_ctx=right_ctx)
            if match is None:
                final_words.append(tok)
                continue

            cand, cost, rule = match
            corrections.append(
                SpellcheckCorrection(
                    page_number=page_number,
                    original=core,
                    corrected=cand,
                    cost=cost,
                    rule=rule,
                )
            )
            final_words.append(f"{lead}{cand}{trail}")

        if not corrections:
            return line, []

        # Reconstruct preserving original leading/trailing whitespace if token count unchanged,
        # otherwise join cleanly with single spaces.
        if len(final_words) == len(word_positions):
            rebuilt = list(chunks)
            for pos, w in zip(word_positions, final_words):
                rebuilt[pos] = w
            return "".join(rebuilt), corrections

        lead_ws = chunks[0] if chunks and chunks[0].isspace() else ""
        trail_ws = chunks[-1] if chunks and chunks[-1].isspace() else ""
        return f"{lead_ws}{' '.join(final_words)}{trail_ws}", corrections

    def correct_text(
        self,
        text: str,
        page_number: int = 0,
    ) -> tuple[str, list[SpellcheckCorrection]]:
        """Spellcheck multi-line text while preserving line breaks."""
        if not text:
            return "", []
        all_corrs: list[SpellcheckCorrection] = []
        out_lines: list[str] = []
        for line in text.splitlines():
            fixed, corrs = self.correct_line(line, page_number=page_number)
            out_lines.append(fixed)
            all_corrs.extend(corrs)
        return "\n".join(out_lines), all_corrs


@lru_cache(maxsize=4)
def _get_cached_lexicon(custom_dict_str: str) -> PersianLexicon:
    custom_path = Path(custom_dict_str) if custom_dict_str else None
    return PersianLexicon.build(custom_dict_path=custom_path)


@lru_cache(maxsize=4)
def _get_cached_spellchecker(
    mode: SpellcheckMode,
    custom_dict_str: str,
    kenlm_path_str: str,
) -> PersianSpellchecker | None:
    if mode == "off":
        return None
    lexicon = _get_cached_lexicon(custom_dict_str)
    kenlm_model = None
    eff_mode: SpellcheckMode = mode

    if mode == "kenlm":
        if kenlm_path_str and Path(kenlm_path_str).is_file():
            try:
                import kenlm  # type: ignore[import-not-found]

                kenlm_model = kenlm.Model(kenlm_path_str)
                logger.info("Loaded KenLM model from %s", kenlm_path_str)
            except Exception as exc:
                logger.warning("Failed to load KenLM (%s); falling back to lexicon mode.", exc)
                eff_mode = "lexicon"
        else:
            logger.warning("SPELLCHECK_MODE=kenlm requested without valid model file; using lexicon mode.")
            eff_mode = "lexicon"

    return PersianSpellchecker(lexicon=lexicon, mode=eff_mode, kenlm_model=kenlm_model)


def get_spellchecker(
    mode: SpellcheckMode,
    custom_dict: Path | str | None = None,
    kenlm_path: Path | str | None = None,
) -> PersianSpellchecker | None:
    """Return cached PersianSpellchecker for mode, or None when mode == 'off'."""
    if mode == "off":
        return None
    custom_str = str(Path(custom_dict).resolve()) if custom_dict else ""
    kenlm_str = str(Path(kenlm_path).resolve()) if kenlm_path else ""
    return _get_cached_spellchecker(mode, custom_str, kenlm_str)
