"""
Smoke test for lingua-language-detector (pemistahl/lingua-py).

Hardcoded Russian phrase: check detected language and confidence breakdown.
Run from repo root: python scripts/test_lingua_russian.py
"""

from __future__ import annotations

import sys

from lingua import Language, LanguageDetectorBuilder

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

# Phrase to classify (Russian: "Hello world")
SAMPLE_TEXT = """
" You can die, and it's the time of your love She's a girl
"""


def main() -> None:
    # Restrict to Cyrillic-script languages — closer to "is this Russian?" in production.
    detector_cyrillic = (
        LanguageDetectorBuilder.from_all_languages_with_cyrillic_script().build()
    )
    detected = detector_cyrillic.detect_language_of(SAMPLE_TEXT)
    confidences = detector_cyrillic.compute_language_confidence_values(SAMPLE_TEXT)

    print("=== Lingua (Cyrillic-script languages only) ===")
    print(f"input: {SAMPLE_TEXT!r}")
    print(f"detect_language_of: {detected!r}")
    if detected is not None:
        print(f"  iso639_1: {detected.iso_code_639_1.name}")
        print(f"  iso639_3: {detected.iso_code_639_3.name}")
        print(f"  language.name: {detected.name}")
    is_russian = detected == Language.RUSSIAN
    print(f"is_russian (== Language.RUSSIAN): {is_russian}")
    print("confidence values (sum to 1.0):")
    for c in confidences:
        print(f"  {c.language.name}: {c.value:.4f}")

    # Optional: same text against all supported languages (heavier model set).
    detector_all = LanguageDetectorBuilder.from_all_spoken_languages().build()
    detected_all = detector_all.detect_language_of(SAMPLE_TEXT)
    print()
    print("=== Lingua (all spoken languages) ===")
    print(f"detect_language_of: {detected_all!r}")
    if detected_all is not None:
        print(f"  iso639_1: {detected_all.iso_code_639_1.name}")
        print(f"  language.name: {detected_all.name}")
    top_all = detector_all.compute_language_confidence_values(SAMPLE_TEXT)[:8]
    print("top confidence values:")
    for c in top_all:
        print(f"  {c.language.name}: {c.value:.4f}")


if __name__ == "__main__":
    main()
