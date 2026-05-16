from scripts.pipeline_lib.constants import CHARSET


def shortcode_to_id(sc: str) -> int:
    mid = 0
    for ch in sc:
        mid = mid * 64 + CHARSET.index(ch)
    return mid


def caption_is_empty(caption: str | None) -> bool:
    if not caption:
        return True
    without_hashtags = " ".join(w for w in caption.strip().split() if not w.startswith("#"))
    return len(without_hashtags.strip()) < 15
