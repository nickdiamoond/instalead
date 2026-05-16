"""Score posts/reels relevance via DeepSeek by caption text.

Takes the latest posts_with_comments JSON from test_batch_posts.py,
sends each caption to DeepSeek, saves results back with AI scores.

Posts with empty/unclear captions get relevance="unknown" for future audio transcription.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.logger import get_logger, setup_logging

setup_logging()
log = get_logger("score_posts")

RELEVANCE_PROMPT = """\
Ты анализируешь описание поста/рилса из Instagram от риелтора. Определи:

1. is_real_estate — пост про недвижимость (продажа/покупка квартир, обзоры ЖК, ипотека)?
2. has_call_to_action — есть ли призыв заинтересованным покупателям писать в комментарии/директ (например "пиши +", "оставь заявку", "напиши в директ за подборкой", "подписывайся")?
3. call_to_action_type — тип призыва: "comment" (писать в комментарии), "direct" (писать в директ), "link" (перейти по ссылке), "none" (нет призыва)

Если описание слишком короткое, состоит только из хештегов/эмодзи, или по нему невозможно определить тематику — верни is_real_estate: null.

Ответь ТОЛЬКО валидным JSON без markdown:
{"is_real_estate": true/false/null, "has_call_to_action": true/false, "call_to_action_type": "comment"|"direct"|"link"|"none"}
"""


def caption_is_empty(caption: str | None) -> bool:
    """Check if caption is missing or essentially empty."""
    if not caption:
        return True
    stripped = caption.strip()
    if not stripped:
        return True
    # Only hashtags and/or emoji, no real text
    without_hashtags = " ".join(w for w in stripped.split() if not w.startswith("#"))
    if len(without_hashtags.strip()) < 15:
        return True
    return False


def score_caption(client: OpenAI, caption: str) -> dict:
    """Score a single caption via DeepSeek."""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": caption[:2000]},
            ],
            temperature=0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content
        if not raw:
            log.warning("deepseek_empty", caption=caption[:60])
            return {"error": "empty_response"}
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0].strip()
        if not text:
            log.warning("deepseek_empty_after_strip", caption=caption[:60])
            return {"error": "empty_response"}
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("deepseek_parse_error", raw=repr(raw[:200]) if raw else "None")
        return {"error": f"parse: {repr(raw[:100]) if raw else 'None'}"}
    except Exception as e:
        log.warning("deepseek_error", error=str(e))
        return {"error": str(e)}


def main():
    load_dotenv()

    deepseek = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    # Find latest posts file
    data_dir = Path("data")
    posts_files = sorted(data_dir.glob("posts_with_comments_*.json"), reverse=True)
    if not posts_files:
        log.error("no_posts_file", msg="Run test_batch_posts.py first")
        return

    src_path = posts_files[0]
    with open(src_path, encoding="utf-8") as f:
        posts = json.load(f)

    log.info("loaded_posts", count=len(posts), source=src_path.name)

    # Score each post
    for i, post in enumerate(posts):
        caption = post.get("caption")

        if caption_is_empty(caption):
            post["relevance"] = "unknown"
            post["is_real_estate"] = None
            post["has_call_to_action"] = None
            post["call_to_action_type"] = None
            log.info(
                "scored",
                i=f"{i+1}/{len(posts)}",
                relevance="unknown",
                reason="empty_caption",
                owner=post["owner_username"],
            )
            continue

        score = score_caption(deepseek, caption)

        if "error" in score:
            post["relevance"] = "error"
            post["is_real_estate"] = None
            post["has_call_to_action"] = None
            post["call_to_action_type"] = None
        elif score.get("is_real_estate") is None:
            post["relevance"] = "unknown"
            post["is_real_estate"] = None
            post["has_call_to_action"] = score.get("has_call_to_action")
            post["call_to_action_type"] = score.get("call_to_action_type")
        else:
            post["relevance"] = "relevant" if score["is_real_estate"] else "irrelevant"
            post["is_real_estate"] = score["is_real_estate"]
            post["has_call_to_action"] = score.get("has_call_to_action", False)
            post["call_to_action_type"] = score.get("call_to_action_type", "none")

        log.info(
            "scored",
            i=f"{i+1}/{len(posts)}",
            relevance=post["relevance"],
            cta=post.get("has_call_to_action"),
            cta_type=post.get("call_to_action_type"),
            comments=post["comments_count"],
            owner=post["owner_username"],
            caption=str(caption)[:60],
        )

    # Save
    out_path = src_path.with_name(src_path.stem.replace("posts_with_comments", "posts_scored") + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    # Summary
    by_relevance = {}
    for p in posts:
        r = p.get("relevance", "unknown")
        by_relevance[r] = by_relevance.get(r, 0) + 1

    cta_comment = sum(1 for p in posts if p.get("call_to_action_type") == "comment")
    cta_direct = sum(1 for p in posts if p.get("call_to_action_type") == "direct")

    print(f"\nScored {len(posts)} posts from {src_path.name}")
    print(f"Relevance: {by_relevance}")
    print(f"CTA type: comment={cta_comment}, direct={cta_direct}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
