"""Standalone DeepSeek relevance scoring (copied from ``scripts/pipeline.py``).

Use this script to experiment with ``RELEVANCE_PROMPT`` without running the
full pipeline. Set ``USER_MESSAGE`` to a non-empty caption (or combined
caption+transcript) to issue a real API call (requires ``DEEPSEEK_API_KEY``).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.db import LeadDB

# Appended to ``RELEVANCE_PROMPT`` in the system message (empty placeholder).
EXTRA_PROMPT_FRAGMENT = ""

RELEVANCE_PROMPT = """\
#Задача
Ты — эксперт-аналитик постов риелторов в Instagram. Твоя задача — проанализировать описание поста/рилса и вернуть строго валидный JSON без markdown и лишнего текста.


#Правила определения полей

1. is_real_estate
true — пост только о покупке/продаже квартиры (или ипотеке) на первичном рынке (не вторичка) и не об аренде.
Обязательные условия для true:

Присутствуют явные или косвенные признаки сделки купли-продажи/ипотеки на квартиру: «продажа», «покупка», «купите», «ипотека», «стоимость квартиры», «цена», «ключи сразу», «новостройка», «ЖК», «сдача корпуса», «ДДУ», «переуступка прав», «от застройщика», «бронирование», «первичный рынок».

Отсутствуют любые признаки аренды.

Отсутствует явное указание на вторичный рынок.

false в случаях:

Пост про аренду квартиры: слова «аренда», «снять», «сдаётся», «сдам», «арендная плата», «в месяц» / «₽/мес.» при описании платы за проживание (не ипотечного платежа).

Пост про покупку/продажу, но квартира явно вторичная: «вторичка», «вторичная квартира», «вторичный рынок», «квартира от собственника» в явном контексте вторички.

Пост про жилую недвижимость, не относящуюся к квартирам (дома, коттеджи, таунхаусы, земля), даже если покупка.

Пост вообще не о жилой недвижимости (коммерческая, офисы и т.п.).

Пост написан не на Русском языке.

null — если текст слишком короткий, неинформативный или невозможно однозначно определить, относится ли он к покупке/продаже квартиры (нет ни признаков покупки, ни аренды, ни вторички, либо контекст неясен).

2. has_call_to_action
true — в тексте есть явный призыв к потенциальным покупателям/клиентам написать в комментарии, директ или перейти по ссылке.
Считаются призывами: «напишите в директ», «в Direct», «пишите в личку», «ставьте +», «комментируйте», «пишите в комментариях», «переходите по ссылке», «жми на ссылку», «ссылка в шапке профиля», «кликай по ссылке», «регистрируйтесь по ссылке».
Не считаются: призывы позвонить, прийти на встречу, информационные сообщения без обращения.
false — если ни одного призыва указанных типов нет.

3. call_to_action_type
"comment" — призыв написать в комментарии (включая «ставьте +», «напишите "хочу" в комментариях»).

"direct" — призыв написать в директ (Direct, личные сообщения).

"link" — призыв перейти по ссылке (активная ссылка, «ссылка в шапке», «кликай»).

"none" — если призыв отсутствует или он не относится к указанным трём каналам.

#Примеры для точной настройки

Вход: «Продаётся 2-к квартира в ЖК «Солнечный», 45 м², цена 5 млн. Ипотека от 3,5%. Пишите в директ.»
Выход: {"is_real_estate": true, "has_call_to_action": true, "call_to_action_type": "direct"}

Вход: «Сдаётся уютная студия, 25 м² за 30 000 ₽/мес. Рядом метро. Вопросы в Direct.»
Выход: {"is_real_estate": false, "has_call_to_action": true, "call_to_action_type": "direct"}

Вход: «Продаётся вторичная квартира, 2-к, 65 м². Цена 10 млн. Все вопросы в комментарии.»
Выход: {"is_real_estate": false, "has_call_to_action": true, "call_to_action_type": "comment"}

Вход: «Выбирайте квартиру в новом ЖК. Планировки и цены по ссылке в шапке профиля.»
Выход: {"is_real_estate": true, "has_call_to_action": true, "call_to_action_type": "link"}

Вход: «Квартира в ипотеку, платёж от 22 000 ₽/мес. Ставьте +, пришлю расчёт.»
Выход: {"is_real_estate": true, "has_call_to_action": true, "call_to_action_type": "comment"}

Вход: «Звоните по поводу квартиры!»
Выход: {"is_real_estate": null, "has_call_to_action": false, "call_to_action_type": "none"}
(неясно, покупка или аренда; призыв к звонку не учитывается)

#Формат ответа
Выдай строго JSON без какого-либо оформления, без markdown:
{"is_real_estate": true/false/null, "has_call_to_action": true/false, "call_to_action_type": "comment"|"direct"|"link"|"none"}
"""

# User message body (same role as Step 2 ``combined`` text). Empty = ``main`` skips the API call.
USER_MESSAGE = """
Аренда это, наверное, единственная тема, по которой у каждого есть мнение 

Кто-то сдаёт, кто-то снимает, у кого-то сдают родители, у кого-то знакомые рассказывают истории, как их кинули или как они сами кого-то выселили. В общем, мимо этой темы пройти сложно, и каждый уже вынес из неё какие-то свои выводы.

Я не на стороне арендаторов и не на стороне собственников. Просто, пока сам не начал разбираться, был уверен, что многие вещи работают совсем по-другому. Например, что хозяин в любой момент может зайти в свою же квартиру или что договор без срока это история «когда хочу, тогда и выгоню».

Оказалось, всё интереснее. Собрал в карусели несколько моментов, которые лично мне показались неочевидными — без позиции «кто прав, кто виноват», просто как это устроено по закону.

А как у вас с этим? Снимаете, сдаёте или давно решили, что лучше своё? Расскажите в комментариях, как сами к этому относитесь.

Шаблон договора найма могу скинуть.

#сосновиков
#новостройки
#ипотека
#недвижимость
#риэлтор
"""


def caption_is_empty(caption: str | None) -> bool:
    if not caption:
        return True
    without_hashtags = " ".join(w for w in caption.strip().split() if not w.startswith("#"))
    return len(without_hashtags.strip()) < 15


def score_caption(client: OpenAI, caption: str) -> dict:
    system_text = RELEVANCE_PROMPT + EXTRA_PROMPT_FRAGMENT
    try:
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": caption[:2000]},
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content
        if not raw:
            return {"error": "empty"}
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def _build_scoring_text(caption: str | None, transcript: str | None) -> str:
    """Concatenate caption and video transcript into a single payload.

    Order is fixed: caption first, transcript second, separated by a
    blank line. Either part may be missing. The result is what gets
    sent to ``RELEVANCE_PROMPT``.
    """
    parts: list[str] = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if transcript and transcript.strip():
        parts.append(transcript.strip())
    return "\n\n".join(parts)


def _apply_score(db: LeadDB, post_id: str, score: dict | None) -> str:
    """Persist a DeepSeek score result. Returns the resolved relevance.

    Copied from ``scripts.pipeline._apply_score``.
    """
    if not score or "error" in score:
        db.upsert_post(
            post_id, relevance="unknown", has_cta=0, cta_type="none"
        )
        return "unknown"
    has_cta = 1 if score.get("has_call_to_action") else 0
    cta_type = score.get("call_to_action_type") or "none"
    is_re = score.get("is_real_estate")
    if is_re is None:
        db.upsert_post(
            post_id, relevance="unknown", has_cta=has_cta, cta_type=cta_type
        )
        return "unknown"
    relevance = "relevant" if is_re else "irrelevant"
    db.upsert_post(
        post_id, relevance=relevance, has_cta=has_cta, cta_type=cta_type
    )
    return relevance


def main() -> int:
    load_dotenv()
    combined = USER_MESSAGE.strip()

    if caption_is_empty(combined):
        print(
            "USER_MESSAGE is empty or too short; "
            "set it (or build text via _build_scoring_text) to call DeepSeek."
        )
        print(f"EXTRA_PROMPT_FRAGMENT repr: {EXTRA_PROMPT_FRAGMENT!r}")
        return 0

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    result = score_caption(client, combined)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "error" in result:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
