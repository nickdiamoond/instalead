# Apify Instagram API — схемы и связь с пайплайном

Документ описывает акторы Apify, которые реально вызывает `scripts/pipeline.py`, и поля ответов, которые мы читаем. Для dev-скриптов и `src/apify_client_wrapper.py` в конце есть краткая справка по legacy-актёрам.

**Общее:**
- Оплата обычно **за каждый item** в датасете (PAY_PER_EVENT), точная сумма — в `run.usageTotalUsd` после прогона.
- ID акторов можно переопределить в `config.yaml` → `apify.actors.*`.
- Лимиты шагов — в `config.yaml` → `pipeline.stepN.*` (в коде есть `DEFAULT_*` в `scripts/pipeline_lib/defaults.py`).

---

## Карта пайплайна (актуально для `scripts/pipeline.py`)

| Шаг | Что делает | Актор(ы) | Ключ config |
|-----|------------|----------|-------------|
| **1** | Находит посты/рилсы | см. режимы ниже | `pipeline.step1.discovery_mode` |
| **2** | Оценка релевантности (DeepSeek + Nexara) | Apify **не** используется | — |
| **3** | Комментарии к релевантным постам | `louisdeconinck` → `apidojo-api` | `apify.actors.comments_primary` / `comments_fallback` |
| **4** | Профили лидов, аватар, лица | `apify/instagram-profile-scraper` | `pipeline.step4.*` |
| **5** | Telegram через Sherlock | внешний API, не Apify | — |
| **6** | Удаление фото с диска | Apify **не** используется | — |

### Step 1 — три режима (`pipeline.step1.discovery_mode`)

**`realtors`** (по умолчанию)
- Список: `search.realtor_accounts`
- Актор: `apify/instagram-post-scraper` (жёстко в коде; в config есть `apify.actors.posts` для справки)
- Вход: `username`, `resultsLimit`, `onlyPostsNewerThan: "{N} days"`, `dataDetailLevel: "basicData"`, `proxy: { useApifyProxy: true }`
- Дополнительно: клиентский фильтр по `timestamp` (`posts_max_age_days` из config)

**`hashtags`**
- Список: `search.hashtags`
- Актор: `apify/instagram-hashtag-scraper` (`apify.actors.hashtag`)
- **Два прогона:** `resultsType: "posts"` и `resultsType: "reels"` с одним `resultsLimit` (`hashtag_results_limit`)
- Вход обоих: `hashtags`, `resultsLimit`, `proxy: { useApifyProxy: true }`
- У актора **нет** `onlyPostsNewerThan` — возраст режем в коде (`filter_items_within_max_age`)
- Слияние posts+reels по `shortCode` (`merge_hashtag_items_by_shortcode`), для рилса нужен валидный `videoUrl`

**`cookie_keywords`**
- Список: `search.cookie_search_keywords`
- Куки: env из `pipeline.step1.cookie_search.session_cookie_env_var` (обычно `INSTAGRAM_SESSION_COOKIE`)
- Актор: `crawlerbros/instagram-keyword-search-scraper` (`apify.actors.cookie_search_posts`)
- Нормализация ответа → тот же формат, что у hashtag/post-scraper (`src/instagram_cookie_search.py`)

**Общие фильтры Step 1 (все режимы):**
- `commentsCount >= pipeline.step1.min_comments_per_post`
- Рил (`type=Video` или `productType=clips`): без валидного HTTPS `videoUrl` пост **не** попадает в БД (нужен для транскрипции в Step 2)
- Дедуп и upsert в `processed_posts` по `shortCode`

### Step 3 — комментарии

1. **Primary:** `louisdeconinck/instagram-comments-scraper`
2. Если primary вернул **0 items** при `status=SUCCEEDED` → **fallback:** `apidojo/instagram-comments-scraper-api`
3. Если оба пустые → посты **не** помечаются как отсканированные (повтор на следующем запуске)

Перед запуском — оценка стоимости и подтверждение (если `pipeline.prompt_terminal_confirmation: true`).

### Step 4 — профили

- Батчи по `pipeline.step4.profile_batch_size` (default 50), не больше `pipeline.step4.batch_limit` лидов за run
- Вход: только `{ "usernames": [...] }` — без `includeAboutSection`

---

## 1. instagram-hashtag-scraper (`apify/instagram-hashtag-scraper`)

**Step 1 в режиме `hashtags`.** Возвращает посты/рилсы по хештегу, не URL тега.

### Запрос (как шлёт пайплайн)

```json
{
  "hashtags": ["квартираспб", "недвижимостьспб"],
  "resultsType": "posts",
  "resultsLimit": 20,
  "proxy": { "useApifyProxy": true }
}
```

Отдельный run с `"resultsType": "reels"` и тем же `resultsLimit`.

| Параметр | Обязателен | Описание |
|----------|------------|----------|
| `hashtags` | да | Без `#` |
| `resultsType` | да (в пайплайне) | `"posts"` или `"reels"` — два вызова |
| `resultsLimit` | нет | Лимит на хештег за run |
| `proxy.useApifyProxy` | да (у нас) | Без прокси IG часто режет запросы |

### Ответ — поля, которые читает пайплайн

```json
{
  "shortCode": "DFgH1jK2lMn",
  "url": "https://www.instagram.com/p/DFgH1jK2lMn/",
  "type": "Video",
  "productType": "clips",
  "caption": "...",
  "commentsCount": 47,
  "likesCount": 350,
  "timestamp": "2026-03-10T14:00:00.000Z",
  "ownerUsername": "anna_realtor_spb",
  "videoUrl": "https://scontent-...",
  "locationName": "Санкт-Петербург",
  "locationId": "12345678"
}
```

- `type` + `productType` — рил vs обычный пост
- `videoUrl` — обязателен для рилса в нашей БД
- `locationName` / `locationId` — только если у поста есть геотег
- `latestComments` в выдаче есть, но **полные комментарии** мы не берём отсюда — только Step 3

**Цена:** порядка ~$0.0023/пост (ориентир; смотреть `usageTotalUsd`).

---

## 2. instagram-post-scraper (`apify/instagram-post-scraper`)

**Step 1 в режиме `realtors`.**

### Запрос (как шлёт пайплайн)

```json
{
  "username": ["anna_realtor_spb", "goncharov_nedvizhimost"],
  "resultsLimit": 20,
  "onlyPostsNewerThan": "7 days",
  "dataDetailLevel": "basicData",
  "proxy": { "useApifyProxy": true }
}
```

| Параметр | Описание |
|----------|----------|
| `username` | Массив username или URL профиля |
| `resultsLimit` | Макс. постов на аккаунт (`pipeline.step1.post_scraper_results_limit`) |
| `onlyPostsNewerThan` | Строка вида `"{posts_max_age_days} days"` |
| `dataDetailLevel` | У нас всегда `"basicData"` (дешевле, без лишних latest comments) |
| `proxy` | `useApifyProxy: true` |

После ответа — тот же клиентский фильтр по `timestamp`, что и для хештегов.

**Формат поста** — как у hashtag-scraper (`shortCode`, `commentsCount`, `videoUrl` для рилсов и т.д.).

**Цена:** ~$0.0017/пост (basicData).

---

## 3. instagram-keyword-search-scraper (`crawlerbros/instagram-keyword-search-scraper`)

**Step 1 в режиме `cookie_keywords`.** Поиск по ключевым словам от имени залогиненной сессии (куки из `.env`).

### Запрос (как шлёт пайплайн)

```json
{
  "keywords": ["новостройки", "квартира спб"],
  "maxPosts": 20,
  "cookies": "[{\"name\":\"sessionid\",\"value\":\"...\",\"domain\":\".instagram.com\",...}]",
  "sessionName": "instalead_cookie_search"
}
```

| Параметр | Источник | Описание |
|----------|----------|----------|
| `keywords` | `search.cookie_search_keywords` | Без `#` |
| `maxPosts` | `pipeline.step1.cookie_search.size_per_keyword` | Лимит постов на ключевое слово |
| `cookies` | JSON-строка массива | Парсится из env (`cookies_json_string_for_actor`) |
| `sessionName` | `pipeline.step1.cookie_search.session_name` | Метка сессии в Apify |

### Ответ актора (сырой) → нормализация

Актор отдаёт свои поля; пайплайн берёт только строки со `status: "success"` и мапит в общий вид:

| Поле актора | Куда попадает |
|-------------|----------------|
| `post_url` | `url`, из URL — `shortCode` |
| `comment_count` | `commentsCount` |
| `like_count` | `likesCount` |
| `username` | `ownerUsername` |
| `caption`, `pub_date` | `caption`, `timestamp` |
| `media_urls[]` | для рилса — первый валидный HTTPS URL → `videoUrl` |
| `media_type` / URL с `/reel/` | `type: Video`, `productType: clips` |
| `location` | `locationName`, `locationId` |

Дальше: дедуп по `shortCode`, фильтр по возрасту, те же правила `min_comments` и `videoUrl`, что у других режимов.

**Цена:** по usage Apify (зависит от объёма keyword search).

---

## 4. instagram-comments-scraper — primary (`louisdeconinck/instagram-comments-scraper`)

**Step 3 PRIMARY.**

### Запрос (как шлёт `scripts/pipeline_lib/apify_runner.py`)

```json
{
  "urls": [
    "https://www.instagram.com/p/DFgH1jK2lMn/",
    "https://www.instagram.com/reel/ABC123/"
  ],
  "proxy": { "useApifyProxy": true },
  "resultsLimit": 10000,
  "maxComments": 10000
}
```

| Параметр | Обязателен | Описание |
|----------|------------|----------|
| `urls` | да | URL постов/рилсов (батч из очереди Step 3) |
| `proxy.useApifyProxy` | **да** | Без прокси актор часто отдаёт 0 items при `SUCCEEDED` |
| `resultsLimit` | **да** | Потолок комментов на пост (у нас = `maxComments`) |
| `maxComments` | **да** | То же значение, что `resultsLimit` |

Значение лимита: `pipeline.step3.louisdeconinck_comments_cap_per_post` (в коде default **10_000** — это потолок, не цель; актор вернёт только реально существующие комментарии, счёт в Apify не завышает).

Опционально у актора есть `cookies` (дешевле за коммент) — **пайплайн сейчас cookies не передаёт**.

### Ответ — snake_case (Instagram raw)

```json
{
  "pk": "17886529642832034",
  "user_id": "2880416097",
  "media_id": "1280676884715465116",
  "text": "Своя",
  "created_at_utc": 1608059608,
  "user": {
    "pk": "2880416097",
    "username": "thomastavellaa",
    "full_name": "Thomas",
    "is_private": false,
    "is_verified": false,
    "profile_pic_url": "https://..."
  }
}
```

**Как сохраняем:**
- Лид: `user.username`, `user.pk` → `lead_accounts` (дедуп по `user_id`, не по username)
- Связь с постом: `media_id` (float64, неточный) матчится к `shortcode` через `shortcode_to_id()` с допуском ±1000
- `text` → `lead_post_links` (обрезка 500 символов), `created_at_utc` → `comment_at`

**Цена (ориентир):** ~$1 / 1K комментов без cookies; с cookies дешевле.

---

## 5. instagram-comments-scraper-api — fallback (`apidojo/instagram-comments-scraper-api`)

**Step 3 FALLBACK** — только если primary вернул 0 items на весь батч.

### Запрос (как шлёт пайплайн)

```json
{
  "startUrls": ["https://www.instagram.com/p/DFgH1jK2lMn/"],
  "proxy": { "useApifyProxy": true },
  "maxItems": 50000
}
```

| Параметр | Описание |
|----------|----------|
| `startUrls` | Те же URL, что у primary |
| `proxy` | `useApifyProxy: true` |
| `maxItems` | **На весь run:** `apidojo_comments_cap_per_post × число_постов` (`pipeline.step3.apidojo_comments_cap_per_post`) |

Пайплайн **не** шлёт `postIds` — только `startUrls`.

### Ответ — camelCase → нормализация

Сырой ответ приводится к форме louisdeconinck в `src/comment_normalizer.normalize_apidojo_api`:

| apidojo | После нормализации |
|---------|-------------------|
| `message` | `text` |
| `createdAt` (ISO) | `created_at_utc` (unix) |
| `id` | `pk` |
| `postId` (shortcode) | `media_id` через `shortcode_to_id` — **точно**, без fuzzy |
| `user.fullName` | `user.full_name` |
| `user.isPrivate` / `isVerified` / `profilePicUrl` | snake_case |

**Цена (ориентир):** $0.0075 за post query + $0.0005 за коммент сверх 15 бесплатных на пост.

Primary оставляем из‑за схемы 1:1 с БД и Step 4; fallback — страховка при пустом primary.

---

## 6. instagram-profile-scraper (`apify/instagram-profile-scraper`)

**Step 4.**

### Запрос

```json
{
  "usernames": ["ivan_petrov", "anna_realtor_spb"]
}
```

Батчи до `profile_batch_size` (default 50). `includeAboutSection` пайплайн не включает.

### Ответ — что используем

- `username`, `id`, `fullName`, `biography`, `private`, `verified`
- `profilePicUrl` / `profilePicUrlHD` — скачивание аватара
- `externalUrl`, `externalUrls` + regex из bio → телефон, Telegram, WhatsApp, email
- `latestPosts` — fallback «лидер лица» из последних N фото поста (если на аватаре не ровно одно лицо)

**Цена:** ~$0.0026/профиль.

---

## Legacy и dev (не daily pipeline)

Эти акторы **не** вызываются из `scripts/pipeline.py` в штатном прогоне, но есть в `config.yaml` / `ApifyWrapper` / тестовых скриптах.

### `apify/instagram-comment-scraper` (`apify.actors.comments`)

Старый официальный скрапер комментов. Step 3 его **не** использует.

```json
{
  "directUrls": ["https://www.instagram.com/p/ABC/"],
  "resultsLimit": 50
}
```

Ответ: `ownerUsername`, `text`, `timestamp` — без `user_id` в том же виде, что у louisdeconinck.

### `apify/instagram-scraper` (`apify.actors.universal`)

Универсальный актор: поиск юзеров, посты по `directUrls`, комментарии через `resultsType`. В daily pipeline не задействован.

**Не использовать** `searchType=hashtag` — вернёт URL хештегов, а не посты.

---

## Оценка стоимости одного цикла (грубо)

Пример для СПБ (цифры ориентировочные; факт — в `logs/pipeline_*.json`):

**Step 1 (10 риелторов × 20 постов):**
- Посты: 200 × ~$0.0017 ≈ **$0.34**

**Step 3 (100 релевантных постов × ~130 комментов в среднем):**
- Primary louisdeconinck: ~13 000 комментов × ~$0.001 ≈ **$13** (без cookies; реальный счёт смотреть в Apify)
- При срабатывании fallback — добавится стоимость apidojo run + возможно «пустой» primary run

**Step 4 (500 новых лидов):**
- Профили: 500 × ~$0.0026 ≈ **$1.30**

Step 2 (DeepSeek/Nexara) и Step 5 (Sherlock) — отдельно от Apify.

---

## Где смотреть в коде

| Тема | Файл |
|------|------|
| Step 1 все режимы | `scripts/pipeline.py` (~строки 226–638) |
| Step 3 primary/fallback | `scripts/pipeline_lib/apify_runner.py` |
| Нормализация apidojo | `src/comment_normalizer.py` |
| Cookie search | `src/instagram_cookie_search.py` |
| Возраст, merge, videoUrl | `src/ig_media_payload.py` |
| Сравнение comment-скраперов | `scripts/test_comment_scrapers.py` |
