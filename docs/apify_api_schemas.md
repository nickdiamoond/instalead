# Apify Instagram API — схемы запросов и ответов

## Общие принципы

- **Модель оплаты:** PAY_PER_EVENT — платим за каждый результат (item) в датасете, не за вычислительные ресурсы.
- **Поиск работает через Google/Facebook Ads**, а не через Instagram API напрямую. Результаты могут отличаться от того, что видит залогиненный пользователь.
- **Пагинация** — внутренняя. Управляется через `resultsLimit`.

---

## 1. instagram-hashtag-scraper (`apify/instagram-hashtag-scraper`)

**Основной актор для поиска постов по хештегам. Возвращает ПОСТЫ, не URL хештегов.**

### Запрос (INPUT)

```json
{
  "hashtags": ["недвижимостьспб", "квартираспб"],   // список хештегов (без #)
  "resultsType": "posts",     // "posts" | "reels"
  "resultsLimit": 20,         // макс. постов на хештег (min: 1, default: 20)
  "keywordSearch": false       // если true — ищет по ключевому слову, не хештегу
}
```

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `hashtags` | `string[]` | **Да** | Хештеги без `#`. Без спецсимволов и пробелов |
| `resultsType` | `enum` | Нет | `"posts"` (default) или `"reels"` |
| `resultsLimit` | `int` | Нет | Макс. результатов на хештег. Default: 20 |
| `keywordSearch` | `bool` | Нет | Поиск по ключевому слову вместо хештега |

### Ответ (OUTPUT) — массив постов

```json
{
  "inputUrl": "https://www.instagram.com/explore/tags/квартираспб",
  "id": "3578291234567890",
  "type": "Video",                    // "Image" | "Video" | "Sidecar"
  "shortCode": "DFgH1jK2lMn",
  "caption": "Обзор квартиры 65м² в ЖК...",
  "hashtags": ["квартираспб", "новостройка"],
  "mentions": ["@developer_spb"],
  "url": "https://www.instagram.com/p/DFgH1jK2lMn/",
  "commentsCount": 47,
  "firstComment": "Сколько стоит?",
  "latestComments": [
    {
      "id": "12345",
      "text": "+",
      "ownerUsername": "ivan_petrov",
      "ownerProfilePicUrl": "https://...",
      "timestamp": "2026-03-15T10:30:00.000Z",
      "likesCount": 0,
      "repliesCount": 0
    }
  ],
  "displayUrl": "https://scontent-...",
  "images": ["https://scontent-..."],
  "likesCount": 350,
  "timestamp": "2026-03-10T14:00:00.000Z",
  "ownerFullName": "Анна Риелтор",
  "ownerUsername": "anna_realtor_spb",
  "ownerId": "1234567890",
  "locationName": "Санкт-Петербург",
  "locationId": "12345678",
  "productType": "clips",             // "clips" = Reels, "feed" = обычный пост
  "musicInfo": { ... }
}
```

**Ключевые поля для нашего пайплайна:**
- `type` / `productType` — отличить Reel от поста (`"Video"` + `"clips"` = Reel)
- `commentsCount` — фильтр: обрабатываем только если >= min_comments
- `timestamp` — фильтр по возрасту поста
- `latestComments[].ownerUsername` — ЛИДЫ (но только несколько последних комментов)
- `shortCode` / `url` — для запроса полных комментариев отдельным актором

**Цена:** ~$0.0023/пост

---

## 2. instagram-comment-scraper (`apify/instagram-comment-scraper`)

**Получает комментарии к конкретному посту/рилсу.**

### Запрос (INPUT)

```json
{
  "directUrls": [
    "https://www.instagram.com/p/DFgH1jK2lMn/",
    "https://www.instagram.com/reel/ABC123/"
  ],
  "resultsLimit": 50,              // макс. комментариев на пост (default: 15)
  "includeNestedComments": false,  // ПЛАТНЫЙ: включить ответы на комменты
  "isNewestComments": false        // ПЛАТНЫЙ: сначала новые
}
```

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `directUrls` | `string[]` | **Да** | URL постов/рилсов (должен содержать `/p/` или `/reel/`) |
| `resultsLimit` | `int` | Нет | Макс. комментов на пост. Default: 15 |
| `includeNestedComments` | `bool` | Нет | Включить ответы (вложенные). Платная функция |
| `isNewestComments` | `bool` | Нет | Сортировка: сначала новые. Платная функция |

### Ответ (OUTPUT) — массив комментариев

```json
{
  "id": "17890012345678",
  "text": "+",
  "ownerUsername": "ivan_petrov",
  "ownerProfilePicUrl": "https://scontent-.../150x150/...",
  "timestamp": "2026-03-15T10:30:00.000Z",
  "likesCount": 0,
  "repliesCount": 2,
  "replies": []            // пустой если includeNestedComments=false
}
```

**Ключевые поля для нашего пайплайна:**
- `ownerUsername` — **ЭТО ЛИД** (Instagram username комментатора)
- `ownerProfilePicUrl` — урезанная аватарка (150x150, для face detection позже)
- `text` — текст комментария
- `timestamp` — время комментария

**Ограничения:**
- Бесплатно: только 15 комментов, отсортированных по популярности
- Starter+ план: полный доступ, до 50+ комментов

**Цена:** ~$0.0023/комментарий

> **Внимание:** этот актор НЕ используется в основном пайплайне. Pipeline.py
> (Step 3) использует `louisdeconinck/instagram-comments-scraper` как primary
> и `apidojo/instagram-comments-scraper-api` как fallback (см. секции 2a и 2b
> ниже). `apify/instagram-comment-scraper` остался как `apify.actors.comments`
> в `config.yaml` для legacy test-скриптов через `ApifyWrapper.get_comments()`.

---

## 2a. instagram-comments-scraper (`louisdeconinck/instagram-comments-scraper`)

**Step 3 PRIMARY.** Используется в `scripts/pipeline.py` для основного забора
комментов под все relevant посты в очереди.

### Запрос (INPUT)

```json
{
  "urls": [
    "https://www.instagram.com/p/DFgH1jK2lMn/",
    "DFgH1jK2lMn"                    // shortcode тоже принимается
  ],
  "maxComments": 100,                // опц. макс. комментов на пост (default: всё)
  "cookies": "..."                   // опц. -- даёт 5x скидку на коммент
}
```

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `urls` | `string[]` | **Да** | URL/shortcode постов или рилсов |
| `maxComments` | `int` | Нет | Cap комментов на пост. Default: бесконечность |
| `cookies` | `string` | Нет | Куки IG — снижает цену с $0.001 до $0.0002 за коммент |

### Ответ (OUTPUT) — Instagram-raw snake_case

```json
{
  "pk": "17886529642832034",          // ID комментария
  "user_id": "2880416097",
  "media_id": "1280676884715465116",  // float64 -- неточный, см. fuzzy match ниже
  "text": "Своя",
  "created_at_utc": 1608059608,       // unix int
  "comment_like_count": 3,
  "child_comment_count": 0,
  "user": {
    "pk": "2880416097",
    "id": "2880416097",
    "username": "thomastavellaa",
    "full_name": "Thomas",
    "is_private": false,
    "is_verified": false,
    "profile_pic_url": "https://scontent-.../150x150/..."
  }
}
```

**Ключевые поля для пайплайна:**
- `user.username` / `user.pk` — лид (его username + permanent numeric id)
- `user.full_name`, `is_private`, `is_verified`, `profile_pic_url` — снимаются
  один в один в `lead_accounts`
- `text` — текст коммента (обрезается до 500 символов в `lead_post_links`)
- `created_at_utc` — unix int, идёт в `lead_post_links.comment_at`
- `media_id` — float64, теряет точность; матчится на `processed_posts.shortcode`
  через `shortcode_to_id()` с допуском ±1000

**Цена:**
- $0.001 за run + $0.001 за коммент (без cookies) ≈ ~$1/1K комментов
- $0.001 за run + $0.0002 за коммент (с cookies) ≈ ~$0.20/1K комментов

**Известный issue (актуально на 2026-04):** актор иногда возвращает 0 items
с `status=SUCCEEDED` для всего батча URL — фактически тихий фейл. Step 3
автоматически фолбэкается на `apidojo/instagram-comments-scraper-api` (секция
2b). Если оба актора отдают пусто, посты остаются в очереди для повторной
попытки на следующем запуске пайплайна.

---

## 2b. instagram-comments-scraper-api (`apidojo/instagram-comments-scraper-api`)

**Step 3 FALLBACK.** Активируется автоматически в `scripts/pipeline.py` когда
primary возвращает пустой датасет на весь батч. Это standby/realtime актор от
apidojo; его camelCase output нормализуется в louisdeconinck-форму через
`src.comment_normalizer.normalize_apidojo_api` перед сохранением.

### Запрос (INPUT)

```json
{
  "startUrls": [                      // плоский массив СТРОК (НЕ объектов)
    "https://www.instagram.com/p/DFgH1jK2lMn/"
  ],
  "postIds": ["DFgH1jK2lMn"],         // альтернативный вход
  "maxItems": 200                     // опц. cap по всему run'у
}
```

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `startUrls` | `string[]` | да* | URL постов. *хотя бы одно из startUrls/postIds |
| `postIds` | `string[]` | да* | shortcodes — альтернатива startUrls |
| `maxItems` | `int` | Нет | Total cap на весь run (не на пост!). Default: бесконечность |

### Ответ (OUTPUT) — camelCase

```json
{
  "inputSource": "https://www.instagram.com/p/DXdv7B1jFDF",
  "postId": "DXdv7B1jFDF",            // shortcode напрямую (НЕ float64!)
  "type": "comment",
  "id": "17919079491232885",
  "userId": "6677757292",
  "message": "Great post!",            // == text у louisdeconinck
  "createdAt": "2026-04-26T20:36:33.000Z",  // ISO 8601, не unix
  "likeCount": 3,
  "replyCount": 1,
  "user": {
    "id": "6677757292",
    "username": "alice",
    "fullName": "Alice",                // camelCase
    "isVerified": true,
    "isPrivate": false,
    "profilePicUrl": "https://..."
  },
  "isRanked": true
}
```

**Различия с louisdeconinck (применяет `normalize_apidojo_api`):**

| louisdeconinck | apidojo-api | конверсия |
|---|---|---|
| `text` | `message` | `text` ← `message` |
| `created_at_utc` (unix int) | `createdAt` (ISO 8601) | parse через `datetime.fromisoformat` → unix int |
| `pk` | `id` | `pk` ← `id` |
| `media_id` (float64 imprecise) | `postId` (shortcode) | `shortcode_to_id(postId)` — точно, без допуска |
| `user.pk` / `user_id` | `user.id` / `userId` | оба читаются через `_safe_str` |
| `user.full_name` | `user.fullName` | snake_case |
| `user.is_private` | `user.isPrivate` | snake_case |
| `user.is_verified` | `user.isVerified` | snake_case |
| `user.profile_pic_url` | `user.profilePicUrl` | snake_case |

**Цена:** $0.0075 за post query + $0.0005 за коммент сверх первых 15 бесплатных
- 1 пост × 100 комментов = $0.0075 + 85 × $0.0005 = $0.05
- 50 постов × 200 комментов = $0.375 + 9250 × $0.0005 = $5.00

На нашем масштабе apidojo-api оказывается дешевле louisdeconinck без cookies
(~$0.50/1K vs $1/1K), но louisdeconinck остаётся primary потому что schema
совпадает с downstream без нормализации.

---

## 3. instagram-profile-scraper (`apify/instagram-profile-scraper`)

**Получает информацию о профиле по username.**

### Запрос (INPUT)

```json
{
  "usernames": ["ivan_petrov", "anna_realtor_spb"],
  "includeAboutSection": false    // ПЛАТНЫЙ: дата регистрации, страна
}
```

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `usernames` | `string[]` | **Да** | Юзернеймы, URL профилей или ID |
| `includeAboutSection` | `bool` | Нет | Доп. инфо (дата регистрации, страна). Платная функция |

### Ответ (OUTPUT) — массив профилей

```json
{
  "inputUrl": "https://www.instagram.com/ivan_petrov/",
  "id": "1234567890",
  "username": "ivan_petrov",
  "url": "https://www.instagram.com/ivan_petrov/",
  "fullName": "Иван Петров",
  "biography": "Живу в СПб, люблю путешествия",
  "followersCount": 543,
  "followsCount": 321,
  "postsCount": 87,
  "isBusinessAccount": false,
  "businessCategoryName": null,
  "private": false,
  "verified": false,
  "profilePicUrl": "https://scontent-.../150x150/...",
  "profilePicUrlHD": "https://scontent-.../1080x1080/...",
  "externalUrl": "https://t.me/ivan_petrov",
  "externalUrls": [
    {"title": "Telegram", "url": "https://t.me/ivan_petrov", "link_type": "..."}
  ],
  "relatedProfiles": [ ... ],
  "latestPosts": [ ... ]
}
```

**Ключевые поля для нашего пайплайна:**
- `private` — если true, фотографии недоступны (пропуск в Module 2)
- `profilePicUrlHD` — аватарка в высоком качестве для face detection
- `externalUrl` — может содержать Telegram, WhatsApp, телефон
- `biography` — может содержать контакты
- `fullName` — для поиска в Telegram

**Цена:** ~$0.0026/профиль

---

## 4. instagram-scraper (`apify/instagram-scraper`) — универсальный

**Используем ТОЛЬКО для: поиска юзеров и получения постов по directUrls.**

### Вариант A: Поиск аккаунтов (риелторов)

```json
{
  "search": "риелтор спб квартира",
  "searchType": "user",
  "searchLimit": 10,
  "resultsType": "details",
  "proxy": { "useApifyProxy": true }
}
```

Ответ — профили (аналогично profile-scraper).

### Вариант B: Посты аккаунта по directUrls

```json
{
  "directUrls": ["https://www.instagram.com/anna_realtor_spb/"],
  "resultsType": "posts",
  "resultsLimit": 20,
  "onlyPostsNewerThan": "30 days",
  "proxy": { "useApifyProxy": true }
}
```

Ответ — массив постов (аналогично hashtag-scraper output).

### Вариант C: Комментарии к посту по directUrls

```json
{
  "directUrls": ["https://www.instagram.com/p/DFgH1jK2lMn/"],
  "resultsType": "comments",
  "resultsLimit": 50,
  "proxy": { "useApifyProxy": true }
}
```

Ответ — массив комментариев (аналогично comment-scraper).

**Цена:** ~$0.0023/результат

### НЕ использовать для:
- `searchType="hashtag"` — возвращает URL хештегов, а не посты
- `searchType="place"` — возвращает метаданные локаций, а не посты

---

## 5. instagram-post-scraper (`apify/instagram-post-scraper`)

**Посты конкретного аккаунта. Альтернатива universal scraper для Method 2.**

### Запрос (INPUT)

```json
{
  "username": ["anna_realtor_spb"],
  "resultsLimit": 24,
  "onlyPostsNewerThan": "30 days",
  "skipPinnedPosts": false,
  "dataDetailLevel": "basicData"    // "basicData" (дешевле) | "detailedData"
}
```

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `username` | `string[]` | **Да** | Юзернеймы или URL профилей |
| `resultsLimit` | `int` | Нет | Макс. постов на аккаунт. Default: 24 |
| `onlyPostsNewerThan` | `string` | Нет | Фильтр по дате |
| `skipPinnedPosts` | `bool` | Нет | Пропустить закреплённые посты |
| `dataDetailLevel` | `enum` | Нет | `"basicData"` / `"detailedData"` |

**Цена:**
- basicData: ~$0.0017/пост
- detailedData: ~$0.0027/пост (включает latest comments)

---

## Наш пайплайн: какой актор для чего

```
МЕТОД 1 (поиск по хештегам):
  instagram-hashtag-scraper  →  [посты]  →  instagram-comment-scraper  →  [лиды]
         $0.0023/пост                              $0.0023/коммент

МЕТОД 2 (от списка риелторов):
  instagram-post-scraper     →  [посты]  →  instagram-comment-scraper  →  [лиды]
       $0.0017/пост                              $0.0023/коммент

ДОПОЛНИТЕЛЬНО:
  instagram-profile-scraper  →  [профиль лида: аватарка, bio, приватность]
         $0.0026/профиль
```

### Оценка стоимости одного цикла (СПБ)

```
Метод 1: 5 хештегов × 20 постов × 20 комментов = 100 постов + 2000 комментов
  Посты:    100 × $0.0023 = $0.23
  Комменты: 2000 × $0.0023 = $4.60
  Итого: ~$4.83

Метод 2: 10 риелторов × 10 постов × 20 комментов = 100 постов + 2000 комментов
  Посты:    100 × $0.0017 = $0.17
  Комменты: 2000 × $0.0023 = $4.60
  Итого: ~$4.77

Профили лидов: 500 уникальных × $0.0026 = $1.30

ОБЩАЯ ОЦЕНКА: ~$6-11 за цикл
```
