"""Extract phone numbers, Telegram, WhatsApp, email from Instagram bio and URLs."""

import re


def extract_contacts(bio: str | None, external_url: str | None, external_urls: list | None) -> dict:
    """Extract all contacts from profile bio and links.

    Returns dict with keys: phone, telegram_username, whatsapp, email.
    Each value is a string or None.
    """
    texts = []
    if bio:
        texts.append(bio)
    if external_url:
        texts.append(external_url)
    for eu in (external_urls or []):
        if isinstance(eu, dict):
            texts.append(eu.get("url", ""))
            texts.append(eu.get("title", ""))
        elif isinstance(eu, str):
            texts.append(eu)

    combined = " ".join(texts)

    return {
        "phone": extract_phone(combined),
        "telegram_username": extract_telegram(combined),
        "whatsapp": extract_whatsapp(combined),
        "email": extract_email(combined),
    }


def extract_phone(text: str) -> str | None:
    """Extract Russian phone number from text.

    Handles formats:
      +7 (999) 123-45-67, +7 999 123 45 67, +79991234567
      8(999)123-45-67, 8 999 123 45 67, 89991234567
      +7-999-123-45-67, 7 999 1234567
      with dots, dashes, spaces, brackets in any combination
    """
    # Normalize: remove common decorative chars but keep structure
    cleaned = text.replace("☎", " ").replace("📞", " ").replace("📲", " ")
    cleaned = cleaned.replace("\u00a0", " ")  # non-breaking space

    # Pattern: +7 or 8, then 10 digits with optional separators
    patterns = [
        # +7 or 7 followed by 10 digits with separators
        r'(?:\+7|(?<!\d)7)[\s.\-]*\(?(\d{3})\)?[\s.\-]*(\d{3})[\s.\-]*(\d{2})[\s.\-]*(\d{2})',
        # 8 followed by 10 digits with separators
        r'(?<!\d)8[\s.\-]*\(?(\d{3})\)?[\s.\-]*(\d{3})[\s.\-]*(\d{2})[\s.\-]*(\d{2})',
        # Compact: +7XXXXXXXXXX or 8XXXXXXXXXX
        r'(?:\+7|(?<!\d)8)(\d{10})',
    ]

    for pattern in patterns:
        m = re.search(pattern, cleaned)
        if m:
            groups = m.groups()
            if len(groups) == 4:
                digits = "".join(groups)
            elif len(groups) == 1:
                digits = groups[0]
            else:
                continue
            if len(digits) == 10:
                return f"+7{digits}"

    return None


def extract_telegram(text: str) -> str | None:
    """Extract Telegram username from text.

    Handles:
      @username, t.me/username, telegram.me/username
      tg: username, telegram: @username
    """
    # t.me/username or telegram.me/username (not invite links)
    m = re.search(r'(?:t\.me|telegram\.me)/([a-zA-Z]\w{3,30})(?:\b|$)', text)
    if m:
        username = m.group(1)
        # Skip invite links (t.me/+xxx, t.me/joinchat)
        if not username.startswith("+") and username.lower() != "joinchat":
            return username

    # @username pattern (not email)
    m = re.search(r'(?:тг|tg|telegram|телеграм)[:\s]*@?([a-zA-Z]\w{3,30})', text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Standalone @username (heuristic: after "тг"/"telegram" context or just @)
    # Be careful not to grab Instagram @mentions
    m = re.search(r'(?:тг|телеграм|telegram|tg)\s*[:\-]?\s*@([a-zA-Z]\w{3,30})', text, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def extract_whatsapp(text: str) -> str | None:
    """Extract WhatsApp contact from text.

    Handles:
      wa.me/79991234567, api.whatsapp.com/send?phone=79991234567
      WhatsApp: +7 999 123 45 67
    """
    # wa.me/number
    m = re.search(r'wa\.me/(\+?\d{10,15})', text)
    if m:
        return normalize_wa_phone(m.group(1))

    # api.whatsapp.com/send?phone=number
    m = re.search(r'whatsapp\.com/send\?phone=(\d{10,15})', text)
    if m:
        return normalize_wa_phone(m.group(1))

    # "WhatsApp" or "WA" followed by phone-like pattern
    m = re.search(
        r'(?:whatsapp|WA|вотсап|ватсап)[:\s]*(\+?[78][\s.\-]*\(?\d{3}\)?[\s.\-]*\d{3}[\s.\-]*\d{2}[\s.\-]*\d{2})',
        text, re.IGNORECASE,
    )
    if m:
        phone = extract_phone(m.group(1))
        if phone:
            return phone

    return None


def extract_email(text: str) -> str | None:
    """Extract email address from text."""
    m = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    return m.group(0) if m else None


def normalize_wa_phone(raw: str) -> str:
    """Normalize WhatsApp phone to +7XXXXXXXXXX format."""
    digits = re.sub(r'\D', '', raw)
    if digits.startswith("7") and len(digits) == 11:
        return f"+{digits}"
    if digits.startswith("8") and len(digits) == 11:
        return f"+7{digits[1:]}"
    if len(digits) == 10:
        return f"+7{digits}"
    return f"+{digits}"
