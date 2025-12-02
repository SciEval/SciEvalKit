DEFAULT_IGNORE_PATTERNS = [
    # OpenAI / Azure OpenAI
    "content_policy_violation",
    "content_filter",
    "safety system",
    "The response was filtered",

    # Gemini / Google
    "SAFETY",
    "finish_reason: SAFETY",
    "HARM_CATEGORY",

    # Claude / Anthropic
    "Output blocked by content filtering policy",

    # General
    "unsafe content",
    "policy violation"
]