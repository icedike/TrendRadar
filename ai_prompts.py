"""Prompt templates for the AI analysis pipeline."""

CLUSTER_PROMPT_TEMPLATE = """
You are the AI research editor for TrendRadar. Group the incoming crypto news
articles into distinct events so readers can avoid duplicates. Read the JSON
payload and output **only** JSON with an `events` array. For each event, include:
- event_id: stable snake_case identifier (e.g. mt_gox_btc_transfer)
- title: short, neutral headline (<= 12 words)
- theme: choose from [regulation, market, technology, defi, nft, personnel,
  security, institutional, macro, ecosystem]
- article_refs: list of article_id values to merge into this event
- rationale: 1 sentence on why the articles belong together
- confidence: value between 0 and 1 describing clustering certainty

Rules:
1. If the payload is empty, return {"events": []}.
2. Do not invent facts—derive titles and rationales from supplied content.
3. Prefer fewer, more meaningful events; singletons are allowed if unique.

Payload:
{payload}

Return strict JSON following the schema {{"events": [{{...}}]}}. Do not emit
any prose outside JSON.
"""

CLASSIFICATION_PROMPT = """
You are categorizing a single crypto news event for TrendRadar. Review the
event summary, then output JSON with keys `theme`, `subcategory`, and
`explanation`.

Themes: regulation, market, technology, defi, nft, personnel, security,
institutional, macro, ecosystem. If **none** of these fit, you may introduce a
new lowercase theme (snake_case) but only when necessary.

Guidance:
- `subcategory` should be a short keyword (e.g. "stablecoins", "venture_funding").
- `explanation` is one brief sentence linking the event to the chosen theme.
- If you create a new theme, justify why existing options do not apply and
  still provide a concise `subcategory` describing the focus.

Event summary: {event_summary}
"""

SUMMARY_PROMPT = """
Summarize the crypto news event below in at most **two sentences** and <= 60
words total. Cover what happened, the entities involved, and why it matters for
the crypto ecosystem. Maintain a neutral, analytical tone.

Event details:
{event_context}
"""

SCORING_PROMPT = """
Assign an importance score for crypto investors to the event below. Consider the
number of unique sources, market impact, regulatory weight, and time-sensitivity.
Return strict JSON formatted as {{"importance": x.x, "confidence": y.y}}, where
importance ranges from 1-10 and confidence from 0-1. Briefly justify the score in
a `justification` string (<= 20 words) describing the primary driver.

Event context:
{event_context}
"""
