"""
fetch_news_sentiment.py
-----------------------------------------------------------------------------
Fetches financial news from multiple free RSS feeds + optional Finnhub API,
runs VADER sentiment analysis with a financial keyword booster, and saves:

  newsData/news_latest.json        — last 60 articles with per-article scores
  newsData/news_history.csv        — append-only deduped archive
  newsData/sentiment_summary.json  — session-level bull/bear aggregate scores

Sources (all free, no login required except Finnhub):
  1. MoneyControl Top News      RSS
  2. MoneyControl Markets       RSS
  3. MoneyControl Commodities   RSS  ← crude oil, gold
  4. LiveMint Markets           RSS
  5. LiveMint Economy           RSS
  6. Finnhub General News       API  (optional — set FINNHUB_API_KEY secret)

Sentiment engine:
  • VADER (vaderSentiment) — fast, offline, no API cost
  • Financial keyword booster: custom score adjustments for market-specific
    terms (FII, crude, RBI, breakout, sell-off, etc.)
"""

import os, json, csv, hashlib, time, re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from html import unescape

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("WARNING: vaderSentiment not installed. Run: pip install vaderSentiment")

# -- Output paths --------------------------------------------------------------
OUTPUT_DIR       = "newsData"
LATEST_JSON      = os.path.join(OUTPUT_DIR, "news_latest.json")
HISTORY_CSV      = os.path.join(OUTPUT_DIR, "news_history.csv")
SUMMARY_JSON     = os.path.join(OUTPUT_DIR, "sentiment_summary.json")
SEEN_IDS_FILE    = os.path.join(OUTPUT_DIR, ".seen_ids.txt")

MAX_LATEST       = 60    # articles kept in news_latest.json
HISTORY_COLS     = ["id", "fetched_at", "published", "source", "title",
                    "summary", "url", "compound", "pos", "neg", "neu",
                    "fin_score", "label"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# -- RSS feed registry ---------------------------------------------------------
RSS_FEEDS = [
    ("MoneyControl Top",   "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("MoneyControl Mkts",  "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("MoneyControl Comm",  "https://www.moneycontrol.com/rss/commodities.xml"),
    ("LiveMint Markets",   "https://www.livemint.com/rss/markets"),
    ("LiveMint Economy",   "https://www.livemint.com/rss/economy"),
]

# -- Financial sentiment keyword booster --------------------------------------
# Score added to VADER compound for each keyword match (case-insensitive).
# Positive = bullish signal, Negative = bearish signal.
FIN_KEYWORDS = {
    # Strong bullish
    "breakout":          +0.25, "all-time high":     +0.30, "rally":          +0.20,
    "bull run":          +0.30, "record high":       +0.30, "strong buying":  +0.25,
    "fii buying":        +0.30, "rate cut":          +0.25, "stimulus":       +0.20,
    "trade deal":        +0.25, "ceasefire":         +0.20, "gdp growth":     +0.20,
    "short covering":    +0.20, "golden cross":      +0.25, "oversold":       +0.15,
    # Mild bullish
    "gains":             +0.10, "advances":          +0.10, "uptick":         +0.10,
    "recovery":          +0.12, "bounce":            +0.10, "support":        +0.08,
    # Strong bearish
    "sell-off":          -0.30, "crash":             -0.35, "plunge":         -0.30,
    "fii selling":       -0.30, "rate hike":         -0.25, "recession":      -0.30,
    "tariff":            -0.20, "sanctions":         -0.25, "war":            -0.30,
    "debt default":      -0.35, "death cross":       -0.30, "overbought":     -0.15,
    "panic":             -0.30, "circuit breaker":   -0.35, "crude surge":    -0.20,
    "oil spike":         -0.20, "inflation":         -0.15, "stagflation":    -0.25,
    # Mild bearish
    "falls":             -0.10, "declines":          -0.10, "slips":          -0.10,
    "concern":           -0.08, "uncertainty":       -0.10, "volatility":     -0.08,
    # Crude / USD/INR specific
    "crude rises":       -0.20, "brent up":          -0.15, "oil rally":      -0.15,
    "rupee weakens":     -0.20, "dollar strength":   -0.15, "inr depreciation":-0.20,
    "rupee strengthens": +0.15, "dollar weakness":   +0.10,
    # RBI / macro
    "rbi cuts":          +0.25, "rbi hikes":         -0.20, "rbi pause":       0.00,
    "cpi below":         +0.15, "cpi above":         -0.15,
}


# -- Helpers -------------------------------------------------------------------
def _ist_now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")


def _article_id(url: str, title: str) -> str:
    return hashlib.md5(f"{url}{title}".encode()).hexdigest()[:12]


def _clean(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)    # strip HTML tags
    text = re.sub(r"#\d+;", " ", text)      # strip leftover HTML entities
    return re.sub(r"\s+", " ", text).strip()


def _fetch_url(url: str, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as r:
                return r.read()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1)


# -- RSS parser ----------------------------------------------------------------
def parse_rss(name: str, url: str) -> list[dict]:
    articles = []
    try:
        raw  = _fetch_url(url)
        root = ET.fromstring(raw)
    except Exception as e:
        print(f"  RSS fetch failed [{name}]: {e}")
        return articles

    for item in root.findall(".//item"):
        title   = _clean(item.findtext("title",       ""))
        summary = _clean(item.findtext("description", ""))
        link    = (item.findtext("link", "") or "").strip()
        pub     = (item.findtext("pubDate", "") or "").strip()

        if not title:
            continue

        articles.append({
            "source":    name,
            "title":     title,
            "summary":   summary[:300],
            "url":       link,
            "published": pub,
        })
    return articles


# -- Finnhub parser ------------------------------------------------------------
def fetch_finnhub(api_key: str) -> list[dict]:
    if not api_key:
        return []
    articles = []
    categories = ["general", "forex", "crypto"]
    for cat in categories:
        url = f"https://finnhub.io/api/v1/news?category={cat}&token={api_key}"
        try:
            raw  = _fetch_url(url)
            data = json.loads(raw)
            for item in data:
                title   = _clean(item.get("headline", ""))
                summary = _clean(item.get("summary",  ""))
                if not title:
                    continue
                ts = item.get("datetime", 0)
                pub = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%a, %d %b %Y %H:%M:%S +0000") if ts else ""
                articles.append({
                    "source":    f"Finnhub/{cat}",
                    "title":     title,
                    "summary":   summary[:300],
                    "url":       item.get("url", ""),
                    "published": pub,
                })
            time.sleep(0.4)   # Finnhub: 60 calls/min; 3 calls here is fine
        except Exception as e:
            print(f"  Finnhub [{cat}] failed: {e}")
    return articles


# -- Sentiment scoring ---------------------------------------------------------
def _fin_boost(text: str) -> float:
    text_lower = text.lower()
    return sum(score for kw, score in FIN_KEYWORDS.items() if kw in text_lower)


def score_article(article: dict, analyzer) -> dict:
    text = f"{article['title']} {article['summary']}"

    if VADER_AVAILABLE and analyzer:
        vs       = analyzer.polarity_scores(text)
        compound = vs["compound"]
        pos      = vs["pos"]
        neg      = vs["neg"]
        neu      = vs["neu"]
    else:
        compound = pos = neg = neu = 0.0

    fin_boost = _fin_boost(text)
    fin_score = round(max(-1.0, min(1.0, compound + fin_boost)), 4)

    if fin_score >= 0.15:
        label = "BULLISH"
    elif fin_score <= -0.15:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {**article,
            "compound":  round(compound, 4),
            "pos":       round(pos, 4),
            "neg":       round(neg, 4),
            "neu":       round(neu, 4),
            "fin_score": fin_score,
            "label":     label}


# -- Deduplication -------------------------------------------------------------
def _load_seen() -> set:
    if os.path.exists(SEEN_IDS_FILE):
        with open(SEEN_IDS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _save_seen(seen: set):
    # Keep only last 2000 IDs to avoid file bloat
    ids = list(seen)[-2000:]
    with open(SEEN_IDS_FILE, "w") as f:
        f.write("\n".join(ids))


# -- Session sentiment summary -------------------------------------------------
def build_summary(scored: list[dict]) -> dict:
    if not scored:
        return {}

    bull   = [a for a in scored if a["label"] == "BULLISH"]
    bear   = [a for a in scored if a["label"] == "BEARISH"]
    neut   = [a for a in scored if a["label"] == "NEUTRAL"]
    avg_fs = sum(a["fin_score"] for a in scored) / len(scored)

    # Weighted score: recent articles count more (list is newest-first)
    weights     = [1 / (i + 1) for i in range(len(scored))]
    w_total     = sum(weights)
    weighted_fs = sum(a["fin_score"] * w for a, w in zip(scored, weights)) / w_total

    if weighted_fs >= 0.12:
        verdict = "BULLISH"
        color   = "#28a745"
    elif weighted_fs <= -0.12:
        verdict = "BEARISH"
        color   = "#dc3545"
    else:
        verdict = "NEUTRAL"
        color   = "#6c757d"

    return {
        "generated_at":    _ist_now(),
        "total_articles":  len(scored),
        "bullish_count":   len(bull),
        "bearish_count":   len(bear),
        "neutral_count":   len(neut),
        "avg_fin_score":   round(avg_fs,      4),
        "weighted_score":  round(weighted_fs, 4),
        "verdict":         verdict,
        "verdict_color":   color,
        "top_bullish":     [a["title"] for a in bull[:3]],
        "top_bearish":     [a["title"] for a in bear[:3]],
        "sources_hit":     list({a["source"] for a in scored}),
    }


# -- Persistence ---------------------------------------------------------------
def save_all(new_articles: list[dict], seen: set):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fetched_at = _ist_now()

    # -- Assign IDs and timestamp ----------------------------------------------
    for a in new_articles:
        a["id"]         = _article_id(a["url"], a["title"])
        a["fetched_at"] = fetched_at

    new_ids = {a["id"] for a in new_articles}

    # -- news_latest.json -----------------------------------------------------
    existing_latest = []
    if os.path.exists(LATEST_JSON):
        try:
            with open(LATEST_JSON) as f:
                existing_latest = json.load(f).get("articles", [])
        except Exception:
            pass

    # Merge: new articles first, then existing (drop duplicates, cap at MAX_LATEST)
    merged_ids  = set()
    merged_list = []
    for a in new_articles + existing_latest:
        if a["id"] not in merged_ids:
            merged_ids.add(a["id"])
            merged_list.append(a)
        if len(merged_list) >= MAX_LATEST:
            break

    summary = build_summary(merged_list)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated_at": fetched_at,
                   "summary": summary,
                   "articles": merged_list}, f, indent=2, ensure_ascii=False)

    # -- sentiment_summary.json -----------------------------------------------
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -- news_history.csv (append-only, new articles only) --------------------
    write_header = not os.path.exists(HISTORY_CSV)
    truly_new    = [a for a in new_articles if a["id"] not in seen]
    if truly_new:
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(truly_new)

    # -- Update seen IDs -------------------------------------------------------
    seen.update(new_ids)
    _save_seen(seen)

    return len(truly_new), summary


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[{_ist_now()}] Starting news sentiment fetch…")

    analyzer      = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
    finnhub_key   = os.getenv("FINNHUB_API_KEY", "")
    seen          = _load_seen()

    # -- Collect from all sources ----------------------------------------------
    all_articles: list[dict] = []

    for name, url in RSS_FEEDS:
        print(f"  Fetching RSS: {name}…")
        arts = parse_rss(name, url)
        print(f"    -> {len(arts)} articles")
        all_articles.extend(arts)

    if finnhub_key:
        print("  Fetching Finnhub…")
        finn_arts = fetch_finnhub(finnhub_key)
        print(f"    -> {len(finn_arts)} articles")
        all_articles.extend(finn_arts)
    else:
        print("  Finnhub skipped (FINNHUB_API_KEY not set)")

    print(f"  Total fetched: {len(all_articles)} articles")

    if not all_articles:
        print("No articles fetched. Exiting.")
        exit(0)

    # -- Score sentiment -------------------------------------------------------
    scored = [score_article(a, analyzer) for a in all_articles]

    # -- Save -----------------------------------------------------------------
    new_count, summary = save_all(scored, seen)

    # -- Print summary ---------------------------------------------------------
    print(f"\n  -- Sentiment Summary ------------------------------")
    print(f"  Verdict:        {summary.get('verdict')} "
          f"(score: {summary.get('weighted_score'):+.3f})")
    print(f"  Articles:       {summary.get('total_articles')} total  "
          f"| {summary.get('bullish_count')} bull  "
          f"| {summary.get('bearish_count')} bear  "
          f"| {summary.get('neutral_count')} neutral")
    print(f"  New to history: {new_count}")
    if summary.get("top_bullish"):
        print(f"  Top bullish:    {summary['top_bullish'][0][:80]}")
    if summary.get("top_bearish"):
        print(f"  Top bearish:    {summary['top_bearish'][0][:80]}")
    print(f"  ---------------------------------------------------\n")
