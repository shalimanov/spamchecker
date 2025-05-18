from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("artifacts")

SPAM_KEYWORDS = [
    r"\bfree\b", r"\bwinner?\b", r"\bprize\b", r"\bcredit\b",
    r"\bcheap\b", r"\bdiscount\b", r"\bbuy\s+now\b",
    r"\bwork\s+from\s+home\b", r"\bmake\s+\$?\d+k?\b",
    r"\bclick\s+here\b", r"\bvisit\s+my\s+profile\b",
    r"\blimited\s+offer\b", r"\bget\s+rich\b", r"\bdouble\s+your\s+income\b",
    r"\bno\s+experience\s+needed\b", r"\bearn\s+\$?\d+\s+a\s+day\b",
    r"\bapply\s+now\b", r"\bcongratulations\b", r"\bguaranteed\b",
    r"\bmiracle\b", r"\bamazing\s+deal\b", r"\bact\s+fast\b",
    r"\bno\s+credit\s+check\b", r"\bsecret\s+to\s+success\b",
    r"\blose\s+weight\s+fast\b", r"\bget\s+followers\b",
    r"\binvest\s+now\b", r"\bmake\s+money\s+fast\b",
    r"\bexclusive\s+deal\b", r"\b100%\s+free\b",
    r"\bbest\s+price\b", r"\bsatisfaction\s+guaranteed\b",
    r"\bwin\s+cash\b", r"\bextra\s+income\b",
    r"\bjoin\s+now\b", r"\bzero\s+risk\b",
    r"\bearn\s+extra\s+cash\b", r"\bincredible\s+offer\b",
    r"\bget\s+started\s+today\b", r"\bviral\s+post\b",
    r"\btrending\s+deal\b", r"\bincrease\s+traffic\b",
    r"\bgrow\s+your\s+account\b", r"\bsubscribe\s+to\s+my\s+channel\b",
    r"\bhot\s+deal\b", r"\bnew\s+followers\b", r"\bjust\s+launched\b",
    r"\bsign\s+up\s+free\b", r"\bbonus\s+included\b",
    r"\bdon’t\s+miss\s+out\b", r"\bguaranteed\s+results\b"
]

SPAM_TEMPLATES = [
    "check out my channel", "visit my website", "biggest sale of the year",
    "earn money online", "contact on whatsapp <NUM>", "earn €<NUM> from home, click here",
    "make thousands weekly", "subscribe for free", "join our giveaway",
    "work from home opportunity", "buy followers now",
    "get instant access", "limited time deal", "claim your reward",
    "hot singles near you", "dm me for details", "try it risk free",
    "only a few spots left", "lowest price online", "message me to know more",
    "start earning today", "follow me for more tips", "boost your profile fast",
    "free download here", "we're hiring", "instant approval loans",
    "one-time offer only", "get your bonus now", "trusted by thousands",
    "win an iPhone", "no skills required", "today only",
    "earn from your phone", "double your sales", "follow us for updates",
    "apply for free", "don’t miss this chance", "free entry contest",
    "contact now", "cheap rates available", "investment with high returns",
    "grow your business now", "instant delivery", "boost your reach",
    "join our VIP group", "share this with friends", "order now to save",
    "grab your offer today", "top trending product", "official giveaway",
    "click the link in bio", "limited availability", "click here"
]

KEYWORD_SCORE     = 1
TEMPLATE_SCORE    = 2
RULE_THRESHOLD    = 2
ML_PROB_THRESHOLD = 0.80
RANDOM_SEED       = 42
