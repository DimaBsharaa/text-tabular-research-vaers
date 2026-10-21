"""Small, auditable helpers for the tennis prototype investigation."""
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("TENNIS_RESEARCH_CACHE", ROOT.parent / "tennis-research-cache"))
OUT = ROOT / "tennis/data/prototypes"
PRIVATE = CACHE / "derived"
SNAPSHOT = "dc55cebd77fffc5f418540c47f9febda500fc342"
SNAPSHOT_TIME = "2015-06-11T16:38:31Z"
SEED = "tennis-prototypes-v1"


def digest(value):
    if not isinstance(value, bytes):
        value = str(value).encode()
    return hashlib.sha256(value).hexdigest()


def normalized(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub("[^a-z0-9]", "", value)


def aliases(full_name):
    words = re.findall(r"[A-Za-z]+", unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode())
    candidates = set()
    for split in range(1, len(words)):
        surname = "".join(words[split:])
        initials = "".join(w[0] for w in words[:split])
        candidates.add(normalized(surname + initials))
        candidates.add(normalized(surname + words[0][0]))
    return candidates


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def text_is_detected(values):
    # Exact cardinality rule in pinned MulTaBench semantic detector, for nonempty
    # nonnumeric strings. This audit does not override or patch that detector.
    values = [str(x) for x in values if isinstance(x, str) and x.strip()]
    return bool(values) and (len(set(values)) >= 100 or len(set(values)) / len(values) >= .8)
