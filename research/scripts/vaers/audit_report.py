"""Everything notebook 06 prints about one built variant. Kept in the repo (not in notebook cells) so that an old
copy of the notebook still prints the current checks after `git pull`."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from research.scripts.vaers import proxy_screen as ps


def report(csv: Path, show=print) -> dict:
    csv = Path(csv)
    schema = json.loads((csv.parent / "schema.json").read_text())
    df = pd.read_csv(csv)
    print(f"===== {csv.parent.name}  (build {schema.get('build_version')}, {len(df)} rows, "
          f"positive rate {df[schema['target']].mean():.4f})")
    show(ps.profile(df, schema))
    res = ps.run(csv, csv.parent / "schema.json")
    show(pd.Series({k: res[k] for k in res if k != "top_positive_ngrams"}))
    print("Top positive n-grams (leakage audit):"); print("\n".join(res["top_positive_ngrams"]))
    print("Column audit, each column ALONE (a high score is a reason to investigate, not proof of leakage):")
    show(ps.column_audit(csv, csv.parent / "schema.json"))
    counts = schema.get("counts", {})
    print("Onset excerpt empty share by label, BEFORE the build-6 filter:",
          counts.get("narrative_onset_empty_share_by_label"),
          "| rows before/after the filter:", counts.get("with_symptom_terms"), "->", counts.get("with_onset_excerpt"))
    if "narrative_onset" in df:
        terms = [t.split("__", 1)[1].rsplit(" (", 1)[0] for t in res["top_positive_ngrams"]
                 if t.startswith("txt_narrative_onset__")][:10]
        print("Where the strongest onset words come from (3 serious examples each):")
        for term in terms:
            hits = df[(df.serious == 1) & df.narrative_onset.str.contains(rf"\b{re.escape(term)}\b", case=False)]
            print(f"  -- '{term}' ({len(hits)} serious rows)")
            for txt in hits.narrative_onset.head(3):
                print(f"     {txt[:200]}")
        print("Random onset excerpts (1 = serious, 0 = not serious):")
        for lab in (1, 0):
            for txt in df[df.serious == lab].narrative_onset.sample(5, random_state=2):
                print(f"  [{lab}] {txt[:220]}")
    for c in ["admin_by", "vax_type", "report_year", "state"]:
        print(f"serious rate by {c}:")
        show(df.groupby(c).serious.agg(["mean", "size"]).sort_values("size", ascending=False).head(8).round(3))
    return res
