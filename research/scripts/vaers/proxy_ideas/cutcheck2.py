import sys, re, json, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from eval_proxy import split, struct, text_feats, gbm
v10 = pd.read_csv(__import__("os").environ.get("VAERS_WORK", "vaers_work") + "/variants/v10.csv", keep_default_na=False, dtype=str); y = v10.serious.astype(int); ex = v10.narrative_onset
O = pd.DataFrame({"n_words": ex.str.split().str.len(), "n_chars": ex.str.len(),
                  "n_sentences": ex.map(lambda t: len([s for s in re.split(r"(?<=[.!?;])\s+", t) if s.strip()])),
                  "ends_without_punct": (~ex.str.strip().str.contains(r"[.!?]$")).astype(int),
                  "bg_none": (v10.patient_background == "none reported").astype(int),
                  "bg_len": v10.patient_background.str.len()})
xs = struct(v10); rows = []
for fold in (0, 1, 2):
    tr, te = split(y, fold); ytr, yte = y.loc[tr].values, y.loc[te].values
    fz = [text_feats(v10, tr, te, c, ytr)[:2] for c in ["patient_background", "narrative_onset"]]
    FZtr = pd.concat([a for a, _ in fz], axis=1); FZte = pd.concat([b for _, b in fz], axis=1)
    E = O[["n_words", "n_chars", "n_sentences", "ends_without_punct"]]
    rows.append({"excerpt_shape_only": gbm(E.loc[tr], ytr, E.loc[te], yte),
                 "table": gbm(xs.loc[tr], ytr, xs.loc[te], yte),
                 "table+shape": gbm(pd.concat([xs, E], axis=1).loc[tr], ytr, pd.concat([xs, E], axis=1).loc[te], yte),
                 "table+text": gbm(pd.concat([xs.loc[tr], FZtr], axis=1), ytr, pd.concat([xs.loc[te], FZte], axis=1), yte),
                 "table+text+shape": gbm(pd.concat([xs.loc[tr], FZtr, E.loc[tr]], axis=1), ytr, pd.concat([xs.loc[te], FZte, E.loc[te]], axis=1), yte)})
d = pd.DataFrame(rows); print(d.round(4).to_string()); print(d.mean().round(4).to_dict())
