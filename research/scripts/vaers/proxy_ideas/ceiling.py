import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from eval_proxy import split, struct, text_feats, gbm, STRUCT_NUM, STRUCT_CAT
from sklearn.metrics import roc_auc_score
for p in sys.argv[1:]:
    df = pd.read_csv(p, keep_default_na=False, dtype=str)
    tc = [c for c in df.columns if c not in STRUCT_NUM + STRUCT_CAT + ["serious", "report_id"]]
    y = df.serious.astype(int); xs = struct(df); out = []
    for f in (0, 1, 2):
        tr, te = split(y, f); ytr, yte = y.loc[tr].values, y.loc[te].values
        sc_tr, sc_te = [], []
        for c in tc:
            _, _, a, b = text_feats(df, tr, te, c, ytr); sc_tr.append(a); sc_te.append(b)
        Str, Ste = pd.concat(sc_tr, axis=1), pd.concat(sc_te, axis=1)
        out.append({"S": gbm(xs.loc[tr], ytr, xs.loc[te], yte),
                    "text_score_only": roc_auc_score(yte, Ste.sum(axis=1)),
                    "S_plus_text_score": gbm(pd.concat([xs.loc[tr], Str], axis=1), ytr, pd.concat([xs.loc[te], Ste], axis=1), yte)})
    m = pd.DataFrame(out).mean()
    print(p.split("/")[-1], m.round(4).to_dict(), "max conceivable gain over table alone:", round(m.S_plus_text_score - m.S, 4))
