"""Offline analysis of cached official artifacts. Does not import benchmark code or train models."""
from pathlib import Path
from html.parser import HTMLParser
import ast
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources"
PAPER = SOURCES / "paper_version"
RESULTS = PAPER / "multabench" / "leaderboard" / "results"
MODELS = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]
STATES = ["no_text", "text_only", "all", "ft"]
SHA = "3bb95079a6146ed35d45187e98bcad5b063273a1"

def literal_assignments(path):
    """Read static dictionaries/enum entries without executing downloaded Python."""
    result = {}
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        elif isinstance(node, ast.ClassDef):
            result[node.name] = {n.targets[0].id: ast.literal_eval(n.value)
                                 for n in node.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
    return result

class PaperTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag == "annotation": self.skip += 1
        if tag == "table": self.table = []
        if tag == "tr" and self.table is not None: self.row = []
        if tag in {"td", "th"} and self.row is not None: self.cell = []
    def handle_data(self, data):
        if self.cell is not None and not self.skip: self.cell.append(data)
    def handle_endtag(self, tag):
        if tag == "annotation": self.skip -= 1
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        if tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None

def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

ALIASES = {
    "Kickstarter Funding":"Kickstarter", "News Channel Category":"News Channel",
    "Consumer Complaint":"Consumer Complaints", "Fake Job Posting":"Fake Job Postings",
    "OSHA Accident Injury":"OSHA Injury", "Google Q&A Type":"Google QA",
    "American Eagle Prices":"American Eagle", "JC Penney Products":"JC Penney",
    "Wikiliq Alcohol":"WikiLiq Spirits", "Chocolate Bar Ratings":"Chocolate Bars",
    "California House Prices":"California Prices", "SF Permit Applications":"SF Permits",
    "Melbourne Airbnb":"Airbnb Melbourne", "Bike Price Bikewale":"BikeWale Bikes",
    "Car Price Cardekho":"CarDekho Cars", "Polish Wine Prices":"Wine Poland",
    "ML/DS Job Salaries":"AI Job Salaries", "Books Goodreads":"Goodreads Books",
    "US Museum Revenues":"US Museums", "Used Cars Saudi Arabia":"Used Cars Saudi",
    "Laptop Indian Prices":"Indian Laptops", "Coffee Review":"Coffee Reviews",
    "Company Employee Size":"Company Size", "Anime Planet Rating":"Anime Planet",
    "FilmTV Movie Rating":"FilmTV Ratings", "Movies Dataset Revenue":"Box Office",
    "NBA Draft VORP":"NBA Draft Value", "Mercedes Italy Cars":"Used Cars Mercedes",
    "Wine Reviews":"Wine Review", "Company Employees":"Company Size",
    "Ramen Ratings":"Ramen Ratings", "Bike Bikewale":"BikeWale Bikes",
}

def historical_results():
    raw = pd.read_csv(RESULTS / "analysis_curation_sensitivity" / "pool_scores_long.csv")
    raw = raw[raw.model.isin(MODELS)].copy()
    assert raw.dataset.nunique() == 56
    assert not raw.duplicated(["dataset", "model", "state", "fold"]).any()
    assert raw.test_score.notna().all()
    known = ("REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO", "TabPFNv2", "ft", 4)
    rows = []
    for dataset in sorted(raw.dataset.unique()):
        for model in MODELS:
            sub = raw[(raw.dataset == dataset) & (raw.model == model)]
            if sub.empty:
                assert model.startswith("TabPFN") and dataset in {"MUL_TEXT_FOOD_WINE_REVIEW", "MUL_TEXT_SOCIAL_SPOTIFY_GENRES"}
                rows.append({"dataset_id":dataset,"model_name":model,"evaluation_available":False,"failure_type":"not evaluated"})
                continue
            expected = {(dataset, model, state, fold) for state in STATES for fold in range(5)}
            actual = set(zip(sub.dataset, sub.model, sub.state, sub.fold))
            assert (expected - actual) <= {known}, (dataset, model, expected - actual)
            assert not (actual - expected)
            means = sub.groupby("state").test_score.mean()
            rounded = means.round(3)
            joint = round(rounded["all"] - max(rounded["no_text"], rounded["text_only"]), 6)
            tar = round(rounded["ft"] - rounded["all"], 6)
            jp, tp = joint >= .001, tar >= .001
            row = {"dataset_id":dataset,"model_name":model,"evaluation_available":True,
                   "structured_score":rounded["no_text"],"unstructured_score":rounded["text_only"],
                   "joint_frozen_score":rounded["all"],"joint_tar_score":rounded["ft"],
                   "joint_signal_gain":joint,"tar_gain":tar,"joint_signal_pass":jp,"tar_pass":tp,
                   "passes_both_criteria":jp and tp,"paper_strict_margin_pass":joint > .001 and tar > .001,
                   "unrounded_joint_gain":means["all"]-max(means["no_text"],means["text_only"]),
                   "unrounded_tar_gain":means["ft"]-means["all"],
                   "failure_type":"pass" if jp and tp else "both" if not jp and not tp else "joint signal" if not jp else "TAR gain",
                   "n_folds_structured":int((sub.state == "no_text").sum()),
                   "n_folds_unstructured":int((sub.state == "text_only").sum()),
                   "n_folds_frozen":int((sub.state == "all").sum()),
                   "n_folds_tar":int((sub.state == "ft").sum()),
                   "fold_gap_note":"One missing TAR fold (4), explicitly documented by official code" if known in expected-actual else ""}
            paired = sub.pivot(index="fold",columns="state",values="test_score")
            paired_tar = (paired["ft"]-paired["all"]).dropna()
            best = "no_text" if means["no_text"] >= means["text_only"] else "text_only"
            paired_joint = (paired["all"]-paired[best]).dropna()
            row.update({"strongest_unimodal_condition":best,"paired_joint_gain_sd":paired_joint.std(ddof=1),
                        "paired_tar_gain_sd":paired_tar.std(ddof=1),"paired_joint_positive_fold_count":int((paired_joint>0).sum()),
                        "paired_tar_positive_fold_count":int((paired_tar>0).sum()),"paired_tar_n_folds":len(paired_tar)})
            for state in STATES:
                row[f"{state}_fold_sd"] = sub.loc[sub.state == state,"test_score"].std(ddof=1)
            rows.append(row)
    results = pd.DataFrame(rows)
    for column in ["evaluation_available","joint_signal_pass","tar_pass","passes_both_criteria","paper_strict_margin_pass"]:
        results[column] = results[column].astype("boolean")
    official = pd.read_csv(RESULTS / "analysis_curation_sensitivity" / "pass_matrix.csv").set_index("dataset")
    for r in results.itertuples():
        flag = official.loc[r.dataset_id, r.model_name]
        assert (pd.isna(flag) and not r.evaluation_available) or (not pd.isna(flag) and bool(flag) == r.passes_both_criteria)
    return raw, results

def build_inventory(results):
    names = literal_assignments(PAPER / "multabench/leaderboard/analysis/pool_names.py")
    friendly = literal_assignments(PAPER / "multabench/scripts/do_dataset_summary.py")
    released_names = {**friendly["_FRIENDLY_NAMES"], **friendly["_FRIENDLY_NAMES_EXTRA_TEXT"]}
    pool_map = names["POOL_TO_RELEASED"]
    displays = {**{k:released_names[v] for k,v in pool_map.items()}, **names["_UNRELEASED_NAMES"]}
    resolver = {norm(v):k for k,v in displays.items()}
    def resolve(value): return resolver[norm(ALIASES.get(value,value))]
    parser = PaperTables()
    parser.feed((SOURCES / "paper_v1.html").read_text())
    table = next(t for t in parser.tables if t and t[0] == ["Dataset","LGBM","Cat","TabM","PFNv2","PFN-2.5","Pass?"])
    published = {}
    for row in table[1:]:
        if len(row) == 7:
            did = resolve(row[0])
            published[did] = {"paper_name":row[0],"passes":int(row[-1]),"flags":[True if "✓" in v else False if "×" in v else None for v in row[1:6]]}
    assert len(published) == 56 and sum(v["passes"] >= 3 for v in published.values()) == 23
    sources = {k:set() for k in displays}
    shared = next(t for t in parser.tables if t and t[0] == ["Dataset","AutoML Multimodal","Grinsztajn et al","CARTE","TextTabBench"])
    for row in shared[1:]:
        if len(row) == 5:
            for col, value in zip(shared[0][1:], row[1:]):
                if "✓" in value:sources[resolve(row[0])].add(col)
    extra = pd.read_csv(RESULTS / "tabstar_corpus/texttabench_datasets.csv")
    for dataset in extra.dataset.unique(): sources[dataset].add("TextTabBench")
    catalog = literal_assignments(PAPER / "multabench/datasets/all_datasets.py")
    urls = {}
    for cls in ["OpenMLDatasetID","KaggleDatasetID","UrlDatasetID"]:
        for key,value in catalog[cls].items():
            url = f"https://www.openml.org/search?type=data&id={value}" if cls == "OpenMLDatasetID" else f"https://www.kaggle.com/datasets/{value}" if cls == "KaggleDatasetID" else value
            urls.setdefault(key,[]).append(url)
    hf_catalog = literal_assignments(SOURCES / "master_all_datasets.py")["MulTaBenchDatasetID"]
    core = set(friendly["_FRIENDLY_NAMES"])
    rows = []
    for did in sorted(displays):
        release = pool_map.get(did)
        hf_slug = hf_catalog.get(release)
        sample_dir = SOURCES / "hf_previews" / str(hf_slug)
        repo_path = sample_dir / "repo.json"
        hf_info = json.loads(repo_path.read_text()) if repo_path.exists() else {}
        pub = published[did]
        sub = results[results.dataset_id == did]
        reconstructed = int(sub.passes_both_criteria.fillna(False).sum())
        row = {"dataset_id":did,"dataset_name":displays[did],"paper_dataset_name":pub["paper_name"],
               "original_benchmark_source":"; ".join(sorted(sources[did])) or None,
               "benchmark_source_coverage":"verified memberships only; absence is unknown",
               "original_dataset_url":"; ".join(urls.get(did,[])) or None,
               "original_source_access_checked":False,
               "task_type":"regression" if did.startswith("REG_") else "binary" if did.startswith("BIN_") else "multiclass",
               "released_dataset_id":release,"released_collection":"core" if release in core else "extended" if release else "unreleased",
               "selected_for_core_benchmark":release in core,
               "passed_curation":pub["passes"]>=3,"number_of_passing_learners":pub["passes"],
               "reproduced_number_of_passing_learners":reconstructed,
               "reproduced_passed_curation":reconstructed>=3,
               "released_dataset_available":bool(hf_info and not hf_info.get("gated") and any(x['rfilename']=='data.parquet' for x in hf_info.get('siblings',[]))),
               "released_dataset_location":f"https://huggingface.co/datasets/multabench/{hf_slug}" if hf_slug else None,
               "hf_revision":hf_info.get("sha"),
               "domain":did.split("_TEXT_")[1].split("_")[0].lower(),
               "metadata_source":f"https://github.com/alanarazi7/MulTaBench/tree/{SHA}; https://arxiv.org/html/2605.10616v1#A3.SS4",
               "notes":"Published outcomes retained separately from reproduction. " + ("Current release changes the source target task; do not join release profiles as original-task properties. " if release and did[:3] != release[:3] else "")}
        # Domains are broad source-corpus tags, not inferred explanations.
        row["domain_source"] = "official original dataset identifier"
        for model, label, flag in zip(MODELS,["lightgbm_pass","catboost_pass","tabm_pass","tabpfn2_pass","tabpfn25_pass"],pub["flags"]):
            row[label] = flag
            row[f"reproduced_{label}"] = sub.loc[sub.model_name == model,"passes_both_criteria"].iloc[0]
        if hf_slug:
            meta = json.loads((sample_dir / "metadata.json").read_text())
            row["released_target_column"] = meta["target"]
            row["released_task_type"] = meta["task_type"]
            # A matching task prefix does not verify the original source target's name.
            row["target_column"] = None
            row["notes"] += "Original source target column name not verified; released target stored separately. "
        rows.append(row)
    inv = pd.DataFrame(rows)
    assert len(inv) == 56 and inv.selected_for_core_benchmark.sum() == 20
    assert inv.reproduced_passed_curation.sum() == 23
    return inv, published

def dataset_profiles(inv):
    summaries = []
    for filename, collection in [("datasets_summary.csv","core"),("datasets_summary_extra.csv","extended")]:
        d = pd.read_csv(RESULTS / filename)
        d = d[d.Modality == "Text"].copy()
        d["collection"] = collection
        summaries.append(d)
    summary = pd.concat(summaries).set_index("Dataset")
    profiles, textrows, colrows, risks, targetrows = [], [], [], [], []
    for record in inv.to_dict("records"):
        did, name, collection = record["dataset_id"], record["dataset_name"], record["released_collection"]
        base = {"dataset_id":did,"dataset_name":name,"released_collection":collection,"passed_curation":record["passed_curation"],
                "profile_scope":"unavailable" if collection == "unreleased" else "historical released summary + current HF preview",
                "historical_task_matches_release":did[:3] == str(record["released_dataset_id"])[:3]}
        if collection == "unreleased":
            profiles.append(base)
            continue
        ss = summary.loc[name]
        base.update({"historical_released_n_rows":ss["N"],"historical_released_structured_features":ss["Struct."],
                     "historical_released_text_features":ss["Text cols"],"historical_released_n_classes":ss["Classes"],
                     "historical_released_task_type":ss["Task"]})
        slug = record["released_dataset_location"].rsplit("/",1)[1]
        folder = SOURCES / "hf_previews" / slug
        meta, repo, size, rows = [json.loads((folder / f).read_text()) for f in ["metadata.json","repo.json","size.json","rows.json"]]
        df = pd.DataFrame([r["row"] for r in rows["rows"]])
        target = meta["target"]
        # HF's viewer flattens parquet categorical dtype to string. Dataset card retains types.
        card = (folder / "README.md").read_text()
        dtype = {column.strip():typ.strip() for column,typ in re.findall(r"^\| ([^|]+) \| ([^|]+) \|$",card,re.M)
                 if column.strip() != "Column"}
        assert len({c.strip() for c in df}) == len(df.columns)
        dtype = {c:dtype[c.strip()] for c in df.columns if c.strip() in dtype}
        assert set(df.columns) == set(dtype), (name, set(df.columns)-set(dtype))
        text = [c for c,t in dtype.items() if t == "string" and c != target]
        numeric = [c for c,t in dtype.items() if t in {"int","float","integer","number","double"} and c != target]
        cat = [c for c,t in dtype.items() if t == "categorical" and c != target]
        boolean = [c for c,t in dtype.items() if t == "bool" and c != target]
        dates = [c for c,t in dtype.items() if t == "datetime" and c != target]
        truncated = sum(len(r.get("truncated_cells",[])) for r in rows["rows"])
        base.update({"hf_n_rows":size["size"]["dataset"]["num_rows"],"dataset_bytes":size["size"]["dataset"]["num_bytes_original_files"],
                     "hf_numeric_features":len(numeric),"hf_categorical_features":len(cat),"hf_boolean_features":len(boolean),"hf_text_features":len(text),"hf_date_features":len(dates),
                     "hf_other_features":len([c for c in df if c!=target])-len(numeric)-len(cat)-len(boolean)-len(text)-len(dates),
                     "sample_n_rows":len(df),"sample_offset":rows["rows"][0]["row_idx"],"sample_truncated_cells":truncated,
                     "sample_feature_missing_rate":df.drop(columns=target).isna().to_numpy().mean(),
                     "sample_duplicate_predictor_row_rate":df.drop(columns=target).duplicated().mean(),
                     "sample_duplicate_full_row_rate":df.duplicated().mean(),
                     "released_target_column":target,"released_task_type":meta["task_type"]})
        allchars, allwords, empty, repeated, total = [], [], 0, 0, 0
        for c in df.columns:
            series = df[c]
            colrows.append({"dataset_id":did,"dataset_name":name,"column_name":c,"released_dtype":dtype[c],"is_target":c == target,
                            "sample_n_rows":len(df),"sample_missing_rate":series.isna().mean(),"sample_n_unique":series.dropna().astype(str).nunique()})
        for c in text:
            values = df[c].dropna().astype(str).str.strip()
            nonempty = values[values.ne("")]
            chars = nonempty.str.len()
            words = nonempty.str.split().str.len()
            n_empty = len(df)-len(nonempty)
            textrows.append({"dataset_id":did,"dataset_name":name,"column_name":c,"sample_n_rows":len(df),
                             "sample_nonempty_n":len(nonempty),"sample_mean_chars":chars.mean(),"sample_median_chars":chars.median(),
                             "sample_p90_chars":chars.quantile(.9),"sample_mean_words":words.mean(),"sample_median_words":words.median(),
                             "sample_empty_rate":n_empty/len(df),"sample_repeated_nonempty_rate":nonempty.duplicated().mean(),
                             "sample_unique_nonempty_ratio":nonempty.nunique()/len(nonempty) if len(nonempty) else np.nan,
                             "sample_short_string_rate":(words <= 3).mean(),"sample_long_text_rate":(words >= 20).mean(),
                             "measurement":"Python len Unicode chars; whitespace split words; stripped; nonempty cells for length",
                             "sample_scope":"seeded contiguous 50-row public preview, current release"})
            allchars.extend(chars.tolist()); allwords.extend(words.tolist())
            empty += n_empty; repeated += nonempty.duplicated().sum(); total += len(df)
        base.update({"sample_mean_text_chars":np.mean(allchars) if allchars else np.nan,
                     "sample_median_text_chars":np.median(allchars) if allchars else np.nan,
                     "sample_mean_text_words":np.mean(allwords) if allwords else np.nan,
                     "sample_median_text_words":np.median(allwords) if allwords else np.nan,
                     "sample_empty_text_rate":empty/total if total else np.nan,
                     "sample_repeated_text_rate":repeated/len(allchars) if allchars else np.nan,
                     "sample_longest_column_mean_chars":max((r['sample_mean_chars'] for r in textrows if r['dataset_id']==did),default=np.nan)})
        y = df[target]
        base["sample_target_missing_rate"] = y.isna().mean()
        base["sample_target_n_unique"] = y.nunique()
        if meta["task_type"] == "reg":
            yn = pd.to_numeric(y,errors="coerce")
            base["sample_target_variance"] = yn.var(ddof=1)
            base["sample_target_mean"] = yn.mean()
            base["sample_target_median"] = yn.median()
            for q in [0,.25,.5,.75,1]:targetrows.append({"dataset_id":did,"dataset_name":name,"kind":"regression quantile","value":q,"statistic":yn.quantile(q),"sample_n_rows":len(df)})
        else:
            counts = y.value_counts(dropna=False)
            base["sample_majority_class_fraction"] = counts.max()/len(df)
            for label, count in counts.items():targetrows.append({"dataset_id":did,"dataset_name":name,"kind":"class count","value":str(label),"statistic":count,"sample_n_rows":len(df)})
        for c in df.columns:
            if c == target:continue
            reasons = []
            if re.search(r"(?i)(salary|wage|price|revenue|rating|score|recommend|income|state|fraud|toxicity|result)", c):
                reasons.append("name overlaps outcome-like concepts; review timing and definition")
            if norm(c) == norm(target):reasons.append("normalized name matches target")
            if c in numeric and meta["task_type"] == "reg":
                pair = pd.DataFrame({"x":pd.to_numeric(df[c],errors="coerce"),"y":pd.to_numeric(y,errors="coerce")}).dropna()
                if len(pair) >= 20 and pair.x.nunique()>1 and pair.y.nunique()>1:
                    corr = pair.x.corr(pair.y)
                    if abs(corr) >= .95:reasons.append(f"preview Pearson |r| >= 0.95 (r={corr:.3f}); not proof of leakage")
            if reasons:risks.append({"dataset_id":did,"dataset_name":name,"column_name":c,"audit_reason":"; ".join(reasons),"evidence_scope":"current public preview/card; manual audit required"})
        profiles.append(base)
    return pd.DataFrame(profiles),pd.DataFrame(textrows),pd.DataFrame(colrows),pd.DataFrame(risks),pd.DataFrame(targetrows)

def main():
    (ROOT / "reports").mkdir(exist_ok=True)
    raw, results = historical_results()
    inv, published = build_inventory(results)
    results = results.merge(inv[["dataset_id","dataset_name","passed_curation","task_type","released_collection"]],on="dataset_id",validate="many_to_one")
    profiles, texts, columns, risks, targets = dataset_profiles(inv)
    frames = {"dataset_inventory":inv,"historical_fold_scores":raw,"learner_curation_results":results,
              "dataset_profiles":profiles,"text_column_profiles":texts,"column_profiles":columns,
              "leakage_audit_candidates":risks,"sample_target_distribution":targets}
    for name,frame in frames.items():frame.to_csv(ROOT / "data" / f"{name}.csv",index=False)
    disagreement = results[["dataset_id","dataset_name","model_name","passes_both_criteria"]].copy()
    pub_flag = {(did,model):flag for did,v in published.items() for model,flag in zip(MODELS,v['flags'])}
    disagreement["paper_published_pass"] = [pub_flag[(did,model)] for did,model in zip(disagreement.dataset_id,disagreement.model_name)]
    mask = disagreement.paper_published_pass.notna() & disagreement.passes_both_criteria.notna()
    disagreement = disagreement.loc[mask & disagreement.paper_published_pass.ne(disagreement.passes_both_criteria)]
    disagreement.to_csv(ROOT / "data" / "published_reproduction_disagreements.csv",index=False)
    summary = {"candidates":len(inv),"published_pass":int(inv.passed_curation.sum()),"reproduced_pass":int(inv.reproduced_passed_curation.sum()),
               "published_reproduction_cell_disagreements":len(disagreement),"historical_fold_rows":len(raw),
               "evaluated_model_dataset_pairs":int(results.evaluation_available.sum()),"released_datasets":int(inv.released_dataset_available.sum()),
               "preview_rows":int(profiles.sample_n_rows.sum()),"preview_text_columns":len(texts),"preview_truncated_cells":int(profiles.sample_truncated_cells.sum()),
               "profile_task_changed":inv.loc[inv.released_dataset_id.notna() & inv.dataset_id.str[:3].ne(inv.released_dataset_id.str[:3]), 'dataset_name'].tolist()}
    (ROOT / "reports" / "analysis_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":main()
