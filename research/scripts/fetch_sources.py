"""Fetch a small, pinned subset of official sources; never download datasets/models."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "sources"
SHA = "3bb95079a6146ed35d45187e98bcad5b063273a1"

def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        content = subprocess.run(["curl", "-fLsS", "--max-time", "45", "--max-filesize", "3000000", url], check=True, capture_output=True).stdout
        if len(content) > 3_000_000:
            raise ValueError(f"Small-file limit exceeded: {url}")
        destination.write_bytes(content)
    content = destination.read_bytes()
    return {"url": url, "local_path": str(destination.relative_to(ROOT)),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

def main():
    tree_path = RAW / "paper_tree.json"
    if not tree_path.exists():
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_bytes(Path("/tmp/multabench-paper-tree.json").read_bytes())
    tree = json.loads(tree_path.read_text())
    paths = ["README.md", "multabench/datasets/all_datasets.py",
             "multabench/datasets/curation.py", "multabench/datasets/curation_mapping.py",
             "multabench/datasets/curation_objects.py", "multabench/benchmark/utils/curation.py",
             "multabench/scripts/do_dataset_summary.py",
             "multabench/leaderboard/analysis/text_full_verification.py",
             "multabench/leaderboard/analysis/pool_names.py",
             "multabench/leaderboard/analysis/pass_matrix.py",
             "multabench/leaderboard/analysis/committee_pool.py",
             "multabench/leaderboard/results/analysis_curation_sensitivity/pool_scores_long.csv",
             "multabench/leaderboard/results/analysis_curation_sensitivity/pass_matrix.csv",
             "multabench/leaderboard/results/tabstar_corpus/text_50_datasets.csv",
             "multabench/leaderboard/results/tabstar_corpus/texttabench_datasets.csv"]
    paths += [x["path"] for x in tree["tree"] if x["type"] == "blob" and (
        x["path"].startswith("multabench/leaderboard/results/text/") or
        x["path"].startswith("multabench/leaderboard/results/text_source/") or
        x["path"].startswith("multabench/benchmark/datasets/") and "_TEXT_" in x["path"] or
        x["path"] in ["multabench/leaderboard/results/datasets_summary.csv",
                     "multabench/leaderboard/results/datasets_summary_extra.csv",
                     "multabench/leaderboard/results/analysis_curation_sensitivity/text_full_uploaded_joint_signal.csv"])]
    jobs = [(f"https://raw.githubusercontent.com/alanarazi7/MulTaBench/{SHA}/{p}", RAW / "paper_version" / p) for p in paths]
    jobs += [("https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/multabench/datasets/all_datasets.py", RAW / "master_all_datasets.py"),
             ("https://arxiv.org/html/2605.10616v1", RAW / "paper_v1.html")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        manifest = list(pool.map(lambda job: fetch(*job), jobs))
    (RAW / "manifest.json").write_text(json.dumps({"paper_version_commit": SHA, "files": manifest}, indent=2))
    print(f"Saved {len(manifest)} specific files; {sum(x['bytes'] for x in manifest):,} bytes. No dataset rows or models downloaded.")

if __name__ == "__main__":
    main()
