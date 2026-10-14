"""Metadata and bounded public viewer previews, never full data.parquet files."""
import ast
import concurrent.futures
import hashlib
import json
import random
import subprocess
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "sources"
DEST = RAW / "hf_previews"

def get(url, path, limit=1_500_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result = subprocess.run(["curl", "-fLsS", "--max-time", "45", "--max-filesize", str(limit), url], capture_output=True)
        if result.returncode:
            return None, {"url": url, "error": result.stderr.decode().strip()}
        path.write_bytes(result.stdout)
    content = path.read_bytes()
    return (content.decode() if path.suffix == ".md" else json.loads(content)), {"url": url, "local_path": str(path.relative_to(ROOT)), "bytes": len(content),
                                 "sha256": hashlib.sha256(content).hexdigest()}

def main():
    tree = ast.parse((RAW / "master_all_datasets.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MulTaBenchDatasetID")
    ids = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)}
    jobs = [(name, repo) for name, repo in ids.items() if "_TEXT_" in name]
    def one(job):
        name, repo = job
        repo_id = f"multabench/{repo}"
        metadata, m = get(f"https://huggingface.co/datasets/{repo_id}/raw/main/metadata.json", DEST / repo / "metadata.json")
        info, a = get(f"https://huggingface.co/api/datasets/{repo_id}", DEST / repo / "repo.json")
        size, s = get("https://datasets-server.huggingface.co/size?" + urlencode({"dataset": repo_id}), DEST / repo / "size.json")
        card, c = get(f"https://huggingface.co/datasets/{repo_id}/raw/{info['sha'] if info else 'main'}/README.md", DEST / repo / "README.md")
        records = [m, a, s, c]
        if size and size.get("size", {}).get("splits"):
            split = size["size"]["splits"][0]
            rng = random.Random(20261014 + int(hashlib.sha256(repo.encode()).hexdigest()[:8], 16))
            offset = rng.randrange(max(1, split["num_rows"] - 49))
            query = {"dataset":repo_id, "config":split["config"], "split":split["split"], "offset":offset, "length":50}
            rows, r = get("https://datasets-server.huggingface.co/rows?" + urlencode(query), DEST / repo / "rows.json")
            r.update({"offset":offset,"requested_rows":50,"sampling":"seeded contiguous block; not representative random rows"})
            records.append(r)
        return {"released_id":name,"hf_repo":repo_id,"files":records}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, jobs))
    (DEST / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"Checked {len(results)} released text datasets. Total cached JSON: {sum(f.get('bytes',0) for d in results for f in d['files']):,} bytes.")
    print(f"Failures: {sum('error' in f for d in results for f in d['files'])}. At most 50 preview rows per dataset; no full datasets downloaded.")

if __name__ == "__main__":
    main()
