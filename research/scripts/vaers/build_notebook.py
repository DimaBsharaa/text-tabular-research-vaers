"""Generate research/notebooks/06_vaers_candidate_build.ipynb (CPU Colab; no GPU needed)."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip() + "\n"})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(s).strip() + "\n",
                  "execution_count": None, "outputs": []})


md("""
    # 06 — VAERS severity-triage candidate: build + cheap screen

    **Task.** From what is known when a vaccine adverse-event report is filed — coded symptoms, the patient's
    background text (history, current illness, medications, allergies) and structured facts (age, sex, vaccine,
    onset interval) — predict whether the event was **serious** (death, life-threatening, hospitalisation,
    prolonged hospitalisation, disability or birth defect). It is the public analogue of a caseworker reading a
    file and grading need.

    **Why it fits the winner profile (notebook 01):** binary classification, tens of thousands of rows,
    medical text whose meaning a generic frozen encoder only partly captures (room for TAR), and structured
    signal (age, vaccine, onset) that the text does not repeat.

    **Before running:** download yearly ZIPs from <https://vaers.hhs.gov/data/datasets.html> (each download asks
    for a verification code in the browser). Suggested: **2019, 2023, 2024** (mixed vaccines, avoids 2021's
    COVID-only flood). Upload them to `MyDrive/text-tabular-research-2/vaers_raw/`. Runtime: CPU is enough.
""")
code("""
    RAW_DIR = "/content/drive/MyDrive/text-tabular-research-2/vaers_raw"
    OUT_BASE = "/content/drive/MyDrive/text-tabular-research-2/candidates"
    ROWS = 60000      # Final rows per variant (the benchmark samples ≤10k train + ≤2k test per fold from these).
    VARIANTS = ["v1", "v6", "v7"]  # v6 = v1 + the OPENING of the reporter's narrative (cut before any care/treatment/test/outcome).
    # v4/v5 (full masked narrative) were rejected: hospital-chart text (remdesivir, IV, O2, ICD codes) leaks seriousness.

    from google.colab import drive
    drive.mount("/content/drive")
    import subprocess, sys
    from pathlib import Path
    PROJECT = Path("/content/text-tabular-research-2")
    if not PROJECT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/DimaBsharaa/text-tabular-research-2.git", str(PROJECT)], check=True)
    else:
        subprocess.run(["git", "-C", str(PROJECT), "pull", "--ff-only"], check=True)
    sys.path.insert(0, str(PROJECT))
    print("commit", subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip())
    zips = sorted(Path(RAW_DIR).glob("*VAERSData.zip")) + sorted(Path(RAW_DIR).glob("*VAERS*.csv"))
    assert zips, f"Put the yearly VAERS ZIPs in {RAW_DIR}"
    print([z.name for z in zips])
""")
md("## 1. Offline tests of the builder (synthetic fixture), then build each variant")
code("""
    r = subprocess.run([sys.executable, "-m", "unittest", "research.scripts.vaers.test_build_vaers"], cwd=PROJECT,
                       capture_output=True, text=True)
    print(r.stderr[-800:]); assert r.returncode == 0, "Builder tests failed — do not use the output"

    import json, pandas as pd
    from IPython.display import display
    from research.scripts.vaers import build_vaers as bv
    built = {}
    for v in VARIANTS:
        out, meta = bv.build(zips, v, n_rows=ROWS)
        commit = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip()
        provenance = {"commit": commit, "build_version": bv.BUILD_VERSION,
                      "inputs": {z.name: __import__("hashlib").sha256(z.read_bytes()).hexdigest() for z in zips}}
        # Every build gets its own folder, so earlier builds are never overwritten.
        csv = bv.write(out, meta, Path(OUT_BASE) / f"build{bv.BUILD_VERSION}" / f"vaers_{v}", v, provenance)
        built[v] = csv
        print(v, json.dumps(meta["counts"]), "positive rate", round(out.serious.mean(), 4))
        print("   outcome words still present per text column (leak counter; 0 expected except symptoms):",
              meta["counts"]["outcome_mentions_left"])
    for kind, terms in meta["removed_terms"].items():
        print(f"Removed {kind} terms ({len(terms)}), first 25:", terms[:25])
""")
md("""
    ## 2. Profile vs passing datasets and the cheap CPU screen

    The screen is TF-IDF + logistic regression on one 80/20 split: **not** the official benchmark, only a filter.
    What we want to see: text-only clearly above structured-only (winners: +0.1 to +0.3), and joint above both.
    **Read the top n-grams:** if words like *hospital, admitted, died, ICU* appear, the label leaks through the text
    and that variant must not be used.
""")
code("""
    from research.scripts.vaers import audit_report
    for v, csv in built.items():
        audit_report.report(csv, show=display)
""")
md("""
    ## 3. Next: the official fold-0 screen on A100

    Open notebook 02, set `MODE = "diagnostic"`, `MODELS_OVERRIDE = ["light", "cat"]`, `FOLDS_OVERRIDE = [0, 1, 2]`
    and the two paths printed below (one variant per Colab session), then Run all. Unchanged official code:
    2 learners × 4 conditions × 3 folds, full TAR.
""")
code("""
    for v, csv in built.items():
        print(f'{v}: CSV_PATH = "{csv}"\\n    SCHEMA_PATH = "{csv.parent / "schema.json"}"')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"name": "06_vaers_candidate_build.ipynb"}},
      "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "research/notebooks/06_vaers_candidate_build.ipynb"
path.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", path)
