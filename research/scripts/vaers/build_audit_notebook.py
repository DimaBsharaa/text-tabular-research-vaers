"""Generate research/notebooks/07_vaers_run_audit.ipynb (CPU Colab): independent audit of a finished screen."""
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
    # 07: Audit of a finished screen (CPU only)

    It checks the exact CSV and the run folder that produced a result:
    - mechanics: same test rows for every condition, TAR encoder trained on training rows only
    - near-duplicate events across train/test
    - the text signal inside year and vaccine groups
    - reporter style versus content
    - original versus retained text
    - all-row source reconstruction and independent care-language review

    Start a **separate fresh CPU runtime**; do not change the runtime of a running GPU notebook.
    The dataset and benchmark are unchanged. This writes a progressive diagnostic report and full matched
    examples JSON to Drive. Candidate duplicates are not confirmed clinical events; proxy scores are not
    official E5/TAR scores. PASS applies only to a named check, never to the whole dataset.
""")
code("""
    CSV_PATH = "/content/drive/MyDrive/text-tabular-research-2/candidates/vaers_v7/vaers_v7.csv"
    SCHEMA_PATH = "/content/drive/MyDrive/text-tabular-research-2/candidates/vaers_v7/schema.json"
    RESULTS_DIR = "/content/drive/MyDrive/text-tabular-research-2/benchmark-results/a12d3b1c04dbbdd7"
    RAW_DIR = "/content/drive/MyDrive/text-tabular-research-2/vaers_raw"
    OUT = "/content/drive/MyDrive/text-tabular-research-2/audits/v7_build8_a12d3b1c04dbbdd7.md"

    from google.colab import drive
    drive.mount("/content/drive")
    import base64, os, subprocess, sys
    from pathlib import Path
    REPO_URL = "https://github.com/DimaBsharaa/text-tabular-research-vaers.git"

    def github_env():
        # The repo is private: git authenticates with the GH_TOKEN Colab secret. The token travels in an
        # HTTP header set through environment variables, so it never appears in a printed command,
        # an error message or .git/config.
        try:
            from google.colab import userdata
            token = userdata.get("GH_TOKEN")
        except Exception:
            token = None
        if not token:
            raise RuntimeError("GH_TOKEN missing: add a GitHub token with read access to "
                               "DimaBsharaa/text-tabular-research-vaers in Colab Secrets")
        header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        return {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {header}"}
    PROJECT = Path("/content/text-tabular-research-vaers")
    if not PROJECT.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT)], check=True, env=github_env())
    else:
        subprocess.run(["git", "-C", str(PROJECT), "pull", "--ff-only"], check=True, env=github_env())
    sys.path.insert(0, str(PROJECT))
    print("commit", subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip())
    subprocess.run([sys.executable, "-m", "unittest", "research.scripts.vaers.test_run_audit",
                    "research.scripts.vaers.test_audit_source_rows", "-q"],
                   cwd=PROJECT, check=True)
    from research.scripts.vaers import run_audit as ra
    assert getattr(ra, "AUDIT_VERSION", None) == "update15-corrected-2", "Restart the runtime to load the corrected auditor"
    print("auditor", ra.AUDIT_VERSION, ra.__file__)
    raw = sorted(Path(RAW_DIR).glob("*VAERSData.zip"))
    report = ra.run(Path(CSV_PATH), Path(SCHEMA_PATH), Path(RESULTS_DIR), raw, Path(OUT))
    print(report)
    print("\\nSaved to", OUT)
    print("Full original/retained examples, when raw sources are available:", Path(OUT).with_suffix(".examples.json"))
""")
md("""
    ## All-row source reconstruction and independent review

    The following checks all candidate labels and 17 predictors against the source records.
    Replaying the same extractor proves reproducibility, not that its rules remove every care detail.
    The additional patterns are review triggers: prior surgery and surgical-tape allergies, for example,
    can be legitimate background. Read the original context before deciding whether a match violates the policy.
""")
code("""
    if raw:
        import json
        import pandas as pd
        from research.scripts.vaers import audit_source_rows
        source_audit = audit_source_rows.audit(Path(CSV_PATH), Path(SCHEMA_PATH), raw, Path(RESULTS_DIR))
        source_out = Path(OUT).with_suffix(".source_rows.json")
        source_out.write_text(json.dumps(source_audit, indent=2) + "\\n")
        display(pd.DataFrame(source_audit["source_reconstruction_checks"]))
        display(pd.DataFrame(source_audit["independent_text_review"])[["pattern", "column", "matches", "serious"]])
        print("Raw event-key candidates:", source_audit["raw_event_key_candidates"])
        print("Full review contexts saved locally to", source_out)
        print("Review-pattern matches do not automatically establish leakage.")
    else:
        print("NOT_VERIFIED: all-row source reconstruction requires the exact yearly raw files.")
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"name": "07_vaers_run_audit.ipynb"}},
      "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "research/notebooks/07_vaers_run_audit.ipynb"
path.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", path)
