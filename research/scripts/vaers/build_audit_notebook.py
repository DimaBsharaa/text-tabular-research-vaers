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
    # 07: Independent audit of a finished screen (CPU only)

    It checks the exact CSV and the run folder that produced a result:
    - mechanics: same test rows for every condition, TAR encoder trained on training rows only
    - near-duplicate events across train/test
    - the text signal inside year and vaccine groups
    - reporter style versus content
    - original versus retained text

    It changes nothing; it writes a report to Drive.
""")
code("""
    CSV_PATH = "/content/drive/MyDrive/text-tabular-research-2/candidates/vaers_v7/vaers_v7.csv"
    SCHEMA_PATH = "/content/drive/MyDrive/text-tabular-research-2/candidates/vaers_v7/schema.json"
    RESULTS_DIR = "/content/drive/MyDrive/text-tabular-research-2/benchmark-results/a12d3b1c04dbbdd7"
    RAW_DIR = "/content/drive/MyDrive/text-tabular-research-2/vaers_raw"
    OUT = "/content/drive/MyDrive/text-tabular-research-2/audits/v7_build8_a12d3b1c04dbbdd7.md"

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
    from research.scripts.vaers import run_audit as ra
    raw = sorted(Path(RAW_DIR).glob("*VAERSData.zip"))
    report = ra.run(Path(CSV_PATH), Path(SCHEMA_PATH), Path(RESULTS_DIR), raw, Path(OUT))
    print(report)
    print("\\nSaved to", OUT)
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"name": "07_vaers_run_audit.ipynb"}},
      "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "research/notebooks/07_vaers_run_audit.ipynb"
path.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", path)
