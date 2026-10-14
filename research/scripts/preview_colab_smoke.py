"""Execute only the bounded CSV preview cells; never install or run models."""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/text-tabular-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import IPython.display

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "research/notebooks/02_tennis_official_benchmark_colab.ipynb"


def main():
    nb = json.loads(PATH.read_text())
    namespace = {"CSV": ROOT / "research/data/tennis_smoke/tennis_smoke.csv",
                 "SCHEMA": ROOT / "research/data/tennis_smoke/schema.json", "json": json, "hashlib": hashlib}
    original_display, original_show = IPython.display.display, plt.show
    count = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code" or "local-preview" not in cell["metadata"].get("tags", []):
                continue
            outputs, stream = [], io.StringIO()

            def flush():
                if stream.getvalue():
                    outputs.append({"output_type": "stream", "name": "stdout", "text": stream.getvalue()})
                stream.seek(0); stream.truncate(0)

            def display(*objects, **kwargs):
                flush()
                for obj in objects:
                    data = {"text/plain": str(obj)}
                    if hasattr(obj, "_repr_html_"):
                        data["text/html"] = obj._repr_html_()
                    outputs.append({"output_type": "display_data", "metadata": {}, "data": data})

            def show(*args, **kwargs):
                flush()
                for num in plt.get_fignums():
                    buffer = io.BytesIO()
                    plt.figure(num).savefig(buffer, format="png", dpi=100, bbox_inches="tight")
                    outputs.append({"output_type": "display_data", "metadata": {}, "data": {
                        "image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": "Executed CSV preview figure"}})
                plt.close("all")

            IPython.display.display, plt.show = display, show
            namespace["display"] = display
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell["source"], str(PATH), "exec"), namespace)
            flush()
            count += 1
            cell["execution_count"], cell["outputs"] = count, outputs
    finally:
        IPython.display.display, plt.show = original_display, original_show
    nb["metadata"]["research_execution"].update(csv_preview_cells_executed=count, network_access=False,
         model_execution="not executed; all Colab runtime cells remain unexecuted")
    PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Executed {count} CSV-preview cells only. No installation, checkout, model import or training.")


if __name__ == "__main__":
    main()
