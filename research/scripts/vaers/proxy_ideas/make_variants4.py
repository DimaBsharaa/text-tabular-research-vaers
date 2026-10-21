"""Round 4 (development only): add manufacturer and administering facility to the matching strata."""
import sys
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from make_variants import OUT, age_band, matched, save  # noqa: E402
full3 = pd.read_csv(OUT / "full3.csv", keep_default_na=False, dtype=str)
base = full3.vax_type + "|" + full3.report_year + "|" + full3.age_years.map(age_band) + "|" + full3.sex + "|" + full3.vax_manu
save(matched(full3, base + "|" + full3.admin_by, name="A8_manu_admin"), "A8_manu_admin")
