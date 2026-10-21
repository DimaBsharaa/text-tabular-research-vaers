"""Round 3 (development only): finer matching strata and a 1:2 ratio, built from full3 like v11a (A2_plus_sex)."""
import sys
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from make_variants import OUT, age_band, matched, save  # noqa: E402

full3 = pd.read_csv(OUT / "full3.csv", keep_default_na=False, dtype=str)
base = full3.vax_type + "|" + full3.report_year + "|" + full3.age_years.map(age_band) + "|" + full3.sex
save(matched(full3, base + "|" + full3.vax_dose_series, name="A6_plus_dose"), "A6_plus_dose")
save(matched(full3, base + "|" + full3.admin_by, name="A7_plus_admin"), "A7_plus_admin")
save(matched(full3, base + "|" + full3.vax_manu, name="A5_plus_manu"), "A5_plus_manu")
save(matched(full3, base, ratio=2, name="A2_ratio2"), "A2_ratio2")
