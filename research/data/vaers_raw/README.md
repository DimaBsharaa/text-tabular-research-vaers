# Raw VAERS files

The three official VAERS yearly files that the dataset is built from, exactly as downloaded on 15 Oct 2026 from
[vaers.hhs.gov/data/datasets.html](https://vaers.hhs.gov/data/datasets.html). VAERS data are public U.S. government data
(CDC and FDA).

| File | Size | sha256 |
|---|---:|---|
| `2019VAERSData.zip` | 12.3 MB | `1e508ab85e5528e418a7ceca3229bbfc6aede73d0c8657f8f1b857974e89c866` |
| `2023VAERSData.zip` | 26.4 MB | `47c570a2add43ef06196793040b7832f6ee241e007f136a66e30227d6f7355da` |
| `2024VAERSData.zip` | 13.9 MB | `8ba2bc09e853044081ebe9b13104be55fa3d96f39f5c6974a35fba993badf284` |

Each ZIP holds three tables joined by `VAERS_ID`: `VAERSDATA` (one row per report: patient, dates, outcome flags, narrative),
`VAERSVAX` (vaccines given) and `VAERSSYMPTOMS` (coded symptom terms).

They are kept here because the VAERS site can update these files as reports are added or corrected; these copies let anyone
rebuild the dataset byte for byte:

```bash
python -m research.scripts.vaers.build_v11b research/data/vaers_raw/2019VAERSData.zip \
    research/data/vaers_raw/2023VAERSData.zip research/data/vaers_raw/2024VAERSData.zip --out-dir out/
```

The build stops with an error if the output does not match the published sha256. A newer download from the VAERS site
can contain added or corrected reports, so it may produce a slightly different dataset.

VAERS reports are unverified and cannot establish that a vaccine caused an event. See the
[VAERS data use guide](https://vaers.hhs.gov/data/dataguide.html).
