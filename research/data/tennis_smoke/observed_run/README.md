# Observed official LightGBM smoke run

This is metadata/results from the user's completed Colab export `tennis_smoke_50c63223f59697c5_results.zip`, not a newly executed experiment. The original ZIP remains local. Its SHA-256 and the retained archive members are recorded in [metadata.json](metadata.json).

All four conditions completed successfully on the same bounded 488-row fixture: 438 outer-training and 50 held-out rows, LightGBM, fold 0. TAR used the shortened smoke settings, not a full official evaluation. See [summary.csv](summary.csv), [results.csv](results.csv) and the four per-job `result.json` files for exact settings, metrics and execution provenance.

| Condition | Held-out AUROC |
| --- | ---: |
| Structured Only | 0.669872 |
| Text Only | 0.427885 |
| Joint Frozen | 0.487179 |
| Joint TAR | 0.448718 |

Joint Signal is **−0.182692**; TAR Gain is **−0.038462**. This fixture does not demonstrate complementary modalities or qualification. Its generated historical text is not the real-interview candidate.

[reference_proof.json](reference_proof.json) records the pinned official commit, **470 matching tracked-file checks**, and imported module paths. Research code at execution was `afa5a71d6d8a489af844c2b258db75d6cfdadd01`. Keeping these outputs does not modify the benchmark. Missing learner rows in the summary mean not run; blank qualification flags are intentional for a smoke fixture. Raw logs and dependency listings remain in the user's original export.
