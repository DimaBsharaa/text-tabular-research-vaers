# Validation performed

All checks ran locally against cached sources; no benchmark training or new prediction experiments ran.

- Verified SHA-256 content hashes for **316** downloaded official source/metadata/card/preview files.
- Matched all **5,519** original curation fold scores against the separately cached original `tabstar_corpus` exports, including exact dataset/model/condition/fold keys and numeric agreement below `1e-12`.
- Reconstructed the official five-learner pass matrix, including the documented missing CarDekho TAR fold and four unavailable TabPFN evaluations. Published and reconstructed flags remain separate.
- Checked 56 unique inventory entries, 23 accepted candidates, 40 released profiles, 50 returned rows per preview, matching metadata/viewer total counts, and rates bounded between zero and one.
- Executed all **22** code cells and saved **13** charts and **35** table outputs, with no error outputs. The notebook and standalone HTML open without downloads; ordinary Jupyter execution remains available.
- Visually inspected representative comparison and structure plots for labels, scale, and sample counts. Plot jitter uses a fixed random seed.
- Checked Git whitespace errors and verified both original PDFs remain in the NLP parent folder with their original sizes. GitHub synchronization follows the user’s authorization.

The lightweight execution runner uses one local Python process rather than a Jupyter kernel. This is disclosed in the notebook metadata and research README. Direct dependency versions are pinned; a complete transitive environment lock is not provided for this exploratory research phase.

After the graph-insight update, all 13 chart outputs were checked to have an immediately following interpretation cell, and the complete notebook was re-executed successfully.
