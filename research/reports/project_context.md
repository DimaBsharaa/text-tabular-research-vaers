# Agreed scope and open decisions

Based on the pasted context, course PDFs, and the recent decisions in the [shared conversation](https://chatgpt.com/share/6acf93cf-eb7c-83eb-858a-783dbdbaf1b8), retrieved on October 14, 2026.

- Switch from Track 1 to **Standard Track 2**: one defensible new dataset, evaluated using the official five learners and four conditions.
- **No dataset, sport, domain, or target is chosen.** NFL draft scouting and tennis previews illustrate the desired logic: qualitative expert context may add information beyond statistics. They are possibilities, not commitments or verified source coverage.
- First investigate the historical candidates in a notebook with visible outputs. Compare similar datasets, condition-level failures, text properties, row counts, and structured features. Challenge hypotheses instead of agreeing with every idea.
- Separate 56 original candidates, 23 passing candidates, 20 core datasets, and 20 extended releases. Extended does not mean failed.
- For later candidate screening, measure complete matched examples after joining structured records, text, and targets. Large source tables do not guarantee enough usable rows.
- Ensure text and structured inputs predate the outcome. Draft projections, eventual selections, post-match reports, and other outcome-bearing fields need careful timing and leakage review.
- Define the unit of observation and entity/time split before future modeling; do not repeatedly tune against a test set to force passage.
- Keep this independent repository separate from MulTaBench. No fork, full benchmark clone, submodule, code modifications, push, large dataset download, or expensive training in this phase.
- Stop after research and review the evidence together before constructing a new dataset or choosing a task.

The current user instruction permits broader EDA to help select an idea and requires notebook outputs. The resulting notebook follows that scope. The conversation's earlier algorithm ideas and any superseded fork suggestions are not the current plan. Prior source or model claims were verified independently where used in this research.
