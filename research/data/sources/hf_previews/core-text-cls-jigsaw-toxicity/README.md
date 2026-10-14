---
tags:
- multabench
- tabular
---

# BIN_TEXT_JIGSAW_TOXICITY

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Is Toxic`
- **Rows:** 100000
- **Original source:** https://www.openml.org/search?type=data&id=46654

Load the benchmark splits with `multabench.load_split("BIN_TEXT_JIGSAW_TOXICITY", fold=0)`.

| Column | Type |
|---|---|
| comment_text | string |
| asian | float |
| atheist | float |
| bisexual | float |
| black | float |
| buddhist | float |
| christian | float |
| female | float |
| heterosexual | float |
| hindu | float |
| homosexual_gay_or_lesbian | float |
| intellectual_or_learning_disability | float |
| jewish | float |
| latino | float |
| male | float |
| muslim | float |
| other_disability | float |
| other_gender | float |
| other_race_or_ethnicity | float |
| other_religion | float |
| other_sexual_orientation | float |
| physical_disability | float |
| psychiatric_or_mental_illness | float |
| transgender | float |
| white | float |
| created_date | datetime |
| funny | int |
| wow | int |
| sad | int |
| likes | int |
| disagree | int |
| Is Toxic | bool |
