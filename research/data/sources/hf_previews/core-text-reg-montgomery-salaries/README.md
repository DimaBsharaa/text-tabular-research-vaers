---
tags:
- multabench
- tabular
---

# REG_TEXT_MONTGOMERY_SALARIES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `current_annual_salary`
- **Rows:** 9228
- **Original source:** https://www.openml.org/search?type=data&id=42125

Load the benchmark splits with `multabench.load_split("REG_TEXT_MONTGOMERY_SALARIES", fold=0)`.

| Column | Type |
|---|---|
| full_name | string |
| gender | categorical |
| 2016_gross_pay_received | float |
| 2016_overtime_pay | float |
| department | categorical |
| department_name | categorical |
| division | string |
| assignment_category | categorical |
| employee_position_title | string |
| underfilled_job_title | categorical |
| date_first_hired | datetime |
| current_annual_salary | float |
