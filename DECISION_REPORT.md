# Decision Report

Generated: `2026-06-14T08:05:06+00:00`
Manifest: `/Users/jiachenzhao/Projects/PDPO/experiments/manifest.yaml`
Result roots: `/Users/jiachenzhao/Projects/PDPO/experiments/run_queue`

## Gate 1: LIMO Pareto-Competitive

Question: per `(dataset, eps)`, is LIMO Pareto-competitive vs rDPO / RE-DPO / MLE / MAP?

Rule: `Yes` only when LIMO utility is within the baseline bootstrap CI and LIMO mean MIA AUC is strictly lower.

| Dataset | Eps | Overall | vs rDPO | vs RE-DPO | vs MLE | vs MAP | LIMO metrics |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| hhrlhf | 1 | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | NA |
| hhrlhf | 2 | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | NA |
| pku | 1 | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | NA |
| pku | 2 | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | NA |
| truthy | 1 | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | NA |
| truthy | 2 | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | NA |

## Gate 2: E2 LIMO vs Random Subset

Question: does private-scorer LIMO beat random subset at matched keep on utility and/or MIA?

### Per Dataset

| Dataset | Eps covered | Utility win all eps | MIA win all eps |
| --- | ---: | --- | --- |
| hhrlhf | 0 | Insufficient data | Insufficient data |
| pku | 0 | Insufficient data | Insufficient data |
| truthy | 0 | Insufficient data | Insufficient data |

### Per Dataset/Eps Detail

| Dataset | Eps | Utility win | MIA win | LIMO | Random subset |
| --- | ---: | --- | --- | --- | --- |
| hhrlhf | 1 | Insufficient data | Insufficient data | NA | NA |
| hhrlhf | 2 | Insufficient data | Insufficient data | NA | NA |
| pku | 1 | Insufficient data | Insufficient data | NA | NA |
| pku | 2 | Insufficient data | Insufficient data | NA | NA |
| truthy | 1 | Insufficient data | Insufficient data | NA | NA |
| truthy | 2 | Insufficient data | Insufficient data | NA | NA |

## Anomalies

- `60` manifest seed cells are missing.

## Cost

- Ledger: `cost_ledger.csv`
- Ledger rows: `0`
- Estimated GPU-hours: `0.0000`
- Estimated spend: `$0.00`
- Credit ceiling: `$200.00`
- Estimated remaining credit: `$200.00`

## Inputs

- Valid result JSONs: `0`
- Missing manifest seed cells: `60`
- Stale cells: `0`
- Extra valid result JSONs outside manifest: `0`
- Invalid JSONs skipped: `0`
