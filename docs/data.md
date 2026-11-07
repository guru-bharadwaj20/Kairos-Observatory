# KAIROS dataset datasheet

This page covers every series in `data/processed/`: where it comes from, its licence, how it is built, and what it gets wrong. Rebuild the dataset with `kairos data refresh` (add `--offline` to build from the committed snapshots), and check it with `kairos data verify`.

## Layout and formats

| Path | Content |
|---|---|
| `data/processed/series/<series_id>.parquet` | One series per file. The columns are `ds`, `y`, `series_id`, `side`, `unit`, `source`, `source_group`, `vintage`, `y_lo`, `y_hi`, plus optional flag columns. The `SeriesMeta` and method metadata are stored as JSON in the Parquet key-value metadata (`kairos.meta`, `kairos.extra`). |
| `data/processed/tables/*.parquet` | Auxiliary tables: `epoch_models`, `benchmark_scores`, `regulatory_lag`, `population_history`, `loo_C`, `loo_H`, `quality_report`. Each one records its licence in `kairos.extra`. |
| `data/processed/events.parquet` | The event calendar (`event`, `ds`, `type`, `lower_window`, `upper_window`, `source_url`, `scope`, `description`). |
| `data/MANIFEST.json` | The SHA-256 of every processed file, plus a content `digest` that the forecast ledger cites. |
| `src/kairos/data/snapshots/` | Small committed copies of every raw input, with their dates in `VINTAGES.json`. |

Conventions:

- `y` is always in raw units. Models apply `SeriesMeta.transform` (`log` or `logit`).
- `ds` is the first day of the month.
- `vintage` is the date the input was retrieved.
- `source_group` marks provenance clusters. Series in different groups count as independent (D10).

## Series

| series_id | Side | Source (licence) | Method | Known biases and secrecy notes |
|---|---|---|---|---|
| `c.frontier_compute` | C | Epoch AI notable models (CC BY 4.0) | Top-3 log-mean envelope, interpolated log-linearly between record months. The `gap` flag marks gaps longer than 12 months (D7). | Training compute is often estimated by Epoch rather than disclosed, and frontier labs increasingly withhold it. Recent months under-count unreleased models. |
| `c.metr_horizon` | C | METR (publication terms) | 50% task time horizon. When several models fall in one month, the highest is kept. 95% CI stored in `y_lo`/`y_hi`. | Curated by reading values off METR charts. The task suite favours software tasks, and there are few points. |
| `c.bench.mmlu`, `c.bench.gpqa_diamond`, `c.bench.swe_bench_verified` | C | Model cards and papers (facts) | Record months of self-reported scores. Uses the logit transform. | Scores are developer-reported with different prompting and scaffolds. The benchmarks may be contaminated, and MMLU is saturated. |
| `c.cost_per_mtok` | C | Provider pricing pages (facts) | Cheapest launch list price, blended 3:1 input:output, among models with MMLU ≥ 0.82. Direction −1. | List prices are not costs, and subsidised pricing exaggerates the decline. |
| `c.ai_research_share` | C | Liang et al. 2024 (figures cited) | Proxy: share of LLM-modified sentences in arXiv CS abstracts. | Measures help with writing, not research labour. Values were read from published figures, and the series stops at 2024-02. |
| `exo.compute` | exo | Epoch AI (CC BY 4.0) | Training compute of notable models, summed over a trailing 12 months. | Inherits Epoch's coverage and estimation error. |
| `exo.investment` | exo | OWID / Quid (CC BY 4.0) | Annual world private AI investment, interpolated log-linearly to months. | Private investment only; definitions change between Quid releases. |
| `exo.researchers` | exo | NeurIPS statistics (facts) | Annual main-track submissions, interpolated. ±5% band. | A proxy for the size of the research workforce that is inflated by resubmissions and LLM-assisted papers. |
| `h.safety_papers` | H | OpenAlex (CC0) | Monthly count of works matching a fixed query, version `openalex-2026.11-v1`. The partial current month is dropped. | Works with only a publication year are dated 1 January, so January spikes appear. Keyword matching misses some relevant papers and includes some unrelated ones. |
| `h.safety_researchers` | H | Field surveys (estimates) | Annual full-time-equivalent estimates with bands of roughly ×0.6 to ×1.6. | Order-of-magnitude figures. Internal lab safety teams are mostly undisclosed. |
| `h.safety_funding` | H | Grant databases (estimates) | Annual philanthropic funding in USD millions, with wide bands. | Excludes lab-internal and government spending. |
| `h.eval_share` | H | System cards and evaluator reports (facts) | Per quarter: the share of Epoch-flagged frontier releases with a disclosed third-party pre-deployment evaluation. 90% Wilson interval. | Only disclosed evaluations count, and Epoch's frontier flag is sparse. Small counts give wide intervals. |
| `h.regulatory_lag` | H | OECD.AI policy database (OECD terms) | Months from a capability's first appearance to the first binding rule anywhere. Direction −1. | Only six capabilities so far. Picking the "first rule" involves judgement. |
| `h.ai_literacy` | H | Pew Research Center (facts) | Share of U.S. adults who have used ChatGPT. Linear interpolation with a Brownian-bridge band that widens between surveys. | U.S. only, and measures use of a single product. |
| `idx.C`, `idx.H` | C/H | KAIROS (CC BY 4.0) | Precision-weighted geometric mean of normalised component growth (`composite.py`), with a PCA variant. Base is 2020-01 = 1. `n_components` records coverage. Leave-one-out results are in `loo_C` and `loo_H`. | Only `dG/dt` is meaningful, not the level (D8). Coverage changes over time, and early months rest on a single component. |
| `val.population` | validation | OWID (HYDE, UN WPP; CC BY 4.0) | Annual values from 1700 onward. Older years are in `population_history`. | Pre-1950 values are reconstructions. |
| `val.transistors` | validation | OWID, after Rupp (CC BY 4.0) | Transistors per microprocessor, annual. | Flagship chips only. |
| `val.nasdaq`, `val.bitcoin` | validation | FRED (terms; © Nasdaq, Coinbase) | Monthly mean of daily closes over 1995–2001 and 2015–2018. | Single-exchange prices. |
| `val.covid_cases` | validation | OWID COVID-19 (CC BY 4.0) | World new cases summed by month over 2020–2022. COVID interventions are in the event calendar under scope `covid`. | Testing intensity varies by country and over time. |

## Event calendar

`calendar/events.csv` (v1) covers frontier releases from 2012 to 2025, major laws (EU AI Act milestones, Chinese generative-AI rules, US executive orders, California SB 53), US export controls, and notable incidents. Every row has a source URL. Feed it to Prophet or NeuralProphet with `kairos.data.events.to_holidays`.

## Quality checks

Every refresh runs `kairos.data.quality`. Ordering, duplicate, unit and side problems are errors and stop the write. Gaps and robust-z outliers are warnings and are stored in `tables/quality_report.parquet`.

## Curated inputs

Inputs without a machine-readable upstream are kept as CSV files under `snapshots/`, with `#` provenance comments and a source URL on every row. These sources are METR, benchmarks, prices, the research-share proxy, NeurIPS counts, safety field size, evaluations, regulatory lag, literacy, the event calendar, and COVID interventions. Corrections are made by editing the CSV in a commit tagged with the task ID. Values read off published charts are approximate, and the CSVs say so.
