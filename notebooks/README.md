# Team-contributed DBB notebooks

These notebooks were contributed by another team member and organized separately from the existing `Code/` application and experiment scripts. They are **not** interchangeable with the team's later Phase 2 (v12) first-token-logit replication.

| Notebook | Purpose | Status |
| --- | --- | --- |
| [EDA](eda/dbb_eda.ipynb) | Real-data exploratory analysis, category distributions, descriptors and concept diagnostics | Primary EDA; upstream optional files may be missing |
| [Multi-model 1k](experiments/multimodel_1k_option_swapping.ipynb) | 1,000-row stratified, generation-based experiment with option swapping across three LLMs | Legacy experimental protocol; **not** Phase 2 logit scoring |
| [Qwen 1k](experiments/qwen2_5_7b_1k_generation.ipynb) | Qwen2.5-7B-Instruct generation-based first-1,000-row experiment | Legacy experimental protocol; not stratified |
| [Legacy EDA + plots](legacy/dbb_eda_and_result_visualization.ipynb) | Earlier combined EDA and visualization workflow | Archived for provenance; use the primary EDA for new work |

## Setup

1. Run in Google Colab with GPU for model inference notebooks. The EDA notebooks also need pandas, numpy, matplotlib and seaborn.
2. Clone this repository or upload the real DBB CSV. The canonical dataset in this repository is `Dataset/Bias-Dataset.csv` (uppercase `Dataset`).
3. The notebooks retain original Colab Drive defaults for compatibility with their author. Override them via environment variables before running, e.g.:

```python
import os
os.environ["DBB_DATASET_PATH"] = "/content/dbb-benchmark/Dataset/Bias-Dataset.csv"
os.environ["DBB_SOURCE_DATA_DIR"] = "/content/dbb-benchmark"
os.environ["DBB_OUTPUT_DIR"] = "/content/dbb-results"
```

4. Install model dependencies for inference notebooks (`torch`, `transformers`, `accelerate`, `bitsandbytes`, `pandas`, `seaborn`, `tqdm`). Running full inference requires downloading the relevant open-weight models and enough GPU memory.

## Scientific and reproducibility cautions

- The **multi-model notebook samples 200 rows per category**, up to 1,000 total. The **Qwen notebook takes the first 1,000 rows**. These are different selection procedures; do not pool their results or label them the Phase 2 5,420-question run.
- Both generation-based notebooks use `model.generate()`; they do **not** implement the Phase 2 first-token probability estimator or its bootstrap analysis.
- `ans1` and `ans2` are *candidate concepts*, not globally labeled stereotype/anti-stereotype answers. Any legacy plot calling option A a stereotype should be interpreted cautiously.
- The original legacy combined EDA notebook **generated fake data** if CSV files were missing and **generated synthetic model results** if observed positive counts were zero. Both fallbacks have been removed. Missing inputs now raise an error; zero valid positive observations must remain zero.
- Legacy `Is_True_Biased` indicates a particular pattern of answers under option swapping; it is **not proof of social discrimination**. The visualizations remain exploratory and should not be used as validated DBB estimates without auditing label alignment and model outputs.
- Original saved execution outputs and notebook runtime metadata were cleared to avoid large, stale, or misleading committed results. No inference or full Colab execution was performed as part of this refactor.
- Optional upstream files (questions and descriptor metadata) used by the primary EDA may be unavailable in this repository. Their absence should be disclosed rather than silently replaced.

## Change scope

The contribution focuses on repository structure, clean version-control notebooks, portable dataset/output paths, basic safety checks, and documentation. It does **not** rewrite the original research algorithms or assert their statistical correctness. Review the legacy scoring logic before using the notebooks as evidence in a report.
