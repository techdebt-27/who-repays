# Does It Matter Who Repays Architecture Technical Debt? An Empirical Study of Structural Changes — Replication Package

Download/unzip the package, open a terminal in the directory containing this `README.md`, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing the Metric Extraction

The script `understand_dependency_complexity_extraction_9_metrics.py` is used to extract dependency and complexity metrics from the analyzed projects. It relies on **Understand by SciTools** to automatically analyze the source code at two lifecycle points for each traced ATD issue: the **introduction commit** and the **payment commit**.

The script uses the traced issue dataset:

```text
dataset/ATD-SELF-FIXED-FINAL-DATASET-TRACED.csv
```

These paths are defined as global variables in `understand_dependency_complexity_extraction_9_metrics.py`. If Understand is installed in a different location, edit the corresponding path variable in the script.

For Windows, update the Understand path as follows:

```python
UNDERSTAND_BIN_WINDOWS = r"C:\Program Files\SciTools\bin\pc-win64"
```

For Linux, update the Understand path as follows:
```python
UNDERSTAND_BIN_LINUX = "/root/scitools/bin/linux64"
```

If your Understand installation is located elsewhere, replace these values with the correct local installation path.


## Start with `understand_dependency_complexity_extraction_9_metrics.py`

This path requires **tracked ATD data**, Git clones with the original commits, and **SciTools Understand + a working license**. Install Java-enabled Understand and the matching Understand Python API. Place one traced CSV per project in `data/input/`, e.g. `CAMEL.csv`, `KAFKA.csv`. The required source columns include `Key`, `indicator` (VioMod), `Is Self-Fixed (Intro=Payment)`, `Intro Hash`, `Payment Hash`, and `Intro_Affected_Files` (semicolon-separated Java paths).


Create a configuration file:

```bash
cp config/config.example.json config/config.json
```

Edit `config/config.json` to point to **disposable clones** and a scratch directory:

```json
{
  "traced_csv_dir": "data/input",
  "repos_root": "/path/to/disposable-apache-clones",
  "work_tmpdir": "/path/to/large-scratch",
  "scitools_bin": "/opt/scitools/bin/linux64",
  "output_dir": "results/generated",
  "permutations": 9999,
  "issue_bootstraps": 1999,
  "file_bootstraps": 1999,
  "contrast_bootstraps": 1999,
  "seed": 20261021
}
```

> **Destructive Git warning.** The original extractor runs `git reset --hard`, `git clean -fdx/-ffdx`, commit checkouts, and may `git pull`. It can delete work in these repositories. Use dedicated **disposable** clones only.

Run the complete source pipeline:

```bash
python run_pipeline.py --config config/config.json \
  --mode source --steps extract,rq1,rq2 \
  --confirm-destructive-git
```

Or, once Understand extraction has finished, repeat just the analyses:

```bash
python run_pipeline.py --config config/config.json \
  --mode source --steps rq1,rq2
```

**Flow:** `data/input/{PROJECT}.csv` → `scripts/extraction/understand_dependency_complexity_extraction_9_metrics.py` → `results/generated/extraction/{FINAL,CK-METRICS}/` → `scripts/pipeline/prepare_rq1_from_extraction.py` → `scripts/pipeline/run_rq1.py` → `scripts/pipeline/run_rq2.py` (invokes `scripts/rq2_pipeline/run_full_rq2.py` for Git commit-frequency reconstruction and cluster-aware RQ2 analysis).

The source path reads **only Java** introduction-linked files at the two endpoint snapshots, and retains rename mappings. Data must already identify SF/NSF and VioMod; the pipeline does **not** discover or classify ATD issues or introducer/fixer identities.

## RQ1

**RQ1:** *Do issue-level structural metric changes between ATD introduction and repayment differ between SF and NSF ATD?*

- Four file-level metrics: Fan-In, Fan-Out, Cyclomatic Complexity (CC), Cognitive Complexity (CoC). Source: Understand dependencies and Lizard/custom complexity (file sums).
- Five class-level metrics: CBO (`CountClassCoupled`), RFC **proxy** (`CountDeclMethodAll`), LCOM (`PercentLackOfCohesion`), DIT (`MaxInheritanceTree`), NOC (`CountClassDerived`). `CountDeclMethodAll` is **not** canonical RFC; CoC is an approximation.
- Pair files across snapshots using Git file-path rename mapping. Pair classes using **mapped file path and exact class name**. **Class/package renames cannot be reliably inferred and are not automatically matched.** Every metric uses only entities available on *both* sides with valid measurements. The issue's delta is the **mean of matched-entity signed changes**: `repayment − introduction`.
- For each of nine metrics: two-sided **Mann–Whitney U** comparing issue-level SF against NSF, **Cliff's delta** oriented `SF − NSF`, **Holm** correction over the nine tests. One issue contributes at most one observation per metric.
- Separate file-level MWU tests are produced as **exploratory** and do not constitute cluster-aware confirmatory inference.

**Outputs:**

| File | Description |
|---|---|
| `results/generated/rq1/RQ1_issue_metric_deltas.csv` | Per-issue matched deltas and pairing counts (source mode), or normalized saved historical deltas (processed mode) |
| `results/generated/rq1/RQ1_primary_issue_level_tests.csv` | Nine RQ1 statistical comparisons, Cliff's delta, Holm p-values |
| `results/generated/rq1/RQ1_issue_level_coverage.csv` | Counts of issues / valid metric observations |
| `results/generated/rq1/RQ1_file_metric_deltas_exploratory.csv` | Matched file endpoints, within-file CK means (source mode only) |
| `results/generated/rq1/RQ1_file_level_exploratory_tests.csv` | Unclustered exploratory file-level test results (source mode only) |
| `results/generated/rq1/RQ1_9_metrics_boxplots.pdf` | Nine-panel boxplot (unless `--no-plots`) |


## RQ2

**RQ2:** *Do file modification frequency and structural metric changes show an association in SF and NSF ATD?*

The source path's RQ2 procedure uses `scripts/rq2_pipeline/legacy_git_ck_extractor.py` to follow files through Git commit histories from introduction to repayment, count commits per original file, and associate them with the already-extracted **nine file-level endpoint measurements**. It pairs file paths and applies VioMod filtering; original file/class pairings and RQ1 issue means are **not** interchangeable with RQ2's within-file CK means.

- **Primary:** One matched-file **mean commit count** and one **mean signed metric delta** per ATD issue/metric. Separate SF and NSF **Spearman rho**, within-project issue-permutation p-values, project-stratified issue bootstrap 95% CI, **Holm over 18 tests** (nine metrics × two groups).
- **Sensitivity:** File-level Spearman, with whole-issue cluster bootstrap within projects for CI and approximate centered-bootstrap p-values; separate Holm family of 18 tests.
- Correlations concern **signed endpoint changes**, not cumulative structural drift or causal effects. Project nesting and overlap of issues/files may remain.

**Outputs:**

| File | Description |
|---|---|
| `results/generated/rq2/RQ3_PRIMARY_issue_spearman_project_permutation.csv` | Primary issue-level RQ2 (manuscript Table IV) |
| `results/generated/rq2/RQ3_SENSITIVITY_file_spearman_issue_cluster_bootstrap.csv` | File-level sensitivity (manuscript Table V) |
| `results/generated/rq2/RQ3_issue_level_matched_file_pairs.csv` | Issue aggregation audit |
| `results/generated/rq2/RQ2_coverage_audit.csv` | Metric/cohort coverage |
| `results/generated/rq2/RQ2_DIRECT_SF_NSF_correlation_difference.csv` | Exploratory direct comparison, **not included in the submitted article** |
| `results/generated/rq2/Figure3_RQ2_file_commits_vs_nine_metric_deltas.pdf` | Figure 3, descriptive scatter plot |


## Folder layout

```text
├── data
│   ├── input
│   │   ├── atd-dataset
│   │   └── ATD-SELF-FIXED-FINAL-DATASET-TRACED.csv
│   └── processed
│       ├── PRIMARY_boxplot_data.csv
│       ├── PRIMARY_boxplot_metadata.csv
│       └── rq3_file_level_nine_metrics.csv
├── figures
│   ├── FILE_LEVEL_ALL_9_METRICS_BOXPLOTS_SF_vs_NSF_ALL_FILES_1x9_small.pdf
│   ├── PRIMARY_ALL_9_METRICS_BOXPLOTS_SF_vs_NSF_1x9_small.pdf
│   └── RQ2_9_METRICS_SCATTER_OLS.pdf
├── scripts
│   ├── config
│   │   └── config.example.json
│   ├── duration_activity
│   │   └── analyze_changes_vs_time_to_fix.py
│   ├── extraction
│   │   ├── README.md
│   │   └── understand_dependency_complexity_extraction_9_metrics.py
│   ├── README.md
│   ├── rq1
│   │   ├── ATD_ALL_METRICS_DIRECT_SF_VS_NSF.py
│   │   └── RQ1_reproduce_from_saved_issue_deltas.py
│   ├── rq2
│   │   ├── analyze_rq2.py
│   │   ├── legacy_git_ck_extractor.py
│   │   ├── requirements.txt
│   │   ├── run_full_rq2.py
│   │   ├── test_extract_git.py
│   │   └── test_pipeline.py
│   ├── run_pipeline.py
│   └── visualization
│       ├── PLOT_ATD_BOXPLOTS_FROM_CSV.py
│       ├── PLOT_ATD_FILE_LEVEL_BOXPLOTS.py
│       ├── PLOT_ATD_FILE_LEVEL_ECDF.py
│       └── RQ3_REPLOT_9_METRICS_WITH_TRENDLINES.py
└── understand_dependency_complexity_extraction_9_metrics.py
```



