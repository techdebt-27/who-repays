# Does It Matter Who Repays Architecture Technical Debt? — Replication Package

**TechDebt 2027 | Nine Structural Metrics | RQ1 + RQ2 | Research reproduction draft**

This package provides a **single entry point** for producing the RQ1 and RQ2 analyses in the nine-metric manuscript. It contains the updated source extractor, explicit RQ1 preparation/statistical code, the previously audited RQ2 Git-tracking and cluster-aware statistical pipeline, reference results, and a **bundled processed dataset**. It does not modify the original GitHub repository.

## Start here (no Understand license needed)

Download/unzip the package, open a terminal in the directory containing this `README.md`, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py --mode processed --steps rq1,rq2
```

**What this runs:** RQ1 from the *saved, originally class-matched issue-level deltas* and RQ2 from the supplied historical 9-metric file-within-issue data. These inputs permit statistical reproduction, **not re-extraction of source code**. The bundled RQ2 CSV is about 93 MB uncompressed; RQ2's 9,999 permutations / 1,999 bootstrap draws are computationally intensive.

For a faster **code check only** (inferential p-values/CI are *not* manuscript-quality):

```bash
python run_pipeline.py --mode processed --steps rq1,rq2 --fast
```

You can also run one research question at a time:

```bash
python run_pipeline.py --mode processed --steps rq1
python run_pipeline.py --mode processed --steps rq2
```

The `--fast` setting uses 199 permutations and 199 bootstrap draws, which is the minimum allowed by the historical RQ2 analyzer. Never cite fast-mode p-values or bootstrap confidence intervals.

## Full path: start with `understand_dependency_complexity_extraction_9_metrics.py`

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

> **Destructive Git warning.** The original extractor runs `git reset --hard`, `git clean -fdx/-ffdx`, commit checkouts, and may `git pull`. It can delete work in these repositories. Use dedicated **disposable** clones only, not development working trees. The explicit confirmation flag acknowledges this risk.

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

## Exactly what RQ1 measures

**RQ1:** *Do issue-level structural metric changes between ATD introduction and repayment differ between SF and NSF ATD?*

- Four file-level metrics: Fan-In, Fan-Out, Cyclomatic Complexity (CC), Cognitive Complexity (CoC). Source: Understand dependencies and Lizard/custom complexity (file sums).
- Five class-level metrics: CBO (`CountClassCoupled`), RFC **proxy** (`CountDeclMethodAll`), LCOM (`PercentLackOfCohesion`), DIT (`MaxInheritanceTree`), NOC (`CountClassDerived`). `CountDeclMethodAll` is **not** canonical RFC; CoC is an approximation, **not** Sonar's official implementation.
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

**Critical distinction:** Reconstructing class pairs using *this* extractor may differ from the original unpublished class-matching decisions in the saved RQ1 dataset. The processed mode reproduces tests on saved issue-level data; source mode provides a **transparent, independently re-executable pairing procedure**, not guaranteed exact manuscript numbers. Inspect the pairing/coverage audit before comparing results.

## Exactly what RQ2 measures

**RQ2:** *Do file modification frequency and structural metric changes show an association in SF and NSF ATD?*

The source path's RQ2 procedure uses `scripts/rq2_pipeline/legacy_git_ck_extractor.py` to follow files through Git commit histories from introduction to repayment, count commits per original file, and associate them with the already-extracted **nine file-level endpoint measurements**. It pairs file paths and applies VioMod filtering; original file/class pairings and RQ1 issue means are **not** interchangeable with RQ2's within-file CK means.

- **Primary:** One matched-file **mean commit count** and one **mean signed metric delta** per ATD issue/metric. Separate SF and NSF **Spearman rho**, within-project issue-permutation p-values, project-stratified issue bootstrap 95% CI, **Holm over 18 tests** (nine metrics × two groups).
- **Sensitivity:** File-level Spearman, with whole-issue cluster bootstrap within projects for CI and approximate centered-bootstrap p-values; separate Holm family of 18 tests.
- The source analysis also writes *exploratory* direct SF/NSF correlation-difference tests. These are kept for audit only, **not** part of the currently shortened manuscript tables.
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

`RQ3_*` filenames are historical: they refer to the **current manuscript's RQ2**, not a third present-day research question.

## Folder layout

```text
TechDebt_2027_Replication_Easy_Pipeline/
├── run_pipeline.py                   # one-command launcher
├── config/config.example.json        # paths, seeds, inferential iterations
├── requirements.txt
├── data/
│   ├── input/                       # place original per-project traced ATD CSVs here
│   └── processed/                   # bundled historical nine-metric CSVs
├── scripts/
│   ├── extraction/understand_dependency_complexity_extraction_9_metrics.py
│   ├── pipeline/
│   │   ├── prepare_rq1_from_extraction.py
│   │   ├── normalize_saved_rq1.py
│   │   ├── run_rq1.py
│   │   └── run_rq2.py
│   ├── rq2_pipeline/                 # historical Git + statistical backend
│   └── ...                           # provenance, independent verifiers, visuals
├── tests/
└── results/
    ├── reference/                   # historical reference CSVs (not freshly generated)
    └── generated/                   # fresh analysis outputs
```

## Tests and troubleshooting

```bash
python -m unittest discover -s tests -p test_easy_pipeline.py -v
python -m unittest discover -s tests -p test_extraction_9_metrics.py -v
python tests/run_smoke_tests.py
```

The bundled tests use mock/synthetic inputs and do not need SciTools Understand. They check extraction metrics, pairing, number of observations, statistical correction, Git-tracking logic, and nine-metric RQ2 data flow. **They do not certify empirical replication of the Apache repos.**

If a source run fails, check: that the original traced CSV has a VioMod `indicator`, exact SF/NSF labels and valid Git hashes; that local repository folder names correspond to the project mapping in `scripts/extraction/`; that the Java classes are matched at both endpoints; that SciTools API and license work; and that there is sufficient free scratch disk space. Inspect the RQ1 coverage table and RQ2 extraction audits instead of substituting missing metrics with zero.

## Limitations and artifact provenance

This is a revised research artifact built from available user research source scripts and data. The old repository [`techdebt-26/who-repays-old`](https://github.com/techdebt-26/who-repays-old) was not read or modified directly. No original repository license is invented; verify licensing before republishing the combined artifact. Version the SciTools Understand CLI/Python API, Lizard and Git used for any fresh extraction, and compare input issue counts, file/class match counts, software versions, and CSV schema with the paper. The primary RQ1 Mann–Whitney test is issue-level but does not explicitly model project dependence. File-level RQ1 p-values are exploratory because observations are nested within issues. The RQ2 project-stratified inference treats analyzed projects as fixed and is observational.

See `scripts/extraction/README.md`, `docs/PREVIOUS_PACKAGE_README.md`, and `docs/SOURCE_PROVENANCE.md` for technical detail and earlier workflows.
