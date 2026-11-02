# ATD structural-metric extraction — nine metrics

## Scope

`understand_dependency_complexity_extraction_9_metrics.py` extends the researcher's original Understand/Lizard extraction script. It analyzes the source snapshots at an ATD issue's **introduction** and **repayment** commits. It extracts four existing metrics (file/class Fan-In, Fan-Out, Cyclomatic and approximate Cognitive Complexity) and five additional **SciTools Understand class metrics** (CBO, RFC proxy, LCOM, DIT and NOC). The input `Key`, `Intro Hash`, `Payment Hash`, and `Intro_Affected_Files` must already be present in the traced dataset CSV.

The script is an **extractor**, not a reproduction of any numerical result in the TechDebt 2027 manuscript. It does not independently calculate matched-class issue changes, Mann–Whitney U, Holm correction, or Spearman inference.

## Definitions and exact Understand metric IDs

| Research metric | Understand ID | Extraction level | Notes |
|---|---|---|---|
| Fan-In | `file_entity.dependsby()` | file and class | Existing extractor implementation; counts linked dependencies |
| Fan-Out | `file_entity.depends()` | file and class | Existing extractor implementation |
| Cyclomatic Complexity | `lizard` function CCN | file sums and class-matched functions | Existing Lizard implementation |
| Cognitive Complexity | custom approximate algorithm | file sums and class-matched functions | **Not** official SonarQube Cognitive Complexity |
| CBO | `CountClassCoupled` | class | Coupling between objects |
| RFC proxy | `CountDeclMethodAll` | class | Understand's method-count metric, **not classical RFC**; may include inherited methods |
| LCOM | `PercentLackOfCohesion` | class | Percentage, not an arbitrary LCOM variant |
| DIT | `MaxInheritanceTree` | class | Depth of inheritance tree |
| NOC | `CountClassDerived` | class | Number of immediate derived classes |

Canonical source documentation: https://docs.scitools.com/plugins/Solutions/.doc/manager_chidamberkemerer.html and https://docs.scitools.com/python/metric.html.

A missing/unsupported Understand result is saved as JSON `null` (not `0`). A valid NOC of zero stays zero. Per-file CBO, RFC proxy, LCOM, DIT and NOC are arithmetic means over available *class* measurements within the corresponding file. Individual class measurements are saved separately: **do not substitute file means for the class-matched RQ1 analysis**.

## Requirements

- Linux with Git and a valid local clone of the relevant Apache project.
- **SciTools Understand** CLI (`und`) and its compatible Python API; a working license and enabled Java class metrics.
- Python dependencies: `pandas`, `lizard`, `tqdm`.
- Enough disk space for repeated temporary Understand databases (the historical script checks for at least 10 GiB available).

```bash
python -m pip install pandas lizard tqdm
```

Configure Understand via `SCITOOLS_UND_BIN` if its installation differs from `/opt/scitools/bin/linux64`.

## Run

The default configuration still targets `CAMEL.csv`. Paths can now also be overridden without editing source:

```bash
export ATD_DATA_DIRECTORY=/path/to/ICSA-ATD-Lifecycle-To-Replicate
export ATD_INPUT_CSV=/path/to/ATD-DATA/SELF-FIXED/csv_by_project/CAMEL.csv
export ATD_REPOS_ROOT=/path/to/ATD-DATA/repos
export ATD_OUTPUT_CSV=/path/to/ATD-DATA/SELF-FIXED/METRICS/FINAL/CAMEL-UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv
export ATD_CK_OUTPUT_CSV=/path/to/ATD-DATA/SELF-FIXED/METRICS/CK-METRICS/CAMEL-UNDERSTAND-CK-METRICS-9-METRICS.csv
export WORK_TMPDIR=/path/to/large/scratch/directory
python understand_dependency_complexity_extraction_9_metrics.py
```

For other projects, update all per-project CSV paths consistently. The repository folder mapping remains in the source; e.g. `AMQ` maps to `activemq`.

**Warning:** The historical Git workflow uses `git reset --hard`, `git clean -fdx` or `-ffdx`, checkouts, and a possible `git pull`. It can discard uncommitted and untracked data. Use a **disposable dedicated clone**, not a worktree containing ongoing changes. The old code's high disk usage and destructive Git operations are not removed by this extension.

## Output data formats

The main metrics output CSV keeps the original columns, including:

- `Intro_ClassDetails_JSON` / `Payment_ClassDetails_JSON`: the class records now also have `cbo`, `rfc_proxy`, `lcom`, `dit`, `noc` and CK-compatible alias keys such as `cbo_CountClassCoupled` and `rfc_understand_CountDeclMethodAll`.
- `Intro_FileDetails_JSON` / `Payment_FileDetails_JSON`: the existing Fan-In/Fan-Out/complexity values plus `file_cbo_mean`, `file_rfc_understand_mean`, `file_lcom_mean`, `file_dit_mean`, `file_noc_mean`, their numbers of available class observations, and `file_ck_class_count`.
- `Intro_CK_ClassDetails_JSON`, `Payment_CK_ClassDetails_JSON`, `Intro_CK_FileSummary_JSON`, `Payment_CK_FileSummary_JSON`, `Intro_CK_Status`, `Payment_CK_Status`: new CK-compatible columns derived from those same snapshot measurements.

If `ATD_CK_OUTPUT_CSV` is not an empty string, a second CSV is written to that path. By default it is named like `CAMEL-UNDERSTAND-CK-METRICS-9-METRICS.csv` under `CK-METRICS/`. It includes the CK columns required by the existing *RQ2 (historically RQ3)* file-level pipeline. It uses the **same** Git snapshots as the main CSV, instead of a separate run.

When resuming a historical output with only the original four metrics, the code detects the absent CK keys and recomputes the incomplete class/file JSON. To rerun with altered metric definitions or a different Understand version, **save/archive and remove the old output CSV** first; values that already have the nine keys are otherwise skipped. Do not merge numbers computed with different versions without an audit.

## Validation, limitations and interpretation

Seven synthetic unit tests cover the added metric IDs, aliases, preservation of missing values, class and file aggregation, old-output refresh detection, CK-format export, and a historical subprocess compatibility fix. Run:

```bash
python -m unittest discover -s tests -p 'test_extraction_9_metrics.py' -v
```

The tests do **not** validate actual SciTools Understand outputs or empirical Apache results. To verify those, run a small known ATD issue against your installed Understand version and compare its class metrics and file sums to the GUI/CK export; confirm all five CK metrics are present and the extracted numbers/LCOM scale match the manuscript's original data. RQ1 **matches individual classes across snapshots**; RQ2 uses within-file class means, yielding different weighting. Missing classes/files should not be treated as zero. Endpoint changes are repayment minus introduction and do not describe intermediate trajectories.
