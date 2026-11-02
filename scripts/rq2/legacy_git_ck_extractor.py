#!/usr/bin/env python3
"""Nine-metric RQ3: ATD-related file changes versus structural metric deltas.

The original four metrics come from FINAL per-file JSON. Five CK metrics come from
Intro_CK_FileSummary_JSON / Payment_CK_FileSummary_JSON in CK-METRICS CSVs.
Each CK value is a within-file class mean (no issue-level metric substitution).
Optional: --primary-cohort PRIMARY_boxplot_data.csv aligns the issue cohort with RQ1/RQ2.
Optional: --cached-git previously_saved_file_level.csv skips repository tracking.

Provides 18 primary SF/NSF correlations (one Holm family), nine separately
corrected pooled correlations (exploratory), issue-aggregated sensitivity,
coverage audit, Git-file tracing audit, CSVs, and publication-style figures.

Per-file metrics are preflight-validated before expensive Git tracking; verify the
candidate metric aliases and file aggregation rules before using the results.
Simple file-level Spearman p-values do not account for issue/project clustering.
"""

import os
import math
import re
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ============================================================
# Input / Output Configuration
# ============================================================

INPUT_DIR = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/ATD-DATA/SELF-FIXED/METRICS/FINAL"

CSV_PATTERN = "*UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv"
CK_DIR = str(Path(INPUT_DIR).parent / "CK-METRICS")
CK_PATTERN = "*-UNDERSTAND-CK-METRICS*.csv"

REPOS_ROOT = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/repos"

OUTPUT_DIR = os.path.join(INPUT_DIR, "RQ3_FILE_LEVEL_9_METRICS")

# Create output only after parsing CLI; importing must not touch host paths.



# ============================================================
# Filter Configuration
# ============================================================

FILTER_INDICATOR = "viomod"


# ============================================================
# Repository Configuration
# ============================================================

AUTO_CLONE_REPOSITORY = False  # Explicit opt-in via CLI --allow-network
GITHUB_ORG_DEFAULT = "apache"

FETCHED_REPOSITORIES: Set[str] = set()


# ============================================================
# Output Files
# ============================================================

OUTPUT_FILE_LEVEL_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_file_level_nine_metrics.csv"
)

OUTPUT_SPEARMAN_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_spearman_nine_metrics.csv"
)

OUTPUT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_descriptive_nine_metrics.csv"
)

OUTPUT_REPO_SELECTION_LOG_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_repository_selection_log.csv"
)

OUTPUT_DEPENDENCY_SCATTER_PDF = os.path.join(
    OUTPUT_DIR,
    "rq3_dependency_vs_commit_count.pdf"
)

OUTPUT_DEPENDENCY_SCATTER_PNG = os.path.join(
    OUTPUT_DIR,
    "rq3_dependency_vs_commit_count.png"
)

OUTPUT_COMPLEXITY_SCATTER_PDF = os.path.join(
    OUTPUT_DIR,
    "rq3_complexity_vs_commit_count.pdf"
)

OUTPUT_COMPLEXITY_SCATTER_PNG = os.path.join(
    OUTPUT_DIR,
    "rq3_complexity_vs_commit_count.png"
)


# ============================================================
# Plot Configuration
# ============================================================

PLOT_DPI = 300
PLOT_MARKER_SIZE = 22
PLOT_ALPHA = 0.65
PLOT_SHOW_GRID = True

GROUP_PLOT_LABELS = {
    "Self-Fixed": "Self-fixed",
    "Non-Self-Fixed": "Non-self-fixed",
}

GROUP_COLORS = {
    "Self-Fixed": "#1f77b4",
    "Non-Self-Fixed": "#ff7f0e",
}


# ============================================================
# Exact JSON keys: inspect your exported Intro_FileDetails_JSON and
# Payment_FileDetails_JSON. These are *candidate* aliases, not validated
# Understand metric mappings. In particular, do not interpret CountDeclMethodAll
# as conventional RFC. The CK metrics are read exclusively from file-level CK summaries, below.
# ============================================================

METRIC_JSON_KEYS = {
    "fan_in": ("file_fan_in", "fan_in"),
    "fan_out": ("file_fan_out", "fan_out"),
    "cyclomatic_complexity": (
        "file_cyclomatic_complexity_sum", "cyclomatic_complexity_sum",
        "file_cyclomatic_complexity", "cyclomatic_complexity",
    ),
    "cognitive_complexity": (
        "file_cognitive_complexity_sum", "cognitive_complexity_sum",
        "file_cognitive_complexity", "cognitive_complexity",
    ),
    "cbo": (), "rfc_proxy": (), "lcom": (), "dit": (), "noc": (),
}
METRIC_LABELS = {
    "fan_in": "Fan-In", "fan_out": "Fan-Out",
    "cyclomatic_complexity": "Cyclomatic complexity",
    "cognitive_complexity": "Cognitive complexity",
    "cbo": "CBO", "rfc_proxy": "RFC proxy (CountDeclMethodAll)",
    "lcom": "LCOM", "dit": "DIT", "noc": "NOC",
}
# Treat 18 SF/NSF Spearman tests as a separate RQ3 Holm family.
# Pooled All correlations are exploratory and corrected in their own family.
RQ3_PRIMARY_GROUPS = ("Self-Fixed", "Non-Self-Fixed")
STRICT_NINE_METRICS = True
# Optional: absolute path to PRIMARY_boxplot_data.csv for the same ATD cohort.
PRIMARY_COHORT_CSV = None

# Exact keys verified in the uploaded GEODE CK export.
# Each value is the arithmetic mean across classes in ONE file, at ONE phase.
CK_FILE_METRIC_KEYS = {
    "cbo": "file_cbo_mean",
    "rfc_proxy": "file_rfc_understand_mean",
    "lcom": "file_lcom_mean",
    "dit": "file_dit_mean",
    "noc": "file_noc_mean",
}
CK_INDEX = {}
OUTPUT_METRIC_COVERAGE = os.path.join(OUTPUT_DIR, "rq3_metric_coverage.csv")
OUTPUT_ALL_SCATTER_PDF = os.path.join(OUTPUT_DIR, "rq3_all_nine_scatter_3x3.pdf")
OUTPUT_ALL_SCATTER_PNG = os.path.join(OUTPUT_DIR, "rq3_all_nine_scatter_3x3.png")

# ============================================================
# Repository Mapping
# ============================================================

PROJECT_REPO_FOLDER_OVERRIDES = {
    "https://github.com/apache/drill": "drill",
    "https://github.com/apache/camel": "camel",
    "https://github.com/apache/kafka": "kafka",
    "https://github.com/apache/cassandra": "cassandra",
    "https://github.com/apache/cassandra-analytics": "cassandra-analytics",
    "https://github.com/apache/activemq": "activemq",
    "https://github.com/apache/lucene": "lucene",
    "https://github.com/apache/solr": "solr",
    "https://github.com/apache/geode": "geode",
    "https://github.com/apache/netbeans": "netbeans",
}

PROJECT_KEY_TO_REPO_FOLDER = {
    "DRILL": "drill",
    "CAMEL": "camel",
    "KAFKA": "kafka",
    "CASSANDRA": "cassandra",
    "AMQ": "activemq",
    "LUCENE": "lucene",
    "SOLR": "solr",
    "GEODE": "geode",
    "NETBEANS": "netbeans",
}

ALLOWED_REPO_FOLDERS = {
    "drill",
    "camel",
    "kafka",
    "cassandra",
    "cassandra-analytics",
    "activemq",
    "lucene",
    "solr",
    "geode",
    "netbeans",
}

PROJECT_REPO_CANDIDATES = {
    "CASSANDRA": [
        {
            "repo_folder": "cassandra",
            "clone_url": "https://github.com/apache/cassandra.git",
        },
        {
            "repo_folder": "cassandra-analytics",
            "clone_url": "https://github.com/apache/cassandra-analytics.git",
        },
    ]
}


# ============================================================
# Basic Utilities
# ============================================================

def normalize_path(path: Any) -> Optional[str]:
    if path is None or pd.isna(path):
        return None

    path = str(path).strip()
    path = path.strip('"').strip("'")
    path = path.replace("\\", "/")
    path = re.sub(r"/+", "/", path)

    if path.startswith("./"):
        path = path[2:]

    return path


def is_java_file(path: Any) -> bool:
    path = normalize_path(path)

    if not path:
        return False

    return path.lower().endswith(".java")


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [] if default is None else default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except (ValueError, TypeError):
        return [] if default is None else default


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def parse_intro_affected_files(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []

    files = []

    for item in re.split(r";|\n|\r", str(value)):
        item = normalize_path(item)

        if item and is_java_file(item):
            files.append(item)

    return sorted(set(files))


def delta_value(payment_value: Any, intro_value: Any) -> float:
    payment_value = pd.to_numeric(payment_value, errors="coerce")
    intro_value = pd.to_numeric(intro_value, errors="coerce")

    if pd.isna(payment_value) or pd.isna(intro_value):
        return np.nan

    return float(payment_value - intro_value)


# ============================================================
# Load Input CSVs
# ============================================================

def load_all_input_csvs(input_dir: str, csv_pattern: str) -> pd.DataFrame:
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob(csv_pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{input_dir}\nwith pattern:\n{csv_pattern}"
        )

    print("============================================================")
    print("Input CSV Files")
    print("============================================================")

    dfs = []

    for csv_file in csv_files:
        print(f"Loading: {csv_file}")
        temp_df = pd.read_csv(csv_file)
        temp_df["source_csv_file"] = csv_file.name
        dfs.append(temp_df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print("============================================================")
    print(f"Total CSV files loaded : {len(csv_files)}")
    print(f"Total rows combined    : {len(combined_df)}")
    print("============================================================\n")

    return combined_df


# ============================================================
# Git Utilities
# ============================================================

def run_git(repo_dir: Path, args: List[str], allow_error: bool = False) -> str:
    cmd = ["git", "-C", str(repo_dir)] + args

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0 and not allow_error:
        raise RuntimeError(
            f"Git command failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def fetch_all_branches_and_tags(repo_dir: Path) -> None:
    repo_key = str(repo_dir)

    if repo_key in FETCHED_REPOSITORIES:
        return

    print(f"[FETCH] Updating repository once: {repo_dir}")

    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "fetch",
            "--all",
            "--tags",
            "--prune"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    FETCHED_REPOSITORIES.add(repo_key)


def commit_exists(repo_dir: Path, commit_hash: Any) -> bool:
    if commit_hash is None or pd.isna(commit_hash):
        return False

    commit_hash = str(commit_hash).strip()

    if not commit_hash:
        return False

    def _cat_file_exists() -> bool:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "cat-file",
                "-e",
                f"{commit_hash}^{{commit}}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0

    if _cat_file_exists():
        return True

    if not AUTO_CLONE_REPOSITORY:
        return False
    fetch_all_branches_and_tags(repo_dir)

    if _cat_file_exists():
        return True

    print(f"[FETCH-SHA] Commit not local. Trying exact fetch: {commit_hash}")

    fetch_attempts = [
        ["fetch", "origin", commit_hash],
        ["fetch", "origin", commit_hash, "--depth=1"],
        ["fetch", "origin", f"+{commit_hash}:refs/temp/commit-{commit_hash[:12]}"],
    ]

    for args in fetch_attempts:
        subprocess.run(
            ["git", "-C", str(repo_dir)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if _cat_file_exists():
            print(f"[OK] Commit fetched successfully: {commit_hash}")
            return True

    print(f"[MISS] Commit still not found after exact fetch: {commit_hash}")

    return False


def get_parent_commit(repo_dir: Path, commit_hash: str) -> Optional[str]:
    output = run_git(
        repo_dir,
        ["rev-list", "--parents", "-n", "1", str(commit_hash).strip()],
        allow_error=True
    )

    parts = output.split()

    if len(parts) >= 2:
        return parts[1]

    return None


def build_commit_range(
    repo_dir: Path,
    intro_hash: str,
    payment_hash: str,
    include_intro: bool = True
) -> str:
    intro_hash = str(intro_hash).strip()
    payment_hash = str(payment_hash).strip()

    if include_intro:
        parent = get_parent_commit(repo_dir, intro_hash)

        if parent:
            return f"{parent}..{payment_hash}"

        return payment_hash

    return f"{intro_hash}..{payment_hash}"


def get_files_at_commit(repo_dir: Path, commit_hash: str) -> Set[str]:
    output = run_git(
        repo_dir,
        ["ls-tree", "-r", "--name-only", str(commit_hash).strip()],
        allow_error=False
    )

    return {
        normalize_path(line)
        for line in output.splitlines()
        if normalize_path(line)
    }


# ============================================================
# Fast Git Log Parsing
# ============================================================

def parse_git_log_name_status(output: str) -> List[Dict[str, Any]]:
    records = []
    current_record = None

    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")

        if not line:
            continue

        if line.startswith("__COMMIT__"):
            commit_hash = line.replace("__COMMIT__", "").strip()

            current_record = {
                "commit": commit_hash,
                "changes": []
            }

            records.append(current_record)
            continue

        if current_record is None:
            continue

        parts = line.split("\t")

        if not parts:
            continue

        status = parts[0].strip()

        if status.startswith("R") or status.startswith("C"):
            if len(parts) >= 3:
                current_record["changes"].append({
                    "status": status,
                    "old_path": normalize_path(parts[1]),
                    "new_path": normalize_path(parts[2]),
                })
        else:
            if len(parts) >= 2:
                path = normalize_path(parts[1])
                current_record["changes"].append({
                    "status": status,
                    "old_path": path,
                    "new_path": path,
                })

    return records


def get_issue_change_log(repo_dir: Path, commit_range: str) -> List[Dict[str, Any]]:
    output = run_git(
        repo_dir,
        [
            "log",
            "--reverse",
            "--name-status",
            "-M",
            "-C",
            "--format=__COMMIT__%H",
            commit_range
        ],
        allow_error=False
    )

    return parse_git_log_name_status(output)


def fast_track_files_for_issue(
    repo_dir: Path,
    intro_hash: str,
    payment_hash: str,
    intro_files: List[str],
    include_intro: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    intro_files = sorted(
        set([
            normalize_path(p)
            for p in intro_files
            if is_java_file(p)
        ])
    )

    commit_range = build_commit_range(
        repo_dir=repo_dir,
        intro_hash=intro_hash,
        payment_hash=payment_hash,
        include_intro=include_intro
    )

    files_at_intro = get_files_at_commit(repo_dir, intro_hash)

    tracking: Dict[str, Dict[str, Any]] = {}

    for intro_file in intro_files:
        exists_at_intro = intro_file in files_at_intro

        tracking[intro_file] = {
            "intro_file": intro_file,
            "current_path": intro_file if exists_at_intro else None,
            "final_path": intro_file if exists_at_intro else None,
            "exists_at_intro": exists_at_intro,
            "deleted": False,
            "rename_count": 0,
            "touched_commits": [],
            "path_history": [intro_file],
            "tracking_status": (
                "tracked_to_payment"
                if exists_at_intro
                else "intro_file_not_found_at_intro_commit"
            ),
        }

    commit_records = get_issue_change_log(repo_dir, commit_range)

    for record in commit_records:
        commit_hash = record["commit"]
        changes = record["changes"]

        current_path_to_intro_files: Dict[str, List[str]] = {}

        for intro_file, state in tracking.items():
            if state["deleted"]:
                continue

            current_path = state["current_path"]

            if current_path:
                current_path_to_intro_files.setdefault(
                    current_path,
                    []
                ).append(intro_file)

        for change in changes:
            status = change["status"]
            old_path = change["old_path"]
            new_path = change["new_path"]

            if status.startswith("R"):
                if old_path in current_path_to_intro_files:
                    for intro_file in current_path_to_intro_files[old_path]:
                        state = tracking[intro_file]
                        state["touched_commits"].append(commit_hash)
                        state["current_path"] = new_path
                        state["final_path"] = new_path
                        state["rename_count"] += 1
                        state["path_history"].append(new_path)

            elif status.startswith("C"):
                # A copy from the tracked file does not modify that source file.
                # Do not increment its commit count for copied descendants.
                pass

            elif status == "D":
                if old_path in current_path_to_intro_files:
                    for intro_file in current_path_to_intro_files[old_path]:
                        state = tracking[intro_file]
                        state["touched_commits"].append(commit_hash)
                        state["deleted"] = True
                        state["final_path"] = None
                        state["current_path"] = None
                        state["tracking_status"] = "deleted_before_or_at_payment"
                        state["path_history"].append("[DELETED]")

            else:
                candidate_paths = set()

                if old_path:
                    candidate_paths.add(old_path)

                if new_path:
                    candidate_paths.add(new_path)

                for candidate_path in candidate_paths:
                    if candidate_path in current_path_to_intro_files:
                        for intro_file in current_path_to_intro_files[candidate_path]:
                            state = tracking[intro_file]
                            state["touched_commits"].append(commit_hash)

    for intro_file, state in tracking.items():
        unique_commits = list(dict.fromkeys(state["touched_commits"]))
        state["touched_commits"] = unique_commits
        state["commit_count"] = len(unique_commits)

        if state["tracking_status"] == "tracked_to_payment" and not state["deleted"]:
            state["final_path"] = state["current_path"]

    return tracking, commit_range


# ============================================================
# Repository Utilities
# ============================================================

def get_project_key_from_row(row: pd.Series) -> str:
    key = str(row.get("Key", "")).strip().upper()

    if "-" in key:
        return key.split("-")[0]

    github_url = str(row.get("GitHub Base URL", "")).strip().rstrip("/")

    if github_url and github_url.lower() != "nan":
        return github_url.split("/")[-1].replace(".git", "").upper()

    return ""


def get_repo_folder_from_row(row: pd.Series) -> str:
    github_url = str(row.get("GitHub Base URL", "")).strip().rstrip("/")

    if github_url in PROJECT_REPO_FOLDER_OVERRIDES:
        return PROJECT_REPO_FOLDER_OVERRIDES[github_url]

    key = str(row.get("Key", "")).strip().upper()

    if "-" in key:
        project_key = key.split("-")[0]

        if project_key in PROJECT_KEY_TO_REPO_FOLDER:
            return PROJECT_KEY_TO_REPO_FOLDER[project_key]

    raise ValueError(
        f"Cannot determine repository folder for row with "
        f"Key={row.get('Key')} and GitHub Base URL={github_url}"
    )


def get_clone_url_from_repo_folder(repo_folder: str) -> str:
    if repo_folder not in ALLOWED_REPO_FOLDERS:
        raise ValueError(
            f"Repository folder is not in the allowed repositories: {repo_folder}"
        )

    return f"https://github.com/apache/{repo_folder}.git"


def clone_repository_if_missing(repo_dir: Path, clone_url: str) -> bool:
    repo_dir = Path(repo_dir)

    if repo_dir.exists() and (repo_dir / ".git").exists():
        return True

    if repo_dir.exists() and not (repo_dir / ".git").exists():
        print(f"[WARNING] Folder exists but is not a Git repository: {repo_dir}")
        return False

    if not AUTO_CLONE_REPOSITORY:
        print(f"[SKIP] Repository not found and AUTO_CLONE_REPOSITORY=False: {repo_dir}")
        return False

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Repository Not Found. Cloning Repository")
    print("============================================================")
    print(f"Clone URL : {clone_url}")
    print(f"Target    : {repo_dir}")
    print("============================================================")

    result = subprocess.run(
        ["git", "clone", clone_url, str(repo_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] Failed to clone repository: {clone_url}")
        print(result.stderr)
        return False

    print(f"[OK] Repository cloned successfully: {repo_dir}")

    return True


def prepare_repository_for_row(
    row: pd.Series,
    intro_hash: str,
    payment_hash: str
) -> Dict[str, Any]:
    project_key = get_project_key_from_row(row)

    default_repo_folder = get_repo_folder_from_row(row)
    default_clone_url = get_clone_url_from_repo_folder(default_repo_folder)

    if project_key in PROJECT_REPO_CANDIDATES:
        candidates = PROJECT_REPO_CANDIDATES[project_key].copy()

        default_candidate = {
            "repo_folder": default_repo_folder,
            "clone_url": default_clone_url,
        }

        if default_candidate not in candidates:
            candidates.insert(0, default_candidate)
    else:
        candidates = [
            {
                "repo_folder": default_repo_folder,
                "clone_url": default_clone_url,
            }
        ]

    tried = []

    for candidate in candidates:
        repo_folder = candidate["repo_folder"]
        clone_url = candidate["clone_url"]

        if repo_folder not in ALLOWED_REPO_FOLDERS:
            tried.append({
                "project_key": project_key,
                "repo_folder": repo_folder,
                "clone_url": clone_url,
                "repo_dir": None,
                "repo_ready": False,
                "intro_commit_found": False,
                "payment_commit_found": False,
                "status": "repo_not_allowed",
            })
            continue

        repo_dir = Path(REPOS_ROOT) / repo_folder

        repo_ready = clone_repository_if_missing(
            repo_dir=repo_dir,
            clone_url=clone_url
        )

        if not repo_ready:
            tried.append({
                "project_key": project_key,
                "repo_folder": repo_folder,
                "clone_url": clone_url,
                "repo_dir": str(repo_dir),
                "repo_ready": False,
                "intro_commit_found": False,
                "payment_commit_found": False,
                "status": "repository_unavailable",
            })
            continue

        intro_exists = commit_exists(repo_dir, intro_hash)
        payment_exists = commit_exists(repo_dir, payment_hash)

        tried.append({
            "project_key": project_key,
            "repo_folder": repo_folder,
            "clone_url": clone_url,
            "repo_dir": str(repo_dir),
            "repo_ready": True,
            "intro_commit_found": intro_exists,
            "payment_commit_found": payment_exists,
            "status": (
                "matched_intro_and_payment"
                if intro_exists and payment_exists
                else "commit_not_found_in_this_candidate"
            ),
        })

        if intro_exists and payment_exists:
            return {
                "repo_ready": True,
                "project_key": project_key,
                "repo_folder": repo_folder,
                "repo_dir": repo_dir,
                "clone_url": clone_url,
                "tried_repositories": tried,
                "repo_selection_status": "matched_intro_and_payment",
                "intro_commit_found": True,
                "payment_commit_found": True,
            }

        print(
            f"[REPO-MISS] {project_key} | repo={repo_folder} | "
            f"intro_found={intro_exists} | payment_found={payment_exists}"
        )

    return {
        "repo_ready": False,
        "project_key": project_key,
        "repo_folder": default_repo_folder,
        "repo_dir": Path(REPOS_ROOT) / default_repo_folder,
        "clone_url": default_clone_url,
        "tried_repositories": tried,
        "repo_selection_status": "commit_not_found_in_expected_or_fallback_repository",
        "intro_commit_found": False,
        "payment_commit_found": False,
    }


def get_self_fixed_group(row: pd.Series) -> Tuple[bool, str]:
    value = str(row.get("Is Self-Fixed (Intro=Payment)", "")).strip().lower()

    if value not in ("true", "1", "yes", "y", "false", "0", "no", "n"):
        raise ValueError(f"Unknown SF/NSF label: {value!r}; do not default to NSF")
    is_self_fixed = value in ["true", "1", "yes", "y"]

    group = "Self-Fixed" if is_self_fixed else "Non-Self-Fixed"

    return is_self_fixed, group


# ============================================================
# Metric Parsing
# ============================================================

def build_metric_map(json_value: Any) -> Dict[str, Dict[str, float]]:
    """Read nine file-level metrics without imputing missing values.

    Candidate JSON aliases must be verified against actual file exports.
    Values here must already be aggregated at the *file level*; do not
    silently substitute class-level metrics without an explicit rule.
    """
    details = safe_json_loads(json_value, default=[])
    if not isinstance(details, list):
        return {}
    metric_map: Dict[str, Dict[str, float]] = {}
    for item in details:
        if not isinstance(item, dict):
            continue
        path = normalize_path(
            item.get("file_path") or item.get("understand_file_relname")
            or item.get("git_numstat_path") or item.get("path")
            or item.get("file") or item.get("name")
        )
        if not is_java_file(path):
            continue
        values: Dict[str, float] = {}
        for metric, aliases in METRIC_JSON_KEYS.items():
            value = np.nan
            for key in aliases:
                if key in item:
                    candidate = safe_float(item[key])
                    if np.isfinite(candidate):
                        value = candidate
                        break
            values[metric] = value
        if path in metric_map:
            raise ValueError(f"Duplicate JSON entry for file {path}; do not silently overwrite")
        metric_map[path] = values
    return metric_map


def parse_payment_path_map(value: Any) -> Dict[str, Optional[str]]:
    data = safe_json_loads(value, default={})

    if not isinstance(data, dict):
        return {}

    result = {}

    for k, v in data.items():
        key = normalize_path(k)
        val = normalize_path(v) if v else None

        if key:
            result[key] = val

    return result


def get_payment_metric_for_intro_file_fast(
    intro_file: str,
    final_path: Optional[str],
    payment_metric_map: Dict[str, Dict[str, float]],
    payment_path_map: Dict[str, Optional[str]],
) -> Tuple[Optional[str], Dict[str, float]]:
    missing = {metric: np.nan for metric in METRIC_JSON_KEYS}
    # Do NOT fall back to original path when the file was deleted.
    # A same-named new file after deletion is not necessarily a matched file.
    if not final_path:
        return None, missing
    candidates = [final_path, payment_path_map.get(intro_file)]
    if final_path == intro_file:
        candidates.append(intro_file)
    for path in dict.fromkeys(normalize_path(p) for p in candidates if p):
        if path in payment_metric_map:
            return path, payment_metric_map[path]
    return None, missing


# ============================================================
# RQ3 File-Level Dataset Construction
# ============================================================

def build_file_level_rq3_dataset(
    input_dir: str,
    csv_pattern: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_all_input_csvs(input_dir, csv_pattern)

    required_columns = [
        "Key",
        "Intro Hash",
        "Payment Hash",
        "Intro_Affected_Files",
        "Is Self-Fixed (Intro=Payment)",
        "Intro_FileDetails_JSON",
        "Payment_FileDetails_JSON",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if FILTER_INDICATOR is not None:
        if "indicator" not in df.columns:
            raise ValueError(
                "FILTER_INDICATOR is enabled, but column 'indicator' is missing."
            )

        before_filter = len(df)

        df = df[
            df["indicator"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(str(FILTER_INDICATOR).lower())
        ].copy()

        after_filter = len(df)

        print("============================================================")
        print("Indicator Filter")
        print("============================================================")
        print(f"Filter indicator : {FILTER_INDICATOR}")
        print(f"Rows before      : {before_filter}")
        print(f"Rows after       : {after_filter}")
        print("============================================================")

    # One row per issue is expected in these input exports.
    duplicate = df.duplicated(["Key"], keep=False)
    if duplicate.any():
        raise ValueError(
            f"Input includes {duplicate.sum()} rows with duplicate issue Keys. "
            "Resolve repeated issue exports before file-level analysis."
        )
    if PRIMARY_COHORT_CSV is not None:
        cohort = pd.read_csv(PRIMARY_COHORT_CSV, dtype={"project": str, "Key": str})
        required = {"project", "Key"}
        if not required.issubset(cohort.columns):
            raise ValueError(f"PRIMARY_COHORT_CSV must include {required}")
        allowed = set(zip(cohort.project.str.upper(), cohort.Key))
        before = len(df)
        df = df.loc[df.apply(
            lambda row: (get_project_key_from_row(row).upper(), str(row.Key)) in allowed,
            axis=1,
        )].copy()
        print(f"[COHORT] Kept {len(df)}/{before} ATD items in primary cohort")
    # PRE-FLIGHT: CK comes from a different folder and a different JSON schema.
    # Do not require CK fields in the original Intro_FileDetails_JSON.
    matched_keys = set(df["Key"]) & set(CK_INDEX)
    print(f"[CK PREFLIGHT] Issues in FINAL: {len(df)}; issues present in CK: {len(matched_keys)}")
    unmatched_issues = sorted(set(df["Key"]) - matched_keys)
    if unmatched_issues:
        print(f"[CK WARNING] {len(unmatched_issues)} issues have no matching Key in CK, e.g. {unmatched_issues[:8]}")
    if not matched_keys:
        raise ValueError("No issue Keys overlap FINAL and CK input CSVs. Check --ck-dir.")
    absent_ck_projects = sorted(set(df["Key"].str.split("-").str[0]) -
                                set(str(k).split("-")[0] for k in CK_INDEX))
    if absent_ck_projects:
        raise ValueError("CK-METRICS CSVs missing for project(s): " + ", ".join(absent_ck_projects) + ". Stop before Git tracking.")

    file_level_rows = []
    repo_selection_log_rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="RQ3 fast file-level tracking"
    ):
        key = row["Key"]
        intro_hash = str(row["Intro Hash"]).strip()
        payment_hash = str(row["Payment Hash"]).strip()

        try:
            repo_info = prepare_repository_for_row(
                row=row,
                intro_hash=intro_hash,
                payment_hash=payment_hash
            )
        except Exception as e:
            repo_info = {
                "repo_ready": False,
                "project_key": get_project_key_from_row(row),
                "repo_folder": None,
                "repo_dir": None,
                "clone_url": None,
                "tried_repositories": [],
                "repo_selection_status": f"repository_selection_error: {e}",
                "intro_commit_found": False,
                "payment_commit_found": False,
            }

        repo_selection_log_rows.append({
            "Key": key,
            "source_csv_file": row.get("source_csv_file", np.nan),
            "project_key": repo_info.get("project_key"),
            "repo_ready": repo_info.get("repo_ready"),
            "repo_folder": repo_info.get("repo_folder"),
            "repo_dir": str(repo_info.get("repo_dir")) if repo_info.get("repo_dir") else None,
            "clone_url": repo_info.get("clone_url"),
            "repo_selection_status": repo_info.get("repo_selection_status"),
            "intro_hash": intro_hash,
            "payment_hash": payment_hash,
            "intro_commit_found": repo_info.get("intro_commit_found"),
            "payment_commit_found": repo_info.get("payment_commit_found"),
            "tried_repositories_json": json.dumps(
                repo_info.get("tried_repositories", []),
                ensure_ascii=False
            ),
        })

        if not repo_info["repo_ready"]:
            print(
                f"[SKIP] Repository/commit unavailable for {key} | "
                f"repo_status={repo_info.get('repo_selection_status')}"
            )
            continue

        repo_folder = repo_info["repo_folder"]
        repo_dir = repo_info["repo_dir"]
        clone_url = repo_info["clone_url"]
        repo_selection_status = repo_info["repo_selection_status"]

        java_files = parse_intro_affected_files(row["Intro_Affected_Files"])

        if not java_files:
            print(f"[SKIP] No Java files in Intro_Affected_Files for {key}")
            continue

        intro_metric_map = build_metric_map(row["Intro_FileDetails_JSON"])
        payment_metric_map = build_metric_map(row["Payment_FileDetails_JSON"])

        payment_path_map = {}

        if "Payment_Path_Map_JSON" in df.columns:
            payment_path_map = parse_payment_path_map(
                row.get("Payment_Path_Map_JSON")
            )

        is_self_fixed, group = get_self_fixed_group(row)

        # A..B does not represent the ATD lifecycle unless A is an ancestor of B.
        ancestry_check = subprocess.run(
            ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor",
             intro_hash, payment_hash],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if ancestry_check.returncode != 0:
            print(f"[SKIP] Introduction is not an ancestor of repayment for {key}")
            continue

        tracking_by_file, commit_range = fast_track_files_for_issue(
            repo_dir=repo_dir,
            intro_hash=intro_hash,
            payment_hash=payment_hash,
            intro_files=java_files,
            include_intro=True
        )

        for intro_file in java_files:
            tracking_result = tracking_by_file.get(intro_file)

            if tracking_result is None:
                continue

            intro_metrics = intro_metric_map.get(
                intro_file, {metric: np.nan for metric in METRIC_JSON_KEYS}
            )
            final_path = tracking_result.get("final_path")
            payment_metric_path, payment_metrics = get_payment_metric_for_intro_file_fast(
                intro_file=intro_file,
                final_path=final_path,
                payment_metric_map=payment_metric_map,
                payment_path_map=payment_path_map,
            )
            touched_commits = tracking_result.get("touched_commits", [])
            record = {
                "source_csv_file": row.get("source_csv_file", np.nan),
                "Key": key,
                "label": row.get("label", np.nan),
                "indicator": row.get("indicator", np.nan),
                "group": group,
                "is_self_fixed": is_self_fixed,
                "project_key": repo_info.get("project_key"),
                "repo_folder": repo_folder,
                "clone_url": clone_url,
                "repo_selection_status": repo_selection_status,
                "intro_hash": intro_hash,
                "payment_hash": payment_hash,
                "commit_range_used": commit_range,
                "intro_file_path": intro_file,
                "exists_at_intro": tracking_result.get("exists_at_intro"),
                "payment_metric_path": payment_metric_path,
                "final_tracked_path": final_path,
                "tracking_status": tracking_result.get("tracking_status"),
                "deleted": tracking_result.get("deleted"),
                "rename_count": tracking_result.get("rename_count"),
                "path_history": ";".join(tracking_result.get("path_history", [])),
                "file_commit_count_intro_to_payment": tracking_result.get("commit_count", 0),
                "touched_commits": ";".join(touched_commits),
            }
            for metric in METRIC_JSON_KEYS:
                before = safe_float(intro_metrics.get(metric, np.nan))
                after = safe_float(payment_metrics.get(metric, np.nan))
                # Same units, same aggregation, matched file at both phases.
                if not tracking_result.get("exists_at_intro", False):
                    before = np.nan
                    after = np.nan
                record[f"intro_{metric}"] = before
                record[f"payment_{metric}"] = after
                record[f"delta_{metric}"] = delta_value(after, before)
            file_level_rows.append(record)

    file_df = pd.DataFrame(file_level_rows)
    repo_selection_log_df = pd.DataFrame(repo_selection_log_rows)

    return file_df, repo_selection_log_df


# ============================================================
# RQ3 Statistical Analysis with Effect Size
# ============================================================

def interpret_correlation_effect_size(rho: float) -> str:
    if pd.isna(rho):
        return "NA"

    abs_rho = abs(rho)

    if abs_rho < 0.10:
        return "negligible"
    elif abs_rho < 0.30:
        return "weak"
    elif abs_rho < 0.50:
        return "moderate"
    else:
        return "strong"


def interpret_correlation_direction(rho: float) -> str:
    if pd.isna(rho):
        return "NA"

    if rho > 0:
        return "positive"
    elif rho < 0:
        return "negative"
    else:
        return "zero"


def spearman_analysis(file_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 18 main SF/NSF tests and 9 separately corrected pooled tests.

    IMPORTANT: Spearman's ordinary p-values assume independent rows. Multiple
    files per issue and reused files across issues violate that assumption;
    regard p-values as exploratory pending a cluster-aware sensitivity test.
    """
    x_col = "file_commit_count_intro_to_payment"
    metrics = [f"delta_{metric}" for metric in METRIC_JSON_KEYS]
    rows = []
    for group in (*RQ3_PRIMARY_GROUPS, "All"):
        subset = file_df if group == "All" else file_df[file_df.group == group]
        for metric in metrics:
            sample = subset[[x_col, metric, "Key", "project_key"]].copy()
            sample[x_col] = pd.to_numeric(sample[x_col], errors="coerce")
            sample[metric] = pd.to_numeric(sample[metric], errors="coerce")
            sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, metric])
            n = len(sample)
            rho, pvalue = np.nan, np.nan
            if n < 3:
                note = "Insufficient data"
            elif sample[x_col].nunique() < 2:
                note = "Commit count has no variation"
            elif sample[metric].nunique() < 2:
                note = "Metric delta has no variation"
            else:
                rho, pvalue = spearmanr(sample[x_col], sample[metric])
                note = "OK"
            rows.append({
                "Group": group,
                "Metric_Delta": metric,
                "Metric_Label": METRIC_LABELS[metric.removeprefix("delta_")],
                "N_Files": n,
                "N_Issues": sample["Key"].nunique(),
                "N_Projects": sample["project_key"].nunique(),
                "Spearman_Rho": rho,
                "Effect_Size_Abs_Rho": abs(rho) if pd.notna(rho) else np.nan,
                "Effect_Size_Category": interpret_correlation_effect_size(rho),
                "Correlation_Direction": interpret_correlation_direction(rho),
                "P_Value": pvalue,
                "Note": note,
                "Test_Family": "RQ3_primary_SF_NSF_18" if group != "All" else "RQ3_exploratory_pooled_9",
            })
    result = pd.DataFrame(rows)
    result["P_Value_Holm"] = np.nan
    result["Significant_Holm_0.05"] = False
    result["Family_Planned_Size"] = 0
    result["Family_Tested_Size"] = 0
    for family, planned in (("RQ3_primary_SF_NSF_18", 18),
                            ("RQ3_exploratory_pooled_9", 9)):
        mask = result.Test_Family.eq(family)
        valid = mask & result.P_Value.notna()
        result.loc[mask, "Family_Planned_Size"] = planned
        result.loc[mask, "Family_Tested_Size"] = int(valid.sum())
        # An untestable planned hypothesis consumes a slot in the correction
        # (p=1), rather than silently reducing the family size.
        candidate_p = result.loc[mask, "P_Value"].fillna(1.0).to_numpy()
        reject, adjusted, _, _ = multipletests(candidate_p, alpha=0.05, method="holm")
        result.loc[valid, "P_Value_Holm"] = adjusted[valid[mask].to_numpy()]
        result.loc[valid, "Significant_Holm_0.05"] = reject[valid[mask].to_numpy()]
    result["Inference_Warning"] = (
        "File-level p-values do not account for clustering within issues/projects; "
        "validate with cluster-aware sensitivity analysis"
    )
    return result


def descriptive_summary(file_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["file_commit_count_intro_to_payment"] + [
        f"delta_{name}" for name in METRIC_JSON_KEYS
    ]
    rows = []
    for group in (*RQ3_PRIMARY_GROUPS, "All"):
        part = file_df if group == "All" else file_df[file_df.group == group]
        for col in columns:
            vals = pd.to_numeric(part[col], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            q1, q3 = (vals.quantile(.25), vals.quantile(.75)) if len(vals) else (np.nan, np.nan)
            rows.append({
                "Group": group, "Variable": col, "N": len(vals),
                "Mean": vals.mean() if len(vals) else np.nan,
                "Median": vals.median() if len(vals) else np.nan,
                "Q1": q1, "Q3": q3, "IQR": q3 - q1,
                "Min": vals.min() if len(vals) else np.nan,
                "Max": vals.max() if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def issue_level_sensitivity(file_df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity: one median matched-file pair per ATD issue and metric.

    Compute the median commit count from the SAME set of matched files used to
    calculate that issue's median delta for the metric. Avoid aggregating X
    over files that have no corresponding Y data.
    """
    x_col = "file_commit_count_intro_to_payment"
    results = []
    for group in (*RQ3_PRIMARY_GROUPS, "All"):
        source = file_df if group == "All" else file_df[file_df.group == group]
        for metric in METRIC_JSON_KEYS:
            delta = f"delta_{metric}"
            subset = source[["project_key", "Key", x_col, delta]].copy()
            subset[[x_col, delta]] = subset[[x_col, delta]].apply(
                pd.to_numeric, errors="coerce")
            subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, delta])
            per_issue = (
                subset.groupby(["project_key", "Key"], dropna=False)[[x_col, delta]]
                .median().reset_index()
            )
            n = len(per_issue)
            rho, p = np.nan, np.nan
            if n < 3:
                note = "Insufficient issues"
            elif per_issue[x_col].nunique() < 2:
                note = "Issue-median commit count has no variation"
            elif per_issue[delta].nunique() < 2:
                note = "Issue-median metric delta has no variation"
            else:
                rho, p = spearmanr(per_issue[x_col], per_issue[delta])
                note = "OK"
            results.append({
                "Group": group, "Metric_Delta": delta, "Metric_Label": METRIC_LABELS[metric],
                "N_Issues": n, "N_Files_Before_Aggregation": len(subset),
                "N_Projects": per_issue.project_key.nunique(),
                "Spearman_Rho": rho, "P_Value": p, "Note": note,
                "Test_Family": "RQ3_primary_SF_NSF_18" if group != "All"
                               else "RQ3_exploratory_pooled_9",
            })
    result = pd.DataFrame(results)
    result.insert(0, "Unit_Of_Analysis", "Issue-median among matched files")
    result["P_Value_Holm"] = np.nan
    result["Significant_Holm_0.05"] = False
    for family in result.Test_Family.unique():
        mask = result.Test_Family.eq(family)
        pvals = result.loc[mask, "P_Value"].fillna(1.).to_numpy()
        reject, p_adj, _, _ = multipletests(pvals, method="holm", alpha=.05)
        valid = result.loc[mask, "P_Value"].notna().to_numpy()
        result.loc[mask, "P_Value_Holm"] = np.where(valid, p_adj, np.nan)
        result.loc[mask, "Significant_Holm_0.05"] = reject & valid
    result["Inference_Warning"] = (
        "Issue aggregation reduces within-issue dependence but does not "
        "adjust for project-level clustering or recurring files."
    )
    return result


def metric_coverage(file_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_JSON_KEYS:
        delta = f"delta_{metric}"
        for group in RQ3_PRIMARY_GROUPS:
            subset = file_df[file_df.group.eq(group)]
            valid = pd.to_numeric(subset[delta], errors="coerce").notna()
            rows.append({
                "Metric": metric, "Group": group,
                "Files": len(subset), "Paired_Files": int(valid.sum()),
                "Paired_Percent": 100. * valid.mean() if len(valid) else np.nan,
                "Issues_With_Paired_Files": subset.loc[valid, "Key"].nunique(),
            })
    return pd.DataFrame(rows)


def check_metric_coverage(file_df: pd.DataFrame) -> pd.DataFrame:
    coverage = metric_coverage(file_df)
    coverage.to_csv(OUTPUT_METRIC_COVERAGE, index=False)
    absent = coverage.groupby("Metric")["Paired_Files"].sum()
    absent = absent[absent.eq(0)].index.tolist()
    if absent:
        message = (
            "No matched file-level values for metrics: " + ", ".join(absent)
            + ". Inspect Intro_FileDetails_JSON / Payment_FileDetails_JSON "
              "for actual metric key names and verify class-to-file aggregation. "
              "See rq3_metric_coverage.csv."
        )
        if STRICT_NINE_METRICS:
            raise ValueError(message)
        print("[WARNING] " + message)
    return coverage


# ============================================================
# Scatter Plot: Dependency and Complexity Figures
# ============================================================

def plot_binned_medians(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """Exploratory summaries, NOT Spearman fitted lines or significance tests."""
    if len(x) < 25 or len(np.unique(x)) < 3:
        return
    work = pd.DataFrame({"x": x, "y": y})
    try:
        bins = pd.qcut(work["x"], q=min(8, len(np.unique(x))), duplicates="drop")
        medians = work.groupby(bins, observed=True).agg(x=("x", "median"), y=("y", "median"))
        if len(medians) > 1:
            ax.plot(medians.x, medians.y, color=color, linewidth=1.5,
                    alpha=.9, marker="o", markersize=2)
    except (ValueError, TypeError):
        return


def create_scatter_plot_panel(
    file_df: pd.DataFrame,
    plot_specs: List[Tuple[str, str]],
    output_png: str,
    output_pdf: str,
    figure_title: Optional[str] = None,
    ncols: int = 2,
) -> None:
    x_col = "file_commit_count_intro_to_payment"
    required = [x_col, "group"] + [col for col, _ in plot_specs]
    missing = [col for col in required if col not in file_df]
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")
    nrows = math.ceil(len(plot_specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.15),
                             sharex=False, sharey=False, squeeze=False,
                             constrained_layout=True)
    handles = {}
    for ax, (metric, label) in zip(axes.flat, plot_specs):
        ax.axhline(0, color=".55", linestyle="--", linewidth=.7)
        for group in RQ3_PRIMARY_GROUPS:
            part = file_df[file_df.group == group]
            temp = part[[x_col, metric]].apply(pd.to_numeric, errors="coerce")
            temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
            if temp.empty:
                continue
            x, y = temp[x_col].to_numpy(), temp[metric].to_numpy()
            marker = ax.scatter(x, y, s=9, alpha=.19, linewidths=0,
                                color=GROUP_COLORS[group], label=GROUP_PLOT_LABELS[group])
            handles.setdefault(group, marker)
            plot_binned_medians(ax, x, y, GROUP_COLORS[group])
        ax.text(.02, .97, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold", bbox={"facecolor":"white",
                "alpha":.85, "edgecolor":"none", "pad":2})
        ax.set_xlabel("File commit count")
        ax.set_ylabel("Change (repayment − introduction)")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(plot_specs):]:
        ax.set_axis_off()
    if figure_title:
        fig.suptitle(figure_title, fontsize=11)
    if handles:
        fig.legend([handles[g] for g in RQ3_PRIMARY_GROUPS if g in handles],
                   [GROUP_PLOT_LABELS[g] for g in RQ3_PRIMARY_GROUPS if g in handles],
                   loc="outside lower center", ncol=2, fontsize=9)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_pdf)


def create_dependency_and_complexity_scatter_plots(file_df: pd.DataFrame) -> None:
    categories = [
        ("dependency", ("fan_in", "fan_out")),
        ("coupling_cohesion", ("cbo", "rfc_proxy", "lcom")),
        ("inheritance", ("dit", "noc")),
        ("complexity", ("cyclomatic_complexity", "cognitive_complexity")),
    ]
    for name, metrics in categories:
        create_scatter_plot_panel(
            file_df,
            [(f"delta_{metric}", METRIC_LABELS[metric]) for metric in metrics],
            os.path.join(OUTPUT_DIR, f"rq3_{name}_scatter.png"),
            os.path.join(OUTPUT_DIR, f"rq3_{name}_scatter.pdf"),
        )
    create_scatter_plot_panel(
        file_df,
        [(f"delta_{m}", METRIC_LABELS[m]) for m in METRIC_JSON_KEYS],
        OUTPUT_ALL_SCATTER_PNG, OUTPUT_ALL_SCATTER_PDF, ncols=3,
    )


# ============================================================
# Sanity Checks
# ============================================================

def print_sanity_checks(
    file_df: pd.DataFrame,
    repo_selection_log_df: pd.DataFrame
) -> None:
    print("\n============================================================")
    print("Sanity Checks")
    print("============================================================")

    print(f"Total file-level rows: {len(file_df)}")
    print(f"Total repository-selection rows: {len(repo_selection_log_df)}")

    if not repo_selection_log_df.empty:
        print("\nRepository selection status:")
        print(
            repo_selection_log_df["repo_selection_status"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nSelected repository distribution:")
        print(
            repo_selection_log_df["repo_folder"]
            .value_counts(dropna=False)
            .to_string()
        )

        failed_repo_df = repo_selection_log_df[
            repo_selection_log_df["repo_ready"] != True
        ]

        if not failed_repo_df.empty:
            print("\nRepository selection failures:")
            print(
                failed_repo_df[
                    [
                        "Key",
                        "project_key",
                        "repo_folder",
                        "intro_hash",
                        "payment_hash",
                        "intro_commit_found",
                        "payment_commit_found",
                        "repo_selection_status"
                    ]
                ].head(50).to_string(index=False)
            )

    if file_df.empty:
        print("\nNo file-level rows generated.")
        print("============================================================\n")
        return

    if "source_csv_file" in file_df.columns:
        print("\nRows by source CSV:")
        print(file_df["source_csv_file"].value_counts(dropna=False).to_string())

    if "indicator" in file_df.columns:
        print("\nIndicator distribution:")
        print(file_df["indicator"].value_counts(dropna=False).to_string())

    if "group" in file_df.columns:
        print("\nGroup distribution:")
        print(file_df["group"].value_counts(dropna=False).to_string())

    if "repo_folder" in file_df.columns:
        print("\nRepository distribution in file-level rows:")
        print(file_df["repo_folder"].value_counts(dropna=False).to_string())

    if "tracking_status" in file_df.columns:
        print("\nTracking status distribution:")
        print(file_df["tracking_status"].value_counts(dropna=False).to_string())

    if "deleted" in file_df.columns:
        print("\nDeleted distribution:")
        print(file_df["deleted"].value_counts(dropna=False).to_string())

    if "exists_at_intro" in file_df.columns:
        print("\nExists at intro distribution:")
        print(file_df["exists_at_intro"].value_counts(dropna=False).to_string())

    if "rename_count" in file_df.columns:
        print("\nRename count summary:")
        print(file_df["rename_count"].describe().to_string())

    if "file_commit_count_intro_to_payment" in file_df.columns:
        print("\nFile commit count summary:")
        print(
            file_df["file_commit_count_intro_to_payment"]
            .describe()
            .to_string()
        )

    metric_cols = [f"delta_{name}" for name in METRIC_JSON_KEYS]

    existing_metric_cols = [c for c in metric_cols if c in file_df.columns]

    if existing_metric_cols:
        print("\nMissing values in metric deltas:")
        print(file_df[existing_metric_cols].isna().sum().to_string())

    print("============================================================\n")


# ============================================================
# CK file-level matching: same observational unit as FINAL/Git tracking
# ============================================================

def _bool_flag(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _file_ck_map(json_string, issue_key, phase):
    """Map ONE file to FIVE within-file class-mean CK metrics."""
    data = safe_json_loads(json_string)
    if not isinstance(data, list):
        return {}
    result = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        path = normalize_path(item.get("file_path") or item.get("understand_file_relname"))
        if not is_java_file(path):
            continue
        if path in result:
            raise ValueError(f"Duplicate CK file {issue_key}/{phase}/{path}")
        result[path] = {name: safe_float(item.get(field))
                        for name, field in CK_FILE_METRIC_KEYS.items()}
    return result


def load_ck_index(ck_directory, ck_pattern=CK_PATTERN):
    ck_directory = Path(ck_directory)
    files = sorted(ck_directory.glob(ck_pattern))
    if not files:
        raise FileNotFoundError(f"No CK CSV files in {ck_directory} matching {ck_pattern}")
    required = {"Key", "Intro Hash", "Payment Hash", "Intro_CK_FileSummary_JSON",
                "Payment_CK_FileSummary_JSON", "Intro_CK_Status", "Payment_CK_Status"}
    index = {}
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = required - set(header)
        if missing:
            raise ValueError(f"{path.name}: required CK columns missing: {sorted(missing)}")
        # Avoid loading the large ClassDetails_JSON and IssueSummary_JSON columns.
        selected = list(required | ({"Payment_Path_Map_JSON"} & set(header)))
        frame = pd.read_csv(path, usecols=selected, dtype=str, low_memory=False)
        print(f"[CK] {path.name}: {len(frame)} issues")
        for record in frame.to_dict('records'):
            key = str(record['Key']).strip()
            if not key or key.lower() == 'nan':
                raise ValueError(f"Invalid Key in {path}")
            if key in index:
                raise ValueError(f"Duplicate CK issue Key {key} in CK files; cannot safely merge")
            index[key] = record
    print(f"[CK] Loaded {len(index)} distinct issue keys from {len(files)} CK files")
    return index


def _ck_row_match_status(row, ck):
    if ck is None:
        return "ck_issue_missing"
    intro_hash = str(row.get("intro_hash", "")).strip()
    payment_hash = str(row.get("payment_hash", "")).strip()
    if not intro_hash or not payment_hash or intro_hash.lower() == 'nan' or payment_hash.lower() == 'nan':
        return "tracking_commit_hash_missing"
    if intro_hash != str(ck['Intro Hash']).strip() or payment_hash != str(ck['Payment Hash']).strip():
        return "commit_hash_mismatch"
    if str(ck['Intro_CK_Status']).strip().lower() != 'success':
        return "intro_ck_not_success"
    if str(ck['Payment_CK_Status']).strip().lower() != 'success':
        return "payment_ck_not_success"
    return "matched_issue_and_hashes"


def enrich_file_level_with_ck(file_df, ck_index):
    """Apply matched CK file summary to already tracked file rows; NO Git calls.

    One row remains one issue/file pair. Match on Key+hashes+file paths.
    No issue-level or class-level metric values are copied onto file rows.
    """
    df = file_df.copy()
    required = {"Key", "intro_hash", "payment_hash", "intro_file_path",
                "final_tracked_path", "deleted", "exists_at_intro", "group", "project_key"}
    if miss := required - set(df.columns):
        raise ValueError(f"Tracked file CSV missing columns: {sorted(miss)}")
    if df.empty:
        raise ValueError("Tracked file dataset is empty")
    if df.duplicated(["Key", "intro_file_path"]).any():
        raise ValueError("Duplicate issue/file observations: cannot safely attach CK metrics")
    for metric in CK_FILE_METRIC_KEYS:
        for prefix in ("intro_", "payment_", "delta_"):
            df[prefix + metric] = np.nan
    df["ck_file_match_status"] = "not_processed"
    df["ck_intro_metric_path"] = None
    df["ck_payment_metric_path"] = None
    # Most CK issues contain numerous files: parse each JSON ONCE per issue.
    for key, indices in df.groupby("Key", sort=False).groups.items():
        ck = ck_index.get(str(key))
        first = df.loc[next(iter(indices))]
        status = _ck_row_match_status(first, ck)
        if status != "matched_issue_and_hashes":
            df.loc[indices, "ck_file_match_status"] = status
            continue
        im = _file_ck_map(ck.get("Intro_CK_FileSummary_JSON"), key, "intro")
        pm = _file_ck_map(ck.get("Payment_CK_FileSummary_JSON"), key, "payment")
        ck_payment_map = parse_payment_path_map(ck.get("Payment_Path_Map_JSON"))
        for ix in indices:
            rec = df.loc[ix]
            if not _bool_flag(rec["exists_at_intro"]):
                df.at[ix, "ck_file_match_status"] = "not_present_at_intro"
                continue
            if _bool_flag(rec["deleted"]):
                df.at[ix, "ck_file_match_status"] = "deleted_before_payment"
                continue
            intro_path = normalize_path(rec['intro_file_path'])
            final_path = normalize_path(rec['final_tracked_path'])
            if not final_path:
                df.at[ix, "ck_file_match_status"] = "final_path_missing"
                continue
            a = im.get(intro_path)
            if a is None:
                df.at[ix, "ck_file_match_status"] = "intro_file_missing_in_ck"
                continue
            # The trace-determined final path takes priority, followed by the
            # exported payment path map. Original name only if unchanged.
            options = [final_path, ck_payment_map.get(intro_path)]
            if final_path == intro_path:
                options.append(intro_path)
            pay_path = next((p for candidate in options
                             if (p := normalize_path(candidate)) and p in pm), None)
            b = pm.get(pay_path) if pay_path else None
            if b is None:
                df.at[ix, "ck_file_match_status"] = "payment_file_missing_in_ck"
                continue
            df.at[ix, "ck_intro_metric_path"] = intro_path
            df.at[ix, "ck_payment_metric_path"] = pay_path
            valid_count = 0
            for metric in CK_FILE_METRIC_KEYS:
                aa, bb = safe_float(a.get(metric)), safe_float(b.get(metric))
                df.at[ix, 'intro_' + metric] = aa
                df.at[ix, 'payment_' + metric] = bb
                df.at[ix, 'delta_' + metric] = delta_value(bb, aa)
                valid_count += int(np.isfinite(aa) and np.isfinite(bb))
            df.at[ix, 'ck_file_match_status'] = ("matched_all_five" if valid_count == 5
                                                  else f"matched_{valid_count}_of_five")
    return df

# ============================================================
# CLI: new run or no-Git recovery from previously saved file-level CSV
# ============================================================

def main(argv=None):
    global INPUT_DIR, REPOS_ROOT, OUTPUT_DIR, PRIMARY_COHORT_CSV, CK_INDEX, AUTO_CLONE_REPOSITORY, CSV_PATTERN
    global OUTPUT_FILE_LEVEL_CSV, OUTPUT_SPEARMAN_CSV, OUTPUT_SUMMARY_CSV
    global OUTPUT_REPO_SELECTION_LOG_CSV, OUTPUT_METRIC_COVERAGE
    global OUTPUT_DEPENDENCY_SCATTER_PDF, OUTPUT_DEPENDENCY_SCATTER_PNG
    global OUTPUT_COMPLEXITY_SCATTER_PDF, OUTPUT_COMPLEXITY_SCATTER_PNG
    global OUTPUT_ALL_SCATTER_PDF, OUTPUT_ALL_SCATTER_PNG
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--ck-dir", default=CK_DIR)
    parser.add_argument("--repos-root", default=REPOS_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--cached-git", type=Path, default=None,
                        help="Existing RQ3 file-level CSV with Git-tracked paths and commit counts")
    parser.add_argument("--primary-cohort", type=Path, default=None,
                        help="Optional PRIMARY_boxplot_data.csv for same issue cohort as RQ1/RQ2")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--allow-network", action="store_true", help="Explicitly allow cloning and git fetch (requires internet)")
    parser.add_argument("--csv-pattern", default=CSV_PATTERN)
    parser.add_argument("--ck-pattern", default=CK_PATTERN)
    args = parser.parse_args(argv)

    INPUT_DIR = str(args.input_dir)
    CSV_PATTERN = args.csv_pattern
    AUTO_CLONE_REPOSITORY = args.allow_network
    REPOS_ROOT = str(args.repos_root)
    OUTPUT_DIR = str(args.output_dir)
    PRIMARY_COHORT_CSV = str(args.primary_cohort) if args.primary_cohort else None
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE_LEVEL_CSV = os.path.join(OUTPUT_DIR, "rq3_file_level_nine_metrics.csv")
    OUTPUT_SPEARMAN_CSV = os.path.join(OUTPUT_DIR, "rq3_spearman_nine_metrics.csv")
    OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "rq3_descriptive_nine_metrics.csv")
    OUTPUT_REPO_SELECTION_LOG_CSV = os.path.join(OUTPUT_DIR, "rq3_repository_selection_log.csv")
    OUTPUT_METRIC_COVERAGE = os.path.join(OUTPUT_DIR, "rq3_metric_coverage.csv")
    OUTPUT_DEPENDENCY_SCATTER_PDF = os.path.join(OUTPUT_DIR,"rq3_dependency_vs_commit_count.pdf")
    OUTPUT_DEPENDENCY_SCATTER_PNG = os.path.join(OUTPUT_DIR,"rq3_dependency_vs_commit_count.png")
    OUTPUT_COMPLEXITY_SCATTER_PDF = os.path.join(OUTPUT_DIR,"rq3_complexity_vs_commit_count.pdf")
    OUTPUT_COMPLEXITY_SCATTER_PNG = os.path.join(OUTPUT_DIR,"rq3_complexity_vs_commit_count.png")
    OUTPUT_ALL_SCATTER_PDF = os.path.join(OUTPUT_DIR,"rq3_all_nine_scatter_3x3.pdf")
    OUTPUT_ALL_SCATTER_PNG = os.path.join(OUTPUT_DIR,"rq3_all_nine_scatter_3x3.png")

    print("[RQ3] FINAL:", INPUT_DIR)
    print("[RQ3] CK source:", args.ck_dir)
    print("[RQ3] Git cache:", args.cached_git or "none (fresh Git tracking)")
    print("[RQ3] CK aggregation:", CK_FILE_METRIC_KEYS)
    CK_INDEX = load_ck_index(args.ck_dir, args.ck_pattern)

    if args.cached_git is not None:
        if not args.cached_git.is_file():
            parser.error(f"Cached Git file CSV does not exist: {args.cached_git}. "
                         "If the earlier run failed before saving, rerun without --cached-git.")
        df = pd.read_csv(args.cached_git, low_memory=False)
        print(f"[CACHE] Loaded {len(df)} file rows; no Git commands will run")
        if args.primary_cohort:
            cohort = pd.read_csv(args.primary_cohort, usecols=['Key'], dtype=str)
            before = len(df)
            df = df[df.Key.astype(str).isin(set(cohort.Key))].copy()
            print(f"[COHORT] Filtered file rows {before} -> {len(df)}")
        repo_log = pd.DataFrame()
    else:
        df, repo_log = build_file_level_rq3_dataset(INPUT_DIR, CSV_PATTERN)
        if repo_log is not None and len(repo_log):
            repo_log.to_csv(OUTPUT_REPO_SELECTION_LOG_CSV, index=False)
        # Save Git tracking BEFORE any metric coverage validations, so a
        # recoverable Git dataset survives even if CK coverage is incomplete.
        if df.empty:
            raise ValueError("No Java issue/file rows were tracked by Git")
        track_file = Path(OUTPUT_DIR) / "rq3_git_tracking_CACHE.csv"
        df.to_csv(track_file, index=False)
        print("[CACHE] Saved reusable Git tracking to:", track_file)

    required_base = [f'delta_{m}' for m in ('fan_in','fan_out',
                    'cyclomatic_complexity','cognitive_complexity')]
    if missing := set(required_base) - set(df.columns):
        raise ValueError(f"Missing original four metric deltas: {sorted(missing)}")

    df = enrich_file_level_with_ck(df, CK_INDEX)
    # Always save enriched data and audit even if metric coverage fails.
    df.to_csv(OUTPUT_FILE_LEVEL_CSV, index=False)
    (df[["Key", "project_key", "group", "ck_file_match_status"]]
       .drop_duplicates().to_csv(Path(OUTPUT_DIR)/"rq3_ck_issue_matching_audit.csv", index=False))
    (df.groupby("ck_file_match_status", dropna=False).agg(
        File_Rows=("Key", "size"), Issues=("Key", "nunique")
    ).reset_index().to_csv(Path(OUTPUT_DIR)/'rq3_ck_file_matching_audit.csv', index=False))
    print('[CK] File pairing diagnostics:')
    print(df.ck_file_match_status.value_counts(dropna=False).to_string())
    coverage = check_metric_coverage(df)
    print('[RQ3] Nonmissing metric deltas by group:')
    print(coverage.to_string(index=False))

    # Statistical inference is handled by analyze_rq2.py, not by this extraction step.
    # In particular, naive independent-file p-values MUST NOT be presented as confirmatory.
    descriptive_summary(df).to_csv(OUTPUT_SUMMARY_CSV, index=False)
    if not args.skip_figures:
        create_dependency_and_complexity_scatter_plots(df)
    manifest = {
        'original_granularity':'one row per issue x tracked Java file',
        'original_metrics_source':'FINAL Intro_FileDetails_JSON / Payment_FileDetails_JSON',
        'new_ck_metrics_source':'CK-METRICS Intro_CK_FileSummary_JSON / Payment_CK_FileSummary_JSON',
        'ck_file_aggregation':CK_FILE_METRIC_KEYS,
        'ck_issue_join':['Key','Intro Hash','Payment Hash'],
        'ck_file_join':['intro_file_path','final_tracked_path','Payment_Path_Map_JSON'],
        'delta':'payment - introduction',
        'git_tracking_reused': bool(args.cached_git),
        'holm_primary_family_size':18,
        'holm_pooled_exploratory_family_size':9,
        'warning':'Spearman file-row p-values do not account for clustering within issues/projects',
    }
    (Path(OUTPUT_DIR)/'rq3_analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('[OK] Outputs written to', OUTPUT_DIR)

if __name__ == '__main__':
    main()
