# Project Review: AI Tracker Correlator (current main, +5 commits)

**Mode**: Branch review (main vs origin/main merge-base) — filtered to substantive source/config/docs changes.  
**Target**: The 5 commits introducing the modular pipeline refactor, model factory unification, hybrid updater, Phase 2 refinements, and CLI operations improvements.  
**Date of review**: 2026 (context)  
**Scope note**: The raw `git diff` for the 5 commits is ~528k lines / 63 MB because logs, data extracts (.jsonl/.txt), eval outputs, images, and artifacts were included in the commits. Review focused on the ~25 changed Python + config + doc files that represent the engineering work. Full diff was size-gated per review skill guidelines.

## Fixes Applied (post-review cleanup)

All "bug" severity issues from the original review plus several high-impact latent bugs exposed by the modular pipeline refactor have been fixed:

## Suggestions Implemented (ongoing)

In addition to the bugs, the following review suggestions / recommendations have been actioned:

- **Centralized config (Issue 5 + short-term rec)**: Removed duplicate `config = PipelineConfig()` blocks in run_cli.py. All eval path wiring now happens in one place after Pydantic construction. CLI args override the (now authoritative) schema defaults. Added missing `max_age` wiring so `--max-age` actually affects the common track pruning logic. Parser defaults kept for UX/--help but runtime values come from the single config object.
- **More config-driven paths (Issue 6)**: Previous fixes already made pairwise classifiers respect `PipelineConfig.pairwise.*_model_path`. Clutter loading already used its sub-config. Added robustness for missing 't' in the default timestamp mean path.
- **Improve error / silent degradation paths (Issue 8)**: In GNNUpdater, both the early "no model" return and the exception path now increment `age` on all tracks before returning them. This ensures the pipeline's coasting / max-age deletion logic still applies even when the GNN is unavailable or crashes.
- **Logging vs print mix (Issue 13)**: Changed many internal "[OK] Loaded..." and "Warning: ..." messages in `src/updater.py` (GNN + NewHybrid) and `src/pipeline.py` (clutter) from bare `print` to `logging.info` / `logging.warning`. Added the missing `import logging` and `import traceback` (the latter was used in a train error path but not imported at top level).
- **Tests (Issue 12 + medium-term rec)**: 
  - Fixed the existing `tests/test_metrics_robustness.py` (was using uppercase 'MOTA' keys that `compute()` never returns; now uses `.get('mota', ...)` so the test actually passes).
  - Added new `tests/test_pipeline.py` with self-contained synthetic-data tests:
    - `test_kalman_pipeline_basic`: full roundtrip (initiate → promote after min_hits → coast out after max_age misses → reset).
    - Config override propagation.
    - Graceful behavior (no crash) for gnn/hybrid when checkpoints are missing.
    - Direct FallbackUpdater dt / frame_t signature coverage.
  - Tests are runnable via `uv run python -m pytest tests/` (or as scripts). They exercise the exact paths that were previously buggy.
- **Repo hygiene (Issue 11)**: Significantly expanded `.gitignore` with patterns for `logs/`, `*.log`, eval outputs (`*eval*.txt`, `v*_*.txt`, `train_debug*.log`, `anomaly*.txt`, `nan_trace.txt`, `sweden_eval.png`, etc.), more data/ raw dirs, tmp scripts, and common viz binaries. Future commits will be much cleaner; existing history bloat remains a separate (larger) cleanup task.
- **pyproject metadata (Issue 10)**: Replaced the "Add your description here" placeholder with a real description, added MIT license and author (Tom Rathbun) from the README/citation.
- **Other small robustness**: Hardened the default `t` computation in `process_frame` against measurements lacking a numeric 't' key. Added `import traceback` to run_cli.py (prevents NameError in train error paths).

Remaining open suggestions (version dispatch simplification, full deprecation of old v3/v4 paths, deeper ARCHITECTURE.md, more comprehensive property-based association tests, one-time history rewrite or data bucket for the large committed radar extracts) are still worthwhile but higher effort / lower urgency now that the crashers and config drift are addressed.

- **Fixed Issue 1 (bug)**: Added `import numpy as np` to src/pipeline.py. The `t=None` default path in `process_frame` (and any direct callers) no longer crashes with NameError. Verified via smoke test exercising the mean branch.
- **Fixed Issue 2 (bug)**: Removed duplicate `import psutil` in run_cli.py (top-level imports are now clean).
- **Fixed Issue 3 (bug)**: Removed unreachable/dead telemetry code block + inner `import logging`/`traceback` in `GNNUpdater.update` (src/updater.py). Also removed the now-unused `self.frame_count`. Error path now uses `logging.exception(...)` (proper traceback). `frame_count` attr cleaned from __init__.
- **Additional bugs cleaned during pass**:
  - FallbackUpdater (kalman mode) signatures were incompatible with the new unified `Pipeline.process_frame` calls: `update(..., frame_t=...)` and `predict(..., dt=...)` would raise TypeError. Updated defs + made predict respect `dt` (scaling) for consistency with GNN path and time-aware data. Verified with `Pipeline(kalman)` + process_frame calls.
  - `clutter_thresh` / `clutter_threshold` name mismatch: GNN path was doing `getattr(..., 'clutter_threshold', ...)` but CLI + StateUpdaterConfig use `clutter_thresh`. The configured value was always ignored for GNN clutter in manage_tracks. Fixed the key.
  - CLI `--max-age` was only wiring to `state_updater.del_age`; the common track pruning logic in `Pipeline.process_frame` reads from `track_manager.max_age` (which kept schema default). Added the missing `config.track_manager.max_age = args.max_age` so the CLI arg actually controls coasting for all modes.
  - Repeated per-frame `get_model_suite` + dispatch in GNN hot path (importlib overhead + risk of drift): now caches the 5 callables (`_frame_to_tensors` etc.) in `_load_model`. Defensive fallback kept.
  - Hardcoded pairwise classifier paths in both GNNUpdater (non-v6) and NewHybridUpdater: now resolve from `PipelineConfig.pairwise.{psr,ssr}_model_path` with the previous hardcoded strings as fallback. Makes config the source of truth and prevents silent wrong-model bugs.
- All core modules (pipeline, updater, run_cli) + kalman/hybrid paths now import and basic smoke (instantiate + process_frame with the previously crashing arg combos) succeed under `uv run`.
- Pre-existing test issue in tests/test_metrics_robustness.py (expects uppercase 'MOTA' key but compute() returns lowercase 'mota' etc.) was not touched by these edits; metrics module loads and basic usage works.
- No new functionality added beyond hardening the recent refactor; changes are minimal, targeted, and focused on correctness + consistency.

The working tree now contains these fixes (use `git diff` to inspect). Recommend committing as a "fix bugs from modular pipeline refactor" change. Further suggestions (tests, hygiene, version dispatch simplification) remain in the Issues section below for follow-up.

## Summary

The last 5 commits represent a significant and positive step toward a **modular, version-aware research platform** for the GNN+KF hybrid radar tracker. Key achievements: introduction of `Pipeline` + Pydantic `PipelineConfig`, a `factory.get_model_suite` + `detect_model_version` mechanism to support evolving model architectures (v3–v6) without code forks, a `NewHybridUpdater` that ports high-performing logic (spatial clustering + learned pairwise association + asynchronous Kalman), Phase 2 training refinements, and operational CLI improvements (rotating logs, process management, --kill/--list fleet commands).

The direction is correct for a project that had accumulated many one-off training/eval scripts. However, the changes also surface (and in some cases introduce) correctness, maintainability, and hygiene problems. The most immediate issues are a **missing import that will crash the main path**, duplicate imports, unreachable code, inconsistent defaults between CLI/ config/ code, and continued repo bloat from committing artifacts.

Overall assessment: Promising refactoring in flight, but the current head is **not yet reliable** for repeated use without fixes. Risk areas are silent degradation on GNN failure paths, version-dispatch complexity, and lack of automated tests around association and track management (the parts that determine MOTA).

## Issues

### Issue 1 -- Severity: bug
- File: src/pipeline.py:124
- Description: `np.mean` is called but `numpy` is never imported. Top-level imports are only `abc`, `typing`, `torch`, `PipelineConfig`, `updater`, and `clutter_classifier`. Any call to `process_frame` with measurements will raise `NameError: name 'np' is not defined`.
- Suggestion: Add `import numpy as np` (or use `statistics.mean` / pure Python for a single-value case). Also consider making the timestamp extraction more robust (measurements may legitimately lack 't').
- Status: **fixed** (see "Fixes Applied" section above)

### Issue 2 -- Severity: bug
- File: run_cli.py:10
- Description: `import psutil` appears twice consecutively (the "Structural Fix" commit moved one but left the original). Harmless but sloppy and contradicts the commit message.
- Suggestion: Remove the duplicate.
- Status: **fixed**

### Issue 3 -- Severity: bug
- File: src/updater.py:234
- Description: Dead / unreachable code in `GNNUpdater.update`. The telemetry block (`self.frame_count += 1`, logging of `updated_tracks`, etc.) sits after multiple `return` statements (normal success path at 224, exception path at 231). It will never execute. Similar scattered `import logging` / `import traceback` inside the method.
- Suggestion: Move telemetry to the correct location (before or instead of early returns) or delete it. Consolidate imports at module top.
- Status: **fixed** (dead code excised; see Fixes Applied)

### Issue 4 -- Severity: bug
- File: src/updater.py:104 (and surrounding)
- Description: Inside `GNNUpdater.update`, `get_model_suite(model_ver)` is called again even though the model was already loaded via the suite in `__init__` / `_load_model`. Combined with repeated `model_ver = getattr(...)` and per-frame re-dispatch, this adds unnecessary import + attribute lookup cost and complexity on the hot path.
- Suggestion: Cache the dispatched callables (`frame_to_tensors`, `build_gnn_edges`, etc.) on `self` after first load, similar to how `self.model` and `self.suite` are already stored. Remove the inner re-import of factory.
- Status: **fixed** (caching implemented + defensive fallback; see Fixes Applied)

### Issue 5 -- Severity: suggestion
- File: src/pipeline.py:161 (process_frame track management) and multiple config/CLI sites
- Description: Track deletion logic (`if t.get('age', 0) < max_age: keep`) is easy to misread. "age" here is used as a coasting/miss counter (incremented in predict and on unmatched updates). The variable name + config field `max_age` / `del_age` and the CLI flag `--max-age` (default 2) vs `PipelineConfig` default (10) vs `StateUpdaterConfig.del_age` (15) create a three-way default mismatch. Different call sites (CLI override only touches some fields) can leave the running system with surprising values.
- Suggestion: 
  1. Rename the counter to `coasted` or `miss_streak` for clarity.
  2. Centralize **all** numeric defaults inside the Pydantic models.
  3. Make the CLI parser only construct/override a `PipelineConfig` instance and pass it down; never have parallel argparse defaults.
  4. Add a docstring or comment block explaining the exact semantics and units.
- Status: open

### Issue 6 -- Severity: suggestion
- File: src/updater.py (GNNUpdater._load_model and NewHybridUpdater.__init__) + src/pipeline.py:101
- Description: Multiple places hard-code checkpoint paths (`checkpoints/pairwise_psr_psr.pt`, `pairwise_ssr_any.pt`, `clutter_classifier.pt`, model paths) instead of reading them from the injected `PipelineConfig` (which already has `PairwiseConfig`, `ClutterFilterConfig`, `StateUpdaterConfig.gnn_model_path`). Some paths use `weights_only=True`, others `False`; error handling ranges from silent `None` to `RuntimeError`.
- Suggestion: Thread config all the way through. Make the updater constructors take the relevant sub-configs (or the full `PipelineConfig`) and resolve paths from there. Standardize `weights_only` policy (prefer True for inference).
- Status: open

### Issue 7 -- Severity: suggestion
- File: src/config_schemas.py:79 (DatasetConfig) + run_cli.py:209
- Description: `DatasetConfig.path` validator calls `Path(v).exists()` at model construction time. This is too eager for CLI usage and for configs that may be created before data is staged. CLI currently bypasses `DatasetConfig` entirely when building `PipelineConfig()`.
- Suggestion: Make the existence check optional (e.g., a `validate_existence: bool = True` or a separate `validate()` step after overrides). Or remove the validator and document that the loader is responsible for clear errors.
- Status: open

### Issue 8 -- Severity: suggestion
- File: src/updater.py:226 (GNNUpdater), pipeline.py (various), NewHybridUpdater
- Description: Many failure paths do `return tracks` (or the input list) after logging a warning or error. This can cause silent coasting or zero-track output with no signal to the caller or metrics that something is wrong. In GNNUpdater the except also swallows the exception after logging.
- Suggestion: At minimum, increment ages on all tracks when returning early due to model failure. Consider a distinguished return or exception for "updater degraded" so the pipeline or caller can react (e.g., fall back to pure Kalman, raise an alert, or mark the run unhealthy). Add structured logging with exception info.
- Status: open

### Issue 9 -- Severity: suggestion
- File: src/factory.py + src/updater.py + all model_v*.py + train_streaming_v*.py
- Description: The version dispatch (string "v3"/"v4"/"v5"/"v6", shape sniffing in `detect_model_version`, per-version ifs for decoder output count, cross-attn presence, clutter_head, dustbin, etc.) has spread across factory, updater, models, and training scripts. Each new version adds branches. The "Unified" goal of the Phase 1 commit is only partially realized.
- Suggestion: Treat the factory + suite as the single source of truth. Push more version-specific behavior into the model classes themselves (e.g., a `version` class attr, a `build_edges(x, node_type, **classifiers_or_none)` that the model knows how to call, or a small adapter). Consider a registry or explicit `ModelVersion` enum instead of magic strings. Document the intended supported versions and a deprecation path for older ones.
- Status: open

### Issue 10 -- Severity: nit
- File: pyproject.toml
- Description: Project metadata is placeholder: `description = "Add your description here"`, no authors, license field, urls, or classifiers. The citation in README is good, but packaging metadata should match.
- Suggestion: Fill in proper `[project]` fields (authors, readme, license, urls). Add a `license` file reference if not already present.
- Status: open

### Issue 11 -- Severity: suggestion (repo-wide)
- File: (multiple commits + root layout)
- Description: The 5 commits (and prior history) added large numbers of logs, data subsets, eval .txt/.png/.log, design artifacts, and temporary scripts directly to the repository. Even with an updated `.gitignore`, the objects are already in history. `mlruns/` (12k+ files), `checkpoints/` (many .pt), `runs/`, `logs/`, and various root-level .log/.txt/.png files pollute clones, increase repo size, and make `git status` / searches noisy. No `data/` policy or `artifacts/` retention guidance exists.
- Suggestion:
  - Add or expand `.gitignore` patterns aggressively for `logs/`, `eval*`, `*.log`, `v*_*.txt`, `optimization_reports/`, `tmp*.py`, `debug*.log`, `train_debug*.log`, `sweden_eval.png`, etc.
  - Document in README or a `DATA.md` / `CONTRIBUTING.md` the canonical datasets and how to obtain them without committing extracts.
  - Consider `git filter-repo` or BFG (one-time) if the repo is still small enough in contributor count, or simply start fresh branches for "clean main" and note that history contains experiment byproducts.
  - Move long-term artifacts to a separate bucket or DVC/git-lfs setup if they must be referenced.
- Status: open

### Issue 12 -- Severity: suggestion (project-wide)
- File: tests/ (only test_metrics_robustness.py exists)
- Description: One test file for metrics. No tests exercising the pipeline, updaters, association logic, factory version dispatch, track management, or end-to-end MOTA on a small synthetic case. Trackers are notoriously sensitive to off-by-one association, ID switch, and coasting bugs; manual visual inspection + occasional eval logs are insufficient.
- Suggestion: Add a small test suite with:
  - Deterministic synthetic data (2-3 targets, known measurements).
  - Assertions on MOTA/MOTP/ID switches for the different modes (gnn/kalman/hybrid).
  - Tests that the modular pipeline + a dummy updater produces the expected number of confirmed tracks.
  - Regression tests that vX model forward passes and manage_tracks produce sane outputs (even if weights are random).
  Use pytest + perhaps hypothesis for property tests on association gates.
- Status: open

### Issue 13 -- Severity: nit
- File: (throughout recent changes)
- Description: Mix of `print("[OK] ...")`, bare `print`, `logging.info`, and `logging.error` (sometimes with ad-hoc `import logging` inside functions). The new rotating handler is an improvement, but the dual channels make output noisy and hard to control in dashboard or batch runs.
- Suggestion: Standardize on the module logger (`logging.getLogger(__name__)`). Keep high-level status prints only when `--interactive`. Route all diagnostics through logging so the RotatingFileHandler + dashboard capture everything uniformly.
- Status: open

## Positive Notes (not issues)

- The `BipartiteCrossAttention` + dustbin design in V6 is a clean way to handle "no match" cases and is a genuine architectural improvement.
- `NewHybridUpdater` (with its spatial clustering via learned classifiers + proper async KF timing) directly addresses real multi-radar fusion problems and re-uses proven components from the legacy high-MOTA hybrid.
- Pydantic configs + the factory are the right abstraction layer for the "modular research platform" goal stated in the README.
- CLI fleet commands (`--list`, `--kill`) and rotating logs show operational maturity that is rare in pure research codebases.
- Phased training hooks (`get_phase_params`) and curriculum args are thoughtful for the "Phase 1/2/3/4" curriculum described in the commits.

## Recommendations / Next Steps (prioritized)

1. **Immediate (before next run)**: Fix the `np` import crash and the duplicate `psutil` import. Verify `uv run run_cli.py --mode hybrid` completes a few frames on a small data file.
2. **Short term**: Unify defaults under `PipelineConfig`, wire the missing CLI overrides (init/coast/suppress thresh), remove dead code, cache the factory dispatchables.
3. **Medium term**: Add a handful of fast unit + integration tests that would have caught the import bug and would guard association/track-management behavior. Add a CI step that runs them on push.
4. **Hygiene**: Decide on a data/artifact policy. Clean future commits. Consider a repo hygiene PR or history rewrite note.
5. **Architecture**: Pick (or declare) the "current" recommended stack (e.g., "v6 + hybrid updater + clutter filter") and make older versions opt-in or remove them from the default path to reduce branching.
6. **Documentation**: Add a short `ARCHITECTURE.md` or update the existing artifacts/ one that describes the current data flow (Pipeline → Clutter → Router → Updater (GNN/Hybrid/KF) → Track Manager) and the contract between components.

See the source files listed in the "changed code files" section of the invocation for the exact diffs that were considered. All line numbers refer to the post-change (HEAD) versions of the files.

**Review file location**: This document (REVIEW_current_project.md) plus any additional reviewer subagent notes.
