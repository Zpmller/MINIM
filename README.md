# MINIM: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization (ICML'26)

本地复现的环境搭建、失败诊断、受控试错、各版本阈值与最终结果见 [项目全程总结](reports/MINIM_项目全程总结.md)（更新至 2026-11-04）。原始配方对照与修改版基线分别记录。

2026-11-05 [CE高分复核](reports/CE效果复核_20261105.md)：分数可复算，但存在跨划分近重复页面及简单规则高分，泛化解释需要收窄；独立审计服务未完成的状态另行记录。

MINIM is a local sanitization layer for LLM-powered agents. Instead of sending a full accessibility tree or UI state to a remote model, MINIM builds a task-conditioned minimal view: keep what the agent needs, abstract sensitive-but-needed fields, and remove everything else.

This repository contains the reference implementation for the ICML 2026 paper **"MINIM: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization"**. The companion dataset is available on [Hugging Face](https://huggingface.co/datasets/Chaoyu112358/MINIM-data).

## Why MINIM?

Modern web and OS agents often operate over structured observations such as accessibility trees. These observations are useful because they expose roles, text, buttons, inputs, and hierarchy in a form agents can act on. They are also risky: the same tree may contain the email the user wants summarized, a sidebar notification, a 2FA code, unrelated tabs, saved-address fields, or other private context.

MINIM targets this failure mode, which we call **Semantic Over-Privileged Observation**. The privacy problem is not just that sensitive information exists; it is that task-irrelevant sensitive information is disclosed together with action-critical UI structure.

The key idea is simple:

- **Sensitivity** asks: how risky is this UI element if disclosed?
- **Task-conditioned necessity** asks: does the current task actually need this element?
- A local policy maps those two scores to **Keep**, **Abstract**, or **Remove** before the remote agent receives the observation.

This turns privacy control into a pre-disclosure bottleneck on the client side, grounded in Contextual Integrity: information should flow only when it is appropriate for the active task context.

## What This Release Provides

This codebase is intentionally compact. It is meant to help researchers reproduce the core MINIM pipeline and build on it:

- Train a local scorer over annotated accessibility-tree nodes.
- Predict sensitivity and task-conditioned necessity for each UI element.
- Apply the MINIM disclosure policy to produce sanitized observations.
- Reproduce the main evaluation scripts for privacy-utility analysis.

The implementation uses a lightweight GATv2 scorer by default and includes the evaluation metrics used in the paper: task-critical context preservation, task-critical interactive preservation, and task-irrelevant sensitive leakage.

## Result Snapshot

On the released augmented WebArena accessibility-tree benchmark, MINIM preserves nearly all task-critical interactive affordances while substantially reducing leakage from irrelevant sensitive UI content.

| Method | TCNP | TCNP-I | TISL |
|--------|------|--------|------|
| Full Observation | 1.0000 | 1.0000 | 1.0000 |
| Random Budget | 0.2284 | 0.2346 | 0.1971 |
| Sensitivity-Only | 0.0401 | 0.0393 | 0.3799 |
| Necessity-Only | 0.9445 | 0.9730 | 0.2032 |
| **MINIM** | **0.9491** | **0.9931** | **0.1010** |

TCNP measures preservation of task-critical context, TCNP-I focuses on task-critical interactive elements, and TISL measures normalized leakage from task-irrelevant sensitive content. Lower TISL is better.

## Dataset

Download the annotated dataset from Hugging Face:

[Hugging Face dataset: Chaoyu112358/MINIM-data](https://huggingface.co/datasets/Chaoyu112358/MINIM-data)

Place it at `data/`:

```bash
data/
├── manifest.jsonl
└── samples/
    └── {domain}/{sample_id}/variants/{variant}/tree_with_scores.json
```

The dataset contains 5,750 annotated accessibility-tree variants across Gmail, Reddit, and shopping domains. Each node preserves standard AX-tree fields and adds two 0-10 annotations: `sensitivity_score` and `necessity_score`.

All scripts default to `--data_root data`.

## Quick Start

```bash
conda create -n minim python=3.10 -y
conda activate minim
pip install -r requirements.txt
```

Train the default scorer:

```bash
python train_minim_scorer.py
```

Run MINIM inference on the test split:

```bash
python run_minim_inference.py --predict_split test
```

Reproduce the main paper artifacts:

```bash
python experiment_table1_main_results.py
python experiment_figure2_threshold_sweep.py
python experiment_table3_task_semantics.py
python experiment_table4_abstraction_ablation.py
python experiment_table10_domain_specific.py
python experiment_figure3_policy_distribution.py
```

Outputs are written to `runs/minim/` and `results/`.

## Small-scale repair study

The original run collapsed to all-zero scores. A separate repair study compares
legacy task hashes and the original L1 objective against unique task IDs,
class-balanced L1, and class-balanced cross entropy:

```bash
bash scripts/run_small_scale_fix.sh
```

The study uses 6 training trees, 3 validation trees, and 3 held-out test trees,
spanning all three domains. It keeps one balanced annotation variant per task
plus the reading variants, avoiding repeated persona copies. All six arms use
the same seed, initial weights, batches, and 256-wide GATv2 architecture.
Class weights come only from training labels. Checkpoints are selected using
validation micro TCNP minus 0.5 times validation TISL, with fixed policy thresholds
1 and 5. Test evaluation runs after all training and validation selection finish.
Artifacts are separate under `runs/pilot_fix/`; these are debugging results,
not a full reproduction of the paper.

The main trainer also supports `--task_encoding unique` and
`--loss_type balanced_ce` (or `balanced_l1`). Both defaults remain the legacy
configuration for reproducibility. Save repaired runs to a new `--output_dir`.
Unique task vocabularies are stored in checkpoints and restored automatically
by inference and all experiment scripts. Unknown tasks are rejected explicitly.
Existing feature caches retain their legacy representation; unique task features
are applied after loading, preserving the old checkpoints and text embeddings.
The extra two arms test the loss function's `(2, 2, 1)` boost defaults and
class-prior correction for balanced CE. The latter subtracts log training class
weights from inference logits, reducing the score bias introduced by reweighting;
classes absent from training are masked instead of boosted. Enable it in the
main trainer with `--task_encoding unique --loss_type balanced_ce --prior_correction`.

## Paper-aligned baseline and training-data scaling

```bash
bash scripts/run_training_scale_study.sh
```

This first runs the frozen official model and CI-L1 loss with the paper's
256-wide GATv2, 10 epochs, and effective batch 32. Node-weighted gradient
accumulation over micro-batches of four adapts the batch to the local GPU.
It then runs a separate repaired-model study using nested sets of 6, 15, and
30 training trees, a common validation set of 15 trees, and the same full
released test split. Official and modified objectives remain separate.
The released data/split differ from the paper; the baseline aligns verifiable
training settings and does not claim to recover the unpublished original split.
See `reports/MINIM_官方对齐清单.md` for the source and configuration audit.

## 27-task collapse diagnostics

`scripts/causal27_diagnostics.py --stage trace` traces the first official CI-L1
epoch and performs controlled optimizer-state recovery probes. `--stage factorial`
compares L1/CE with and without class balancing on the same 27-task subset,
retaining the original task encoding. Diagnostic outputs are separate under
`runs/causal27_20261102/` and `results/causal27_20261102/`.

The experimental `--loss_type ci_ce` option retains CI risk weights and node-mean
normalization, but supervises the existing 11-bin heads using cross entropy.
It omits class balancing and inference prior correction. This changes the paper's
reported training objective; the default remains `ci_l1`. It is not an author
checkpoint or a claim of reproducing the author's original training recipe.

## Full-data working baseline and reference performance

```bash
bash scripts/run_reference_profile.sh
```

This local experiment uses the complete released split (4,637 train, 629
validation, 484 test), 256-wide GATv2, native batch 32, and 10 epochs. It changes
the original objective to CI-weighted cross entropy and replaces colliding task
hashes with checkpoint-owned unique task IDs. It uses no class balancing or
inference prior correction. Original trainer defaults remain available.

The last checkpoint is calibrated using validation only. The policy selection
rule seeks the paper's preservation targets and leakage budget, retains better
preservation, and prefers lower leakage; it does not force results to equal
published numbers. Both fixed official thresholds (1/5) and the validation-selected
policy are evaluated. The protocol is in `scripts/reference_profile_protocol.json`;
checkpoints and outputs use isolated `reference_profile_20261104` directories.
All six table/figure scripts are exported for the selected policy and checked
against independent metric computation. The paper's 662-variant test split is
unavailable, so this is a modified working baseline rather than an exact numerical
reproduction. See `reports/MINIM_参考性能对齐_20261104.md` after completion.

## How MINIM Decides What to Reveal

For each node, the scorer predicts `(s_hat, n_hat)`:

```python
if n_hat < tau_nec:
    action = "Remove"
elif s_hat >= tau_sens:
    action = "Abstract"
else:
    action = "Keep"
```

Default thresholds are `tau_nec=1.0` and `tau_sens=5.0`. `Remove` drops a node from the sanitized tree. `Abstract` preserves the node's structural role while replacing sensitive text fields with `[REDACTED]`. `Keep` passes the element through unchanged.

This three-way policy is the core of MINIM: binary allow/deny filters either leak too much or break tasks, while abstraction keeps useful structure without exposing raw sensitive values.

## Repository Guide

Core files:

- `train_minim_scorer.py`: model, features, and training loop.
- `run_minim_inference.py`: checkpoint loading and K/A/R sanitization.
- `evaluate_metrics.py`: TCNP, TCNP-I, and TISL.
- `experiment_*.py`: scripts for reproducing the released tables and figures.

Generated files:

- `runs/minim/`: checkpoints, split manifest, and predictions.
- `results/`: experiment CSVs and figures.

## Citation

If you use MINIM, please cite:

```bibtex
@inproceedings{yu2026minim,
  title={MINIM: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization},
  author={Yu, Hexuan and Zhang, Chaoyu and Jin, Heng and Shi, Shanghao and Zhang, Ning and Hou, Y. Thomas and Lou, Wenjin},
  booktitle={Forty-Third International Conference on Machine Learning},
  year={2026},
  address={Seoul, South Korea}
}
```

