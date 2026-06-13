# Setup the repo .venv via uv (installs all deps + the local package)
setup:
    uv sync

# Format code according to project style
format:
    uvx ruff format .

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format + lint (CI-style target)
all: format check

# ── Dual-SIM ─────────────────────────────────────────────────────────────────
# Full sweeps use dedicated primary configs (config/config_*.yaml) with
# hydra.mode=MULTIRUN so no -m flag is needed.
# Sweep values are defined in each config file.
#
# Per-architecture aggregates run every sweep for that case:
#   just joint        # = run-multi + run-snr + run-slayer
#   just disjoint     # = run-disjoint + run-disjoint-snr
#   just mono         # = run-mono + run-mono-snr
#
# Examples:
#   just run-multi                            # layers sweep (PPFE, cifar10)
#   just run-multi alignment=linear
#   just run-multi datasets=cifar100
#   just run-snr
#   just run-slayer
#   just run-disjoint
#   just run-disjoint-snr

run-multi *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_layers {{ARGS}}

run-snr *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_snr {{ARGS}}

run-slayer *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_slayer {{ARGS}}

run-disjoint *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_layers {{ARGS}}

run-disjoint-snr *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_snr {{ARGS}}

# Run every joint Dual-SIM sweep: layers + SNR + slayer
joint: run-multi run-snr run-slayer

# Run every disjoint Dual-SIM sweep: layers + SNR
disjoint: run-disjoint run-disjoint-snr

# Single run without multirun (useful for quick tests)
run *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py {{ARGS}}

# ── Mono-SIM (TX-only ablation) ───────────────────────────────────────────────
# Examples:
#   just mono                                 # both sweeps (layers + SNR)
#   just run-mono                             # full layers sweep
#   just run-mono alignment=linear
#   just run-mono experiment.use_channel=true
#   just run-mono-snr                         # SNR sweep with MIMO channel

run-mono *ARGS: train-clf
    uv run python scripts/mono_sim.py --config-name config_mono_layers {{ARGS}}

run-mono-snr *ARGS: train-clf
    uv run python scripts/mono_sim.py --config-name config_mono_snr {{ARGS}}

# Run every Mono-SIM sweep: layers + SNR
mono: run-mono run-mono-snr

# ── Neural classifier ─────────────────────────────────────────────────────────
# Train one MLP per (dataset, rx_enc, seed) and save to models/classifiers/.
# Sweep seeds and rx_enc models are defined in config/config_train_clf.yaml.
#
# Examples:
#   just train-clf                              # full sweep (all seeds × models)
#   just train-clf datasets=cifar100
#   just train-clf "training.seed=27,42"

train-clf *ARGS:
    uv run python scripts/train_classifier.py {{ARGS}}

# ── Analysis ──────────────────────────────────────────────────────────────────

# Extract per-seed Mono-SIM Parquets from detailed results
extract-mono:
    uv run python scripts/extract_monosim.py

# Generate all plots → results/plots/
plot:
    uv run python scripts/plot_results.py
