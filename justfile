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
#   just joint        # = run-joint + run-joint-snr + run-joint-slayer
#   just disjoint     # = run-disjoint + run-disjoint-snr + CSI + 4-SIM variants
#   just mono         # = run-mono + run-mono-snr

run-joint *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_joint_layers {{ARGS}}

run-joint-snr *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_joint_snr {{ARGS}}

run-joint-slayer *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_joint_slayer {{ARGS}}

# Run every joint Dual-SIM sweep: layers + SNR + slayer
joint: run-joint run-joint-snr run-joint-slayer

run-disjoint *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_layers {{ARGS}}

run-disjoint-snr *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_snr {{ARGS}}

run-disjoint-csi *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_csi_layers {{ARGS}}

run-disjoint-csi-snr *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_csi_snr {{ARGS}}

run-disjoint-4sim *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_4sim_layers {{ARGS}}

run-disjoint-4sim-snr *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py --config-name config_disjoint_4sim_snr {{ARGS}}

# Aggregate: CSI disjoint (layers + SNR)
disjoint-csi: run-disjoint-csi run-disjoint-csi-snr

# Aggregate: 4-SIM disjoint (layers + SNR)
disjoint-4sim: run-disjoint-4sim run-disjoint-4sim-snr

# Run every disjoint Dual-SIM sweep: all variants
disjoint: run-disjoint run-disjoint-snr run-disjoint-csi run-disjoint-csi-snr run-disjoint-4sim run-disjoint-4sim-snr

# Single run without multirun (useful for quick tests)
run *ARGS: train-clf
    uv run python scripts/dual_classification_multiple.py {{ARGS}}

# ── Mono-SIM (TX-only ablation) ───────────────────────────────────────────────

run-mono *ARGS: train-clf
    uv run python scripts/mono_sim.py --config-name config_mono_layers {{ARGS}}

run-mono-snr *ARGS: train-clf
    uv run python scripts/mono_sim.py --config-name config_mono_snr {{ARGS}}

# Run every Mono-SIM sweep: layers + SNR
mono: run-mono run-mono-snr

# ── Neural classifier ─────────────────────────────────────────────────────────
# Train one MLP per (dataset, rx_enc, seed) and save to models/classifiers/.
# Sweep seeds and rx_enc models are defined in config/config_train_clf.yaml.

train-clf *ARGS:
    uv run python scripts/train_classifier.py {{ARGS}}

# ── Analysis ──────────────────────────────────────────────────────────────────

# Extract per-seed Mono-SIM Parquets from detailed results
extract-mono:
    uv run python scripts/extract_monosim.py

# Generate all plots → results/plots/
plot:
    uv run python scripts/plot_results.py
