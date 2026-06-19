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

# ── SIM experiments (all via scripts/classification.py) ──────────────────────
# Full sweeps use dedicated primary configs (config/config_*.yaml) with
# hydra.mode=MULTIRUN so no -m flag is needed.
# The experiment YAML's _target_ selects the runner automatically.
#
# Per-architecture aggregates:
#   just joint        # = run-joint + run-joint-snr + run-joint-slayer
#   just disjoint     # = run-disjoint-asym + run-disjoint-sym + 4-SIM variants
#   just mono         # = run-mono + run-mono-snr

run-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_layers {{ARGS}}

run-joint-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_snr {{ARGS}}

run-joint-slayer *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_slayer {{ARGS}}

# Run every joint Dual-SIM sweep: layers + SNR + slayer + channel-dim
joint: run-joint run-joint-snr run-joint-slayer run-joint-channel-dim

run-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers {{ARGS}}

run-disjoint-asym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr {{ARGS}}

run-disjoint-sym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers {{ARGS}}

run-disjoint-sym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr {{ARGS}}

run-disjoint-4sim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers {{ARGS}}

run-disjoint-4sim-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr {{ARGS}}

# ── Channel-dim sweep (L=10, M_int=32, channel_dim in {1,2,4,6,8,10,20,192,384}) ──

run-joint-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_channel_dim {{ARGS}}

run-disjoint-asym-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_channel_dim {{ARGS}}

run-disjoint-sym-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim {{ARGS}}

run-disjoint-4sim-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_channel_dim {{ARGS}}

# Run all channel-dim sweeps
channel-dim: run-joint-channel-dim run-disjoint-asym-channel-dim run-disjoint-sym-channel-dim run-disjoint-4sim-channel-dim

# Aggregate: asymmetric disjoint (layers + SNR + channel-dim)
disjoint-asym: run-disjoint-asym run-disjoint-asym-snr run-disjoint-asym-channel-dim

# Aggregate: symmetric disjoint (layers + SNR + channel-dim)
disjoint-sym: run-disjoint-sym run-disjoint-sym-snr run-disjoint-sym-channel-dim

# Aggregate: 4-SIM disjoint (layers + SNR + channel-dim)
disjoint-4sim: run-disjoint-4sim run-disjoint-4sim-snr run-disjoint-4sim-channel-dim

# Run every disjoint Dual-SIM sweep: all variants
disjoint: disjoint-asym disjoint-sym

# Single run without multirun (useful for quick tests)
run *ARGS: train-clf
    uv run python scripts/classification.py {{ARGS}}

# ── Mono-SIM (TX-only ablation) — same script, different config ───────────────

run-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers {{ARGS}}

run-mono-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr {{ARGS}}

# Run every Mono-SIM sweep: layers + SNR
mono: run-mono run-mono-snr

# ── n_proto sweep (PPFE only, L=10, M_int=32, channel_dim=20) ────────────────

run-n-proto-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_joint {{ARGS}}

run-n-proto-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono {{ARGS}}

run-n-proto-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym {{ARGS}}

# Run all three n_proto sweeps
n-proto: run-n-proto-joint run-n-proto-mono run-n-proto-disjoint-asym

# ── Neural classifier ─────────────────────────────────────────────────────────
# Train one MLP per (dataset, rx_enc, seed) and save to models/classifiers/.
# Sweep seeds and rx_enc models are defined in config/config_train_clf.yaml.

train-clf *ARGS:
    uv run python scripts/train_classifier.py {{ARGS}}

# ── Analysis ──────────────────────────────────────────────────────────────────

# Generate all plots → results/plots/
plot:
    uv run python scripts/plot_results.py
