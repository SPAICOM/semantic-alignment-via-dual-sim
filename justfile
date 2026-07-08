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

# Run wandb leet for experiment tracking
leet:
    # Run wandb leet for experiment tracking
    uv run wandb beta leet run wandb

# ── SIM experiments (all via scripts/classification.py) ──────────────────────
# Full sweeps use dedicated primary configs (config/config_*.yaml) with
# hydra.mode=MULTIRUN so no -m flag is needed.
#
# Dataset override (default: cifar10):
#   just dataset=cifar100 run-joint
#   just dataset=cifar100 joint
#
# Tipology overrides (pass as extra ARGS or use dedicated recipes below):
#   sim.constructive=true           — H in loss instead of target
#   sim.alignment_side=rx           — alignment on RX-SIM (asym only)
#
# Example:
#   just dataset=cifar100 run-disjoint-asym sim.constructive=true sim.alignment_side=rx

dataset := "cifar10"

# ── Joint Dual-SIM ────────────────────────────────────────────────────────────

# Joint Dual-SIM: sweep over SIM layers
run-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_layers datasets={{dataset}} {{ARGS}}

# Joint Dual-SIM: sweep over SNR
run-joint-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_snr datasets={{dataset}} {{ARGS}}

# Joint Dual-SIM: sweep over inter-layer spacing
run-joint-slayer *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_slayer datasets={{dataset}} {{ARGS}}

# Joint Dual-SIM: sweep over channel dimension
run-joint-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_channel_dim datasets={{dataset}} {{ARGS}}

# Aggregate: all joint Dual-SIM sweeps
joint: run-joint run-joint-snr run-joint-channel-dim

# ── Disjoint asym ─────────────────────────────────────────────────────────────
# Tipology: alignment_side ∈ {tx, rx} × constructive ∈ {false, true}
# Results go to results/disjoint_asym_*/equalized|constructive/tx|rx/

# Disjoint asym (tx, equalized): sweep over SIM layers
run-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers datasets={{dataset}} {{ARGS}}

# Disjoint asym (tx, equalized): sweep over SNR
run-disjoint-asym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr datasets={{dataset}} {{ARGS}}

# constructive=true, alignment_side=tx  →  tipology: constructive/tx
# Disjoint asym (tx, constructive): sweep over SIM layers
run-disjoint-asym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers datasets={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint asym (tx, constructive): sweep over SNR
run-disjoint-asym-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr datasets={{dataset}} sim.constructive=true {{ARGS}}

# constructive=true, alignment_side=rx  →  tipology: constructive/rx
# Disjoint asym (rx, constructive): sweep over SIM layers
run-disjoint-asym-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers datasets={{dataset}} sim.constructive=true sim.alignment_side=rx {{ARGS}}

# Disjoint asym (rx, constructive): sweep over SNR
run-disjoint-asym-snr-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr datasets={{dataset}} sim.constructive=true sim.alignment_side=rx {{ARGS}}

# Aggregate: all disjoint asym (tx, equalized) sweeps
disjoint-asym: run-disjoint-asym run-disjoint-asym-snr
# Aggregate: all disjoint asym (tx, constructive) sweeps
disjoint-asym-c: run-disjoint-asym-c run-disjoint-asym-snr-c
# Aggregate: all disjoint asym (rx, constructive) sweeps
disjoint-asym-rx-c: run-disjoint-asym-rx-c run-disjoint-asym-snr-rx-c

# Aggregate: all disjoint asym sweeps
disjoint-asym-all: disjoint-asym disjoint-asym-c disjoint-asym-rx-c

# ── Disjoint sym ──────────────────────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/disjoint_sym_*/equalized|constructive/

# Disjoint sym (equalized): sweep over SIM layers
run-disjoint-sym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers datasets={{dataset}} {{ARGS}}

# Disjoint sym (equalized): sweep over SNR
run-disjoint-sym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr datasets={{dataset}} {{ARGS}}

# Disjoint sym (equalized): sweep over channel dimension
run-disjoint-sym-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim datasets={{dataset}} {{ARGS}}

# constructive=true  →  tipology: constructive
# Disjoint sym (constructive): sweep over SIM layers
run-disjoint-sym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers datasets={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint sym (constructive): sweep over SNR
run-disjoint-sym-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr datasets={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint sym (constructive): sweep over channel dimension
run-disjoint-sym-channel-dim-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim datasets={{dataset}} sim.constructive=true {{ARGS}}

# Aggregate: all disjoint sym (equalized) sweeps
disjoint-sym: run-disjoint-sym run-disjoint-sym-snr run-disjoint-sym-channel-dim
# Aggregate: all disjoint sym (constructive) sweeps
disjoint-sym-c: run-disjoint-sym-c run-disjoint-sym-snr-c run-disjoint-sym-channel-dim-c
# Aggregate: all disjoint sym sweeps
disjoint-sym-all: disjoint-sym disjoint-sym-c

# ── Disjoint 4-SIM ────────────────────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/disjoint_4sim_*/equalized|constructive/

# Disjoint 4-SIM (equalized): sweep over SIM layers
run-disjoint-4sim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers datasets={{dataset}} {{ARGS}}

# Disjoint 4-SIM (equalized): sweep over SNR
run-disjoint-4sim-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr datasets={{dataset}} {{ARGS}}

# Disjoint 4-SIM (equalized): sweep over channel dimension
run-disjoint-4sim-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_channel_dim datasets={{dataset}} {{ARGS}}

# Disjoint 4-SIM (constructive): sweep over SIM layers
run-disjoint-4sim-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers datasets={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint 4-SIM (constructive): sweep over SNR
run-disjoint-4sim-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr datasets={{dataset}} sim.constructive=true {{ARGS}}

# Aggregate: all disjoint 4-SIM (equalized) sweeps
disjoint-4sim: run-disjoint-4sim run-disjoint-4sim-snr run-disjoint-4sim-channel-dim
# Aggregate: all disjoint 4-SIM (constructive) sweeps
disjoint-4sim-c: run-disjoint-4sim-c run-disjoint-4sim-snr-c
# Aggregate: all disjoint 4-SIM sweeps
disjoint-4sim-all: disjoint-4sim disjoint-4sim-c

# ── Cross-architecture aggregates ─────────────────────────────────────────────

# Aggregate: all standard disjoint variants
disjoint: disjoint-asym disjoint-sym disjoint-4sim

# Aggregate: all constructive disjoint variants
disjoint-constructive: disjoint-asym-c disjoint-sym-c disjoint-4sim-c

# ── Mono-SIM (TX-only ablation) ───────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/mono*/equalized|constructive/

# Mono-SIM (equalized): sweep over SIM layers
run-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers datasets={{dataset}} {{ARGS}}

# Mono-SIM (equalized): sweep over SNR
run-mono-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr datasets={{dataset}} {{ARGS}}

# constructive=true  →  tipology: constructive (loss ‖β H G_T − A‖², no G_eq at RX)
# Mono-SIM (constructive): sweep over SIM layers
run-mono-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers datasets={{dataset}} sim.constructive=true {{ARGS}}

# Mono-SIM (constructive): sweep over SNR
run-mono-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr datasets={{dataset}} sim.constructive=true {{ARGS}}

# Aggregate: all mono-SIM (equalized) sweeps
mono: run-mono run-mono-snr
# Aggregate: all mono-SIM (constructive) sweeps
mono-c: run-mono-c run-mono-snr-c
# Aggregate: all mono-SIM sweeps
mono-all: mono mono-c

# ── n_proto sweep (PPFE only) ─────────────────────────────────────────────────

# n_proto sweep: joint Dual-SIM
run-n-proto-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_joint datasets={{dataset}} {{ARGS}}

# n_proto sweep: mono-SIM (equalized)
run-n-proto-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono datasets={{dataset}} {{ARGS}}

# n_proto sweep: disjoint asym (tx, equalized)
run-n-proto-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym datasets={{dataset}} {{ARGS}}

# n_proto sweep: disjoint sym (equalized)
run-n-proto-disjoint-sym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_sym datasets={{dataset}} {{ARGS}}

# n_proto sweep: disjoint sym (constructive)
run-n-proto-disjoint-sym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_sym_c datasets={{dataset}} {{ARGS}}

# n_proto sweep: disjoint asym (tx, constructive)
run-n-proto-disjoint-asym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym_c datasets={{dataset}} {{ARGS}}

# n_proto sweep: disjoint asym (rx, constructive)
run-n-proto-disjoint-asym-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym_rx_c datasets={{dataset}} {{ARGS}}

# n_proto sweep: mono-SIM (constructive)
run-n-proto-mono-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono_c datasets={{dataset}} {{ARGS}}

# Aggregate: all n_proto (equalized) sweeps
n-proto: run-n-proto-joint run-n-proto-mono run-n-proto-disjoint-asym run-n-proto-disjoint-sym
# Aggregate: all n_proto (constructive) sweeps
n-proto-c: run-n-proto-disjoint-sym-c run-n-proto-disjoint-asym-c run-n-proto-disjoint-asym-rx-c run-n-proto-mono-c
# Aggregate: all n_proto sweeps
n-proto-all: n-proto n-proto-c

# ── Neural classifier ─────────────────────────────────────────────────────────

# Train a neural classifier (prerequisite for SIM experiments)
train-clf *ARGS:
    uv run python scripts/train_classifier.py datasets={{dataset}} {{ARGS}}

# ── Single run (no multirun, useful for quick tests) ─────────────────────────

# Single run (no multirun, for quick tests)
run *ARGS: train-clf
    uv run python scripts/classification.py datasets={{dataset}} {{ARGS}}

# ── Unrolled GD (progressive block training) ──────────────────────────────────
# Trains K=100 layers in blocks of m, saving results at each block endpoint.
# Config: config/training/unrolled.yaml

# Unrolled GD: joint Dual-SIM
run-unrolled *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_joint datasets={{dataset}} {{ARGS}}

# Unrolled GD: mono-SIM (equalized)
run-unrolled-mono *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_mono datasets={{dataset}} {{ARGS}}

# Unrolled GD: mono-SIM (constructive)
run-unrolled-mono-c *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_mono datasets={{dataset}} sim.constructive=true {{ARGS}}

# Unrolled GD: disjoint sym (equalized)
run-unrolled-disjoint-sym *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_disjoint_sym datasets={{dataset}} {{ARGS}}

# Unrolled GD: disjoint sym (constructive)
run-unrolled-disjoint-sym-c *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_disjoint_sym datasets={{dataset}} sim.constructive=true {{ARGS}}

# Unrolled GD: disjoint asym (tx, equalized)
run-unrolled-disjoint-asym *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_disjoint_asym datasets={{dataset}} {{ARGS}}

# Unrolled GD: disjoint asym (tx, constructive)
run-unrolled-disjoint-asym-c *ARGS: train-clf
    uv run python scripts/compare_unrolled.py --config-name config_compare_unrolled_disjoint_asym datasets={{dataset}} sim.constructive=true {{ARGS}}

# Aggregate: all unrolled GD experiments
unrolled: run-unrolled run-unrolled-mono run-unrolled-mono-c run-unrolled-disjoint-sym run-unrolled-disjoint-sym-c run-unrolled-disjoint-asym run-unrolled-disjoint-asym-c

# ── Analysis ──────────────────────────────────────────────────────────────────

# Generate publication-ready result plots
plot:
    uv run python scripts/plot_results.py
