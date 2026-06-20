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
#
# Tipology overrides (pass as extra ARGS or use dedicated recipes below):
#   sim.constructive=true           — H in loss instead of target
#   sim.alignment_side=rx           — alignment on RX-SIM (asym only)
#
# Example:
#   just run-disjoint-asym sim.constructive=true sim.alignment_side=rx

# ── Joint Dual-SIM ────────────────────────────────────────────────────────────

run-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_layers {{ARGS}}

run-joint-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_snr {{ARGS}}

run-joint-slayer *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_slayer {{ARGS}}

run-joint-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_channel_dim {{ARGS}}

joint: run-joint run-joint-snr run-joint-channel-dim

# ── Disjoint asym ─────────────────────────────────────────────────────────────
# Tipology: alignment_side ∈ {tx, rx} × constructive ∈ {false, true}
# Results go to results/disjoint_asym_*/tx|tx_c|rx|rx_c/

run-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers {{ARGS}}

run-disjoint-asym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr {{ARGS}}

# constructive=true, alignment_side=tx  →  tipology: tx_c
run-disjoint-asym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers sim.constructive=true {{ARGS}}

run-disjoint-asym-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr sim.constructive=true {{ARGS}}

# constructive=true, alignment_side=rx  →  tipology: rx_c
run-disjoint-asym-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers sim.constructive=true sim.alignment_side=rx {{ARGS}}

run-disjoint-asym-snr-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr sim.constructive=true sim.alignment_side=rx {{ARGS}}

# Aggregates
disjoint-asym: run-disjoint-asym run-disjoint-asym-snr
disjoint-asym-c: run-disjoint-asym-c run-disjoint-asym-snr-c
disjoint-asym-rx-c: run-disjoint-asym-rx-c run-disjoint-asym-snr-rx-c

disjoint-asym-all: disjoint-asym disjoint-asym-c disjoint-asym-rx-c

# ── Disjoint sym ──────────────────────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/disjoint_sym_*/standard|constructive/

run-disjoint-sym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers {{ARGS}}

run-disjoint-sym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr {{ARGS}}

run-disjoint-sym-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim {{ARGS}}

# constructive=true  →  tipology: constructive
run-disjoint-sym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers sim.constructive=true {{ARGS}}

run-disjoint-sym-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr sim.constructive=true {{ARGS}}

run-disjoint-sym-channel-dim-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim sim.constructive=true {{ARGS}}

disjoint-sym: run-disjoint-sym run-disjoint-sym-snr run-disjoint-sym-channel-dim
disjoint-sym-c: run-disjoint-sym-c run-disjoint-sym-snr-c run-disjoint-sym-channel-dim-c
disjoint-sym-all: disjoint-sym disjoint-sym-c

# ── Disjoint 4-SIM ────────────────────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/disjoint_4sim_*/standard|constructive/

run-disjoint-4sim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers {{ARGS}}

run-disjoint-4sim-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr {{ARGS}}

run-disjoint-4sim-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_channel_dim {{ARGS}}

run-disjoint-4sim-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers sim.constructive=true {{ARGS}}

run-disjoint-4sim-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr sim.constructive=true {{ARGS}}

disjoint-4sim: run-disjoint-4sim run-disjoint-4sim-snr run-disjoint-4sim-channel-dim
disjoint-4sim-c: run-disjoint-4sim-c run-disjoint-4sim-snr-c
disjoint-4sim-all: disjoint-4sim disjoint-4sim-c

# ── Cross-architecture aggregates ─────────────────────────────────────────────

# All standard (non-constructive) disjoint variants
disjoint: disjoint-asym disjoint-sym disjoint-4sim

# All constructive disjoint variants
disjoint-constructive: disjoint-asym-c disjoint-sym-c disjoint-4sim-c

# ── Mono-SIM (TX-only ablation) ───────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/mono*/standard|constructive/

run-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers {{ARGS}}

run-mono-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr {{ARGS}}

# constructive=true  →  tipology: constructive (loss ‖β H G_T − A‖², no G_eq at RX)
run-mono-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers sim.constructive=true {{ARGS}}

run-mono-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr sim.constructive=true {{ARGS}}

mono: run-mono run-mono-snr
mono-c: run-mono-c run-mono-snr-c
mono-all: mono mono-c

# ── n_proto sweep (PPFE only) ─────────────────────────────────────────────────

run-n-proto-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_joint {{ARGS}}

run-n-proto-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono {{ARGS}}

run-n-proto-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym {{ARGS}}

run-n-proto-disjoint-sym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_sym {{ARGS}}

run-n-proto-disjoint-sym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_sym_c {{ARGS}}

run-n-proto-disjoint-asym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym_c {{ARGS}}

run-n-proto-mono-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono_c {{ARGS}}

n-proto: run-n-proto-joint run-n-proto-mono run-n-proto-disjoint-asym run-n-proto-disjoint-sym
n-proto-c: run-n-proto-disjoint-sym-c run-n-proto-disjoint-asym-c run-n-proto-mono-c
n-proto-all: n-proto n-proto-c

# ── Neural classifier ─────────────────────────────────────────────────────────

train-clf *ARGS:
    uv run python scripts/train_classifier.py {{ARGS}}

# ── Single run (no multirun, useful for quick tests) ─────────────────────────

run *ARGS: train-clf
    uv run python scripts/classification.py {{ARGS}}

# ── Analysis ──────────────────────────────────────────────────────────────────

plot:
    uv run python scripts/plot_results.py
