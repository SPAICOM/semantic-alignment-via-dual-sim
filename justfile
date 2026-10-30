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
clean: format check

# Run wandb leet for experiment tracking
leet:
    uv run wandb beta leet run wandb

# ── SIM experiments ───────────────────────────────────────────────────────────
# Leaf recipes are prefixed cls- (classification) to distinguish from future
# rec- (reconstruction) recipes sharing the same methodology/sweep structure.
#
# Full sweeps use dedicated primary configs (config/config_*.yaml) with
# hydra.mode=MULTIRUN so no -m flag is needed.
#
# Dataset override (default: cifar10):
#   just dataset=cifar100 cls-joint-layers
#   just dataset=cifar100 joint
#
# Tipology overrides (pass as extra ARGS or use dedicated recipes below):
#   sim.constructive=true           — H in loss instead of target
#   sim.alignment_side=rx           — alignment on RX-SIM (asym only)
#
# Example:
#   just dataset=cifar100 cls-disjoint-asym-layers sim.constructive=true sim.alignment_side=rx

dataset := "cifar10"

# ── Joint Dual-SIM ────────────────────────────────────────────────────────────

# Joint Dual-SIM: sweep over SIM layers
cls-joint-layers *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_layers datasets/classification={{dataset}} {{ARGS}}

# Joint Dual-SIM: sweep over SNR
cls-joint-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_snr datasets/classification={{dataset}} {{ARGS}}

# Joint Dual-SIM: sweep over inter-layer spacing (linear + ppfe)
cls-joint-slayer *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_slayer datasets/classification={{dataset}} {{ARGS}}

# Disjoint Sym (constructive): sweep over inter-layer spacing (linear + ppfe)
cls-disjoint-sym-slayer-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_slayer datasets/classification={{dataset}} {{ARGS}}

# Aggregate: joint + disjoint-sym-constructive slayer sweeps
slayer: cls-joint-slayer cls-disjoint-sym-slayer-c

# Joint Dual-SIM: sweep over channel dimension
cls-joint-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_channel_dim datasets/classification={{dataset}} {{ARGS}}

# Aggregate: joint Dual-SIM (layers + SNR)
joint: cls-joint-layers cls-joint-snr

# ── Disjoint asym ─────────────────────────────────────────────────────────────
# Tipology: alignment_side ∈ {tx, rx} × constructive ∈ {false, true}
# Results go to results/classification/disjoint_asym_*/equalized|constructive/tx|rx/

# Disjoint asym (tx, equalized): sweep over SIM layers
cls-disjoint-asym-layers *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers datasets/classification={{dataset}} {{ARGS}}

# Disjoint asym (tx, equalized): sweep over SNR
cls-disjoint-asym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr datasets/classification={{dataset}} {{ARGS}}

# constructive=true, alignment_side=tx  →  tipology: constructive/tx
# Disjoint asym (tx, constructive): sweep over SIM layers
cls-disjoint-asym-layers-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint asym (tx, constructive): sweep over SNR
cls-disjoint-asym-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# constructive=true, alignment_side=rx  →  tipology: constructive/rx
# Disjoint asym (rx, constructive): sweep over SIM layers
cls-disjoint-asym-rx-layers-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_layers datasets/classification={{dataset}} sim.constructive=true sim.alignment_side=rx {{ARGS}}

# Disjoint asym (rx, constructive): sweep over SNR
cls-disjoint-asym-snr-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_snr datasets/classification={{dataset}} sim.constructive=true sim.alignment_side=rx {{ARGS}}

# Aggregate: disjoint asym (tx, equalized) — layers + SNR
disjoint-asym: cls-disjoint-asym-layers cls-disjoint-asym-snr
# Aggregate: disjoint asym (tx, constructive) — layers + SNR
disjoint-asym-c: cls-disjoint-asym-layers-c cls-disjoint-asym-snr-c
# Aggregate: disjoint asym (rx, constructive) — layers + SNR
disjoint-asym-rx-c: cls-disjoint-asym-rx-layers-c cls-disjoint-asym-snr-rx-c
# Aggregate: disjoint asym — equalized + constructive (layers + SNR, no channel-dim)
disjoint-asym-full: disjoint-asym disjoint-asym-c disjoint-asym-rx-c
# Aggregate: all disjoint asym sweeps
disjoint-asym-all: disjoint-asym disjoint-asym-c disjoint-asym-rx-c

# ── Disjoint sym ──────────────────────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/classification/disjoint_sym_*/equalized|constructive/

# Disjoint sym (equalized): sweep over SIM layers
cls-disjoint-sym-layers *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers datasets/classification={{dataset}} {{ARGS}}

# Disjoint sym (equalized): sweep over SNR
cls-disjoint-sym-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr datasets/classification={{dataset}} {{ARGS}}

# Disjoint sym (equalized): sweep over channel dimension
cls-disjoint-sym-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim datasets/classification={{dataset}} {{ARGS}}

# constructive=true  →  tipology: constructive
# Disjoint sym (constructive): sweep over SIM layers
cls-disjoint-sym-layers-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_layers datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint sym (constructive): sweep over SNR
cls-disjoint-sym-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_snr datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint sym (constructive): sweep over channel dimension
cls-disjoint-sym-channel-dim-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_channel_dim datasets/classification={{dataset}} sim.constructive=true {{ARGS}}


# Aggregate: disjoint sym (equalized) — layers + SNR
disjoint-sym: cls-disjoint-sym-layers cls-disjoint-sym-snr
# Aggregate: disjoint sym (constructive) — layers + SNR
disjoint-sym-c: cls-disjoint-sym-layers-c cls-disjoint-sym-snr-c
# Aggregate: disjoint sym — equalized + constructive (layers + SNR, no channel-dim)
disjoint-sym-full: disjoint-sym disjoint-sym-c
# Aggregate: all disjoint sym sweeps
disjoint-sym-all: disjoint-sym disjoint-sym-c

# ── Disjoint 4-SIM ────────────────────────────────────────────────────────────
# Tipology: constructive ∈ {false, true}
# Results go to results/classification/disjoint_4sim_*/equalized|constructive/

# Disjoint 4-SIM (equalized): sweep over SIM layers
cls-disjoint-4sim-layers *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers datasets/classification={{dataset}} {{ARGS}}

# Disjoint 4-SIM (equalized): sweep over SNR
cls-disjoint-4sim-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr datasets/classification={{dataset}} {{ARGS}}

# Disjoint 4-SIM (equalized): sweep over channel dimension
cls-disjoint-4sim-channel-dim *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_channel_dim datasets/classification={{dataset}} {{ARGS}}

# Disjoint 4-SIM (constructive): sweep over SIM layers
cls-disjoint-4sim-layers-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_layers datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint 4-SIM (constructive): sweep over SNR
cls-disjoint-4sim-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_4sim_snr datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Aggregate: disjoint 4-SIM (equalized) — layers + SNR
disjoint-4sim: cls-disjoint-4sim-layers cls-disjoint-4sim-snr
# Aggregate: disjoint 4-SIM (constructive) — layers + SNR
disjoint-4sim-c: cls-disjoint-4sim-layers-c cls-disjoint-4sim-snr-c
# Aggregate: disjoint 4-SIM — equalized + constructive (layers + SNR, no channel-dim)
disjoint-4sim-full: disjoint-4sim disjoint-4sim-c
# Aggregate: all disjoint 4-SIM sweeps
disjoint-4sim-all: disjoint-4sim disjoint-4sim-c

# ── Cross-architecture aggregates ─────────────────────────────────────────────

# Aggregate: all standard disjoint variants
disjoint: disjoint-asym disjoint-sym

# Aggregate: all constructive disjoint variants
disjoint-constructive: disjoint-asym-c disjoint-sym-c

# Layers sweep across all methods (joint + asym + sym + mono, excl. 4-SIM)
cls-all-layers: cls-joint-layers \
                cls-disjoint-asym-layers cls-disjoint-asym-layers-c cls-disjoint-asym-rx-layers-c \
                cls-disjoint-sym-layers cls-disjoint-sym-layers-c \
                cls-mono-layers cls-mono-layers-c cls-mono-rx-layers-c

# SNR sweep across all methods (joint + asym + sym + mono, excl. 4-SIM)
cls-all-snr: cls-joint-snr \
             cls-disjoint-asym-snr cls-disjoint-asym-snr-c cls-disjoint-asym-snr-rx-c \
             cls-disjoint-sym-snr cls-disjoint-sym-snr-c \
             cls-mono-snr cls-mono-snr-c cls-mono-rx-snr-c

# ── Mono-SIM ─────────────────────────────────────────────────────────────────
# TX side: equalized (G_T≈A + ZF/MMSE at RX) and constructive (H·G_T≈A, no G_eq)
# RX side: constructive only (G_R·H≈A, no equalized variant)
# Results go to results/classification/mono/{equalized|constructive}/{tx|rx}/

# Mono-SIM TX (equalized): sweep over SIM layers
cls-mono-layers *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers datasets/classification={{dataset}} {{ARGS}}

# Mono-SIM TX (equalized): sweep over SNR
cls-mono-snr *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr datasets/classification={{dataset}} {{ARGS}}

# Mono-SIM TX (constructive, loss ‖β H G_T − A‖²): sweep over SIM layers
cls-mono-layers-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers datasets/classification={{dataset}} sim.constructive=true sim.alignment_side=tx {{ARGS}}

# Mono-SIM TX (constructive): sweep over SNR
cls-mono-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr datasets/classification={{dataset}} sim.constructive=true sim.alignment_side=tx {{ARGS}}

# Mono-SIM RX (constructive, loss ‖β G_R H − A‖²): sweep over SIM layers
cls-mono-rx-layers-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_layers datasets/classification={{dataset}} sim=mono_rx {{ARGS}}

# Mono-SIM RX (constructive): sweep over SNR
cls-mono-rx-snr-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_snr datasets/classification={{dataset}} sim=mono_rx {{ARGS}}

# Aggregate: mono-SIM TX (equalized) — layers + SNR
mono: cls-mono-layers cls-mono-snr
# Aggregate: mono-SIM TX (constructive) — layers + SNR
mono-c: cls-mono-layers-c cls-mono-snr-c
# Aggregate: mono-SIM RX (constructive) — layers + SNR
mono-rx-c: cls-mono-rx-layers-c cls-mono-rx-snr-c
# Aggregate: mono-SIM — equalized/tx + constructive/tx (layers + SNR)
mono-full: mono mono-c
# Aggregate: all mono-SIM sweeps (TX equalized + TX constructive + RX constructive)
mono-all: mono mono-c mono-rx-c

# ── n_proto sweep (PPFE only) ─────────────────────────────────────────────────

# n_proto sweep: joint Dual-SIM
cls-n-proto-joint *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_joint datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: mono-SIM (equalized)
cls-n-proto-mono *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: mono-SIM (constructive)
cls-n-proto-mono-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_mono_c datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: disjoint asym (tx, equalized)
cls-n-proto-disjoint-asym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: disjoint asym (tx, constructive)
cls-n-proto-disjoint-asym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym_c datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: disjoint asym (rx, constructive)
cls-n-proto-disjoint-asym-rx-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_asym_rx_c datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: disjoint sym (equalized)
cls-n-proto-disjoint-sym *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_sym datasets/classification={{dataset}} {{ARGS}}

# n_proto sweep: disjoint sym (constructive)
cls-n-proto-disjoint-sym-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_n_proto_disjoint_sym_c datasets/classification={{dataset}} {{ARGS}}

# Aggregate: all n_proto (equalized) sweeps
n-proto: cls-n-proto-joint cls-n-proto-mono cls-n-proto-disjoint-asym cls-n-proto-disjoint-sym
# Aggregate: all n_proto (constructive) sweeps
n-proto-c: cls-n-proto-mono-c cls-n-proto-disjoint-asym-c cls-n-proto-disjoint-asym-rx-c cls-n-proto-disjoint-sym-c
# Aggregate: all n_proto sweeps
n-proto-all: n-proto n-proto-c

# ── Neural classifier ─────────────────────────────────────────────────────────

# Train a neural classifier (prerequisite for SIM experiments)
train-clf *ARGS:
    uv run python scripts/train_classifier.py datasets/classification={{dataset}} {{ARGS}}

# ── Image reconstruction (rec-) ───────────────────────────────────────────────
# Same SIM runners, sim/ and training/ configs and sweep structure as cls-; only
# the downstream task differs: the RX decodes the aligned embedding with a
# locally trained ConvVAE and is scored by PSNR. One job per strategy
# (alignment/reconstruction=linear|ppfe, both swept by the -layers recipes).
# Dataset resolves to config/datasets/reconstruction/<name>.yaml
#   (cifar10, cifar100, mnist, fashion_mnist):
#   just dataset=mnist rec-joint-layers
# Autoencoder pair (any trained variant under models/autoencoders/<dataset>/):
#   just rec-joint-layers reconstruction.tx_variant=base_mmd1 reconstruction.rx_variant=small_mmd1
# rec- recipes do NOT depend on train-ae: training is long (up to 200 epochs)
# and the pair is selectable — run `just train-ae` explicitly first.
# Results go to results/reconstruction/<subdir>/<tipology>/ (Parquet + PNG grid).

# Train ConvVAE autoencoders (base + small variants) for reconstruction alignment
train-ae *ARGS:
    uv run python scripts/train_autoencoder.py datasets/reconstruction={{dataset}} {{ARGS}}

# Extract paired TX/RX latents from two trained autoencoders → data/latents_reconstruction/
extract-latents *ARGS:
    uv run python scripts/extract_latents.py datasets/reconstruction={{dataset}} {{ARGS}}

# Linear alignment between two autoencoders, no SIM/channel (logged to W&B)
align-ae *ARGS:
    uv run python scripts/align_reconstruction.py datasets/reconstruction={{dataset}} {{ARGS}}

# Fast no-channel sweep of alignment hyperparameters (lmb, n_clusters, n_proto, pilots)
sweep-align *ARGS:
    uv run python scripts/sweep_alignment.py datasets/reconstruction={{dataset}} {{ARGS}}

# Single reconstruction run (no multirun; joint Dual-SIM + PPFE by default)
rec *ARGS:
    uv run python scripts/reconstruction.py datasets/reconstruction={{dataset}} {{ARGS}}

# Joint Dual-SIM reconstruction: sweep over SIM layers
rec-joint-layers *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_joint_layers datasets/reconstruction={{dataset}} {{ARGS}}

# Mono-SIM TX (equalized) reconstruction: sweep over SIM layers
rec-mono-layers *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_mono_layers datasets/reconstruction={{dataset}} {{ARGS}}

# Mono-SIM TX (constructive) reconstruction: sweep over SIM layers
rec-mono-layers-c *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_mono_layers datasets/reconstruction={{dataset}} sim.constructive=true {{ARGS}}

# Mono-SIM RX (constructive) reconstruction: sweep over SIM layers
rec-mono-rx-layers-c *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_mono_layers datasets/reconstruction={{dataset}} sim.constructive=true sim.alignment_side=rx {{ARGS}}

# Disjoint sym (equalized) reconstruction: sweep over SIM layers
rec-disjoint-sym-layers *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_disjoint_sym_layers datasets/reconstruction={{dataset}} {{ARGS}}

# Disjoint sym (constructive) reconstruction: sweep over SIM layers
rec-disjoint-sym-layers-c *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_disjoint_sym_layers datasets/reconstruction={{dataset}} sim.constructive=true {{ARGS}}

# Mono-SIM TX (constructive) reconstruction, legacy MNIST figure setup
# (L=25, M=64x64, base_mmd1 → small_mmd1, seed 42; Linear + PPFE). Dataset is fixed by the config.
rec-mono-c *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_mono_constructive {{ARGS}}

# Joint Dual-SIM reconstruction, legacy MNIST figure setup
# (L=25, M=64x64, ch=128, base_mmd1 → small_mmd1, seed 42; Linear + PPFE). Dataset is fixed by the config.
rec-joint *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_joint {{ARGS}}

# Disjoint sym (constructive) reconstruction, legacy MNIST figure setup
# (L=25, M=64x64, ch=128, base_mmd1 → small_mmd1, seed 42; Linear + PPFE). Dataset is fixed by the config.
rec-disjoint-sym-c *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_disjoint_sym_constructive {{ARGS}}

# Disjoint sym (constructive) reconstruction: sweep over channel_dim (20, 32, 64, 128),
# legacy MNIST figure setup otherwise. Dataset is fixed by the config.
rec-disjoint-sym-channel-dim-c *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_disjoint_sym_channel_dim {{ARGS}}

# Mono-SIM TX (constructive) reconstruction grid search: lr × scheduler × L (Linear only).
rec-mono-grid *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_mono_grid {{ARGS}}

# Disjoint sym (constructive) reconstruction grid search: shared TX/RX lr × scheduler × L (Linear only).
rec-disjoint-sym-grid *ARGS:
    uv run python scripts/reconstruction.py --config-name config_reconstruction_disjoint_sym_grid {{ARGS}}

# Final reconstruction runs: Mono TX (constructive), Disjoint sym (constructive) and Joint
# Dual-SIM on MNIST, Fashion-MNIST and CIFAR-10 (Linear + PPFE each), M=64x64, L=50, cosine
# lr schedule. lr from the grid searches: 0.1 (Mono, Disjoint sym), 0.03 (Joint).
# Results: results/reconstruction/{mono_final,disjoint_sym_final,joint_final}/ (+ .pt links).
# Finished runs are skipped, so re-running only fills in what is missing.
rec-final *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    common="training.scheduler=cosine sim.L=50 sim.M_int=64"
    for ds in mnist fashion_mnist cifar10; do
        just rec-mono-c datasets/reconstruction=$ds training.lr=0.1 $common sim.name=mono_final sim.result_subdir=mono_final {{ARGS}}
        just rec-disjoint-sym-c datasets/reconstruction=$ds alignment.lr_tx=0.1 alignment.lr_rx=0.1 $common sim.name=disjoint_sym_final sim.result_subdir=disjoint_sym_final {{ARGS}}
        just rec-joint datasets/reconstruction=$ds alignment.lr_tx=0.03 alignment.lr_rx=0.03 $common sim.name=joint_final sim.result_subdir=joint_final {{ARGS}}
    done

# Aggregate: reconstruction layers sweep across all methods
rec-all-layers: rec-joint-layers rec-mono-layers rec-mono-layers-c rec-mono-rx-layers-c \
                rec-disjoint-sym-layers rec-disjoint-sym-layers-c

# ── Single run (no multirun, useful for quick tests) ─────────────────────────

# Single run (no multirun, for quick tests)
cls *ARGS: train-clf
    uv run python scripts/classification.py datasets/classification={{dataset}} {{ARGS}}

# ── Unrolled GD (progressive block training) ──────────────────────────────────
# Trains K=100 layers in blocks of m, saving results at each block endpoint.
# Config: config/training/unrolled.yaml


# Unrolled GD: classic Adam baseline — sweeps over warm_start_type: none,only_semantic,average
cls-unrolled-baseline *ARGS: train-clf
    uv run python scripts/compute_unrolled_baseline_classification.py --config-name config_unrolled_baseline datasets/classification={{dataset}} {{ARGS}}

# Unrolled GD: classic Adam baseline — only_semantic warm-start
cls-unrolled-baseline-ws-sem *ARGS: train-clf
    uv run python scripts/compute_unrolled_baseline_classification.py --config-name config_unrolled_baseline datasets/classification={{dataset}} warmstart=only_semantic {{ARGS}}

# Unrolled GD: classic Adam baseline — average warm-start
cls-unrolled-baseline-ws-avg *ARGS: train-clf
    uv run python scripts/compute_unrolled_baseline_classification.py --config-name config_unrolled_baseline datasets/classification={{dataset}} warmstart=average {{ARGS}}

# Unrolled GD: joint Dual-SIM — PPFE only
cls-unrolled-ppfe *ARGS: train-clf
    uv run python scripts/unrolled_classification.py --config-name config_unrolled_joint datasets/classification={{dataset}} alignment/classification=ppfe {{ARGS}}

# Unrolled GD: joint Dual-SIM — Linear only
cls-unrolled-linear *ARGS: train-clf
    uv run python scripts/unrolled_classification.py --config-name config_unrolled_joint datasets/classification={{dataset}} alignment/classification=linear {{ARGS}}

# Unrolled GD: joint Dual-SIM — sweep h0_var × warm_start_type (PPFE, 8 multirun jobs)
cls-unrolled-joint *ARGS: train-clf
    uv run python scripts/unrolled_classification.py --config-name config_unrolled_joint datasets/classification={{dataset}} {{ARGS}}


# ── Unrolled GD for reconstruction (same pipeline, ReconstructionTask) ────────
# Config: config/config_unrolled_reconstruction.yaml — scores each unrolled step by the RX
# decoder's PSNR/MSE on the first reconstruction.unrolled_eval_images test images.
# Results: results/reconstruction/unrolled/

# Unrolled GD reconstruction: joint Dual-SIM — PPFE only
rec-unrolled-ppfe *ARGS:
    uv run python scripts/unrolled_reconstruction.py --config-name config_unrolled_reconstruction datasets/reconstruction={{dataset}} alignment/reconstruction=ppfe {{ARGS}}

# Unrolled GD reconstruction: joint Dual-SIM — Linear only
rec-unrolled-linear *ARGS:
    uv run python scripts/unrolled_reconstruction.py --config-name config_unrolled_reconstruction datasets/reconstruction={{dataset}} alignment/reconstruction=linear {{ARGS}}

# Unrolled GD reconstruction: joint Dual-SIM — sweep h0_var × warm_start_type (PPFE)
rec-unrolled-joint *ARGS:
    uv run python scripts/unrolled_reconstruction.py --config-name config_unrolled_reconstruction datasets/reconstruction={{dataset}} {{ARGS}}

# Aggregate: all unrolled GD experiments
unrolled: cls-unrolled-joint

# ── Convergence curves (save_curve=true) ─────────────────────────────────────
# Each recipe runs classification.py with training.save_curve=true, saving a
# per-iteration accuracy list (sim_accuracy_probe_curve / _neural_curve) in the
# parquet result.  Fixed geometry: L=10, M_int=32, channel_dim=20.
# All produce 200 eval points. Joint: k_T=k_R=100, max_iters=200 cycles.
# Override from CLI:  just cls-joint-curve training.max_iters=120

# Joint Dual-SIM convergence curve
cls-joint-curve *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_curve datasets/classification={{dataset}} {{ARGS}}

# Mono-SIM (equalized) convergence curve
cls-mono-curve *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_curve datasets/classification={{dataset}} {{ARGS}}

# Mono-SIM (constructive) convergence curve
cls-mono-curve-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_mono_curve datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint sym (equalized) convergence curve
cls-disjoint-sym-curve *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_curve datasets/classification={{dataset}} {{ARGS}}

# Disjoint sym (constructive) convergence curve
cls-disjoint-sym-curve-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_sym_curve datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Disjoint asym (tx, equalized) convergence curve
cls-disjoint-asym-curve *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_curve datasets/classification={{dataset}} {{ARGS}}

# Disjoint asym (tx, constructive) convergence curve
cls-disjoint-asym-curve-c *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_disjoint_asym_curve datasets/classification={{dataset}} sim.constructive=true {{ARGS}}

# Aggregate: joint + mono-constructive + disjoint-sym-constructive
curve: cls-joint-curve cls-mono-curve-c cls-disjoint-sym-curve-c
# curve: cls-joint-curve cls-mono-curve cls-disjoint-sym-curve cls-disjoint-asym-curve
# curve-c: cls-mono-curve-c cls-disjoint-sym-curve-c cls-disjoint-asym-curve-c
# curve-all: curve curve-c

# ── Joint LR grid search ─────────────────────────────────────────────────────
# Grid search over lr_tx × lr_rx for joint SIM.
# L=10, k_T=k_R=1, max_iters=3000 (1500 cycles), seed=42, snr=inf.
# 25 jobs (5×5 grid).

# Joint Dual-SIM LR grid search
cls-joint-lr-search *ARGS: train-clf
    uv run python scripts/classification.py --config-name config_joint_lr_search datasets/classification={{dataset}} {{ARGS}}

# ── Analysis ──────────────────────────────────────────────────────────────────

# Classification: publication-ready result plots
plot-cls:
    uv run python scripts/plot_results_classification.py

# Reconstruction figure: decoded test images per architecture/strategy (MNIST, Fashion-MNIST, CIFAR-10).
# Depends on rec-final, whose saved links (.pt) it decodes; finished runs are skipped.
plot-rec: rec-final
    uv run python scripts/plot_results_reconstruction.py

