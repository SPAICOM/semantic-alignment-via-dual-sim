<div align="center">

# Over-the-Air Semantic Alignment via Dual Stacked Intelligent Metasurfaces

[Mario Edoardo Pandolfo](https://scholar.google.com/citations?user=wAeScL8AAAAJ)<sup>1</sup>,
[Kyriakos Stylianopoulos](https://scholar.google.com/citations?user=42KAgRMAAAAJ)<sup>2</sup>,
[George C. Alexandropoulos](https://scholar.google.com/citations?user=3Ltyd9sAAAAJ)<sup>2</sup>,
[Paolo Di Lorenzo](https://scholar.google.com/citations?&user=VZYvspQAAAAJ)<sup>1</sup>

<sup>1</sup>Sapienza University of Rome · <sup>2</sup> National and Kapodistrian University of Athens

</div>

<h5 align="center">
    
[![ieee](https://img.shields.io/static/v1?label=IEEE+Paper&message=ID-HERE&color=0057b7&logo=ieee)](https://ieeexplore.ieee.org/document/ID-HERE)
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/REPO-NAME-HERE/blob/main/LICENSE)

 <br>

</h5>

---

> [!NOTE]
> **Abstract**
>Semantic communications enable task-oriented information exchange, but suffer from latent-space mismatch when transmitter and receiver employ heterogeneous machine learning models. Existing semantic alignment methods rely on digital processing, introducing computational and latency overhead. This paper proposes a dual stacked intelligent metasurfaces (dual-SIM) architecture that performs semantic alignment entirely in the wave domain. By jointly exploiting programmable metasurfaces at both ends of a link and the wireless channel, the proposed architecture realizes an end-to-end linear transformation between heterogeneous latent spaces. We show phase-only multilayer SIM can approximate both supervised linear aligners and zero-shot Parseval-frame equalizers. We investigate single-SIM, disjoint asymmetric and symmetric dual-SIM, and jointly optimized dual-SIM configurations, considering channel equalization and constructive channel exploitation. For joint design, we develop an alternating projected gradient algorithm with convergence guarantees to first-order stationary points. We further introduce a deep-unrolled network that unrolls gradient-based iterations to reduce computational cost of SIM reconfiguration under channel variations. Numerical results with heterogeneous vision transformer encoders show accurate semantic alignment from moderate to high signal-to-noise ratios. The joint and symmetric disjoint dual-SIM designs achieve the highest accuracy over compact channels, the joint design leads at shallow stack depths, and the advantage over single-SIM designs diminishes at equal depth.

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [`just`](https://github.com/casey/just)

Follow the installation instructions from their official documentation.

### Setup the development environment

From the project root, run:

```bash
just setup
```

The `setup` recipe will:

- Create the `.venv` virtual environment (if it does not exist)
- Install all project dependencies using `uv`

After the command completes, the development environment will be ready to use. 🚀

## Pipeline

Every experiment is launched through a [`just`](https://github.com/casey/just) recipe; run `just --list` to see all of them. The recipes below reproduce the experiments of the paper.

### Classification

Classification recipes run on CIFAR-10 by default; select another dataset with the `dataset` variable (e.g. `just dataset=cifar100 cls-all-layers`). Each recipe first trains the neural classifier (`train-clf`), which is skipped if a checkpoint already exists. Results are saved to `results/classification/`.

```bash
# SIM layers: sweep over the number of SIM layers for Joint Dual-SIM, Disjoint asym,
# Disjoint sym and Mono-SIM, in their equalized and constructive variants
just cls-all-layers

# SNR: sweep over the SNR for the same architectures
just cls-all-snr

# Inter-layer spacing: sweep over the spacing between SIM layers
# for Joint Dual-SIM and Disjoint sym (constructive)
just slayer
```

### Deep unrolling

Unrolled optimisation of the Joint Dual-SIM on random MIMO channels (classification), trained progressively for K = 1, 5, 10, 15, 30 unrolled steps. Results are saved to `results/classification/unrolled/`.

```bash
# Classic alternating-Adam baseline on the test channels, the reference for the unrolled model
just cls-unrolled-baseline

# Unrolled Joint Dual-SIM: sweep over warm start (none, only_semantic, average)
# and channel spread around H_0 (h0_var)
just cls-unrolled-joint
```

### Reconstruction

The reconstruction recipe uses the trained autoencoders in `models/autoencoders/` (train them with `just train-ae`). Results, together with the trained SIM links (`.pt`), are saved to `results/reconstruction/{mono,disjoint_sym,joint}/`. Finished runs are skipped, so re-running it only fills in what is missing.

```bash
# Mono-SIM, Disjoint sym and Joint Dual-SIM on MNIST, Fashion-MNIST and CIFAR-10,
# with Linear and PPFE alignment
just rec-all
```

### Plots

```bash
# Classification plots from the results already in results/classification/,
# saved to results/classification/plots/<dataset>/
just plot-cls

# Reconstruction figures (decoded test images per architecture and strategy),
# saved to results/reconstruction/plots/<dataset>/.
# Depends on rec-all, so any missing reconstruction run is launched first
just plot-rec
```

## Citation

If you find this code useful for your research, please consider citing the following paper:

```bibtex
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
