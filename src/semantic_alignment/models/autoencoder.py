"""Convolutional beta-VAE — the TX/RX encoder pair for image-reconstruction alignment.

The classification pipeline downloads precomputed ViT embeddings from the
Semasia collection; no equivalent precomputed-embedding collection exists for
reconstruction, so this repo owns both the encoder and the decoder end-to-end.
Train two variants (``base``, ``small``) per dataset via
:func:`train_autoencoder` — mirroring ``vit_base``/``vit_small`` in the
classification pipeline — to get a semantically-similar-but-structurally-
different TX/RX pair whose bottleneck embeddings feed the SIM alignment.

Why a (beta-)VAE and not a plain autoencoder
---------------------------------------------
The SIM alignment pipeline whitens TX/RX embeddings with a Cholesky factor
estimated from a small pilot set and assumes a roughly Gaussian,
well-conditioned embedding distribution (see
:mod:`semantic_alignment.utils.alignment`). A plain undercomplete
autoencoder's latent geometry is unconstrained and can be arbitrarily
skewed/ill-conditioned, which destabilises that whitening step and the
least-squares estimate of the alignment matrix A. The KL term regularises the
latent towards an isotropic Gaussian, keeping the pilot covariance
well-conditioned and making nearby latents decode to visually similar images
— important since A is estimated from a handful of pilots and then applied to
embeddings it never saw. ``beta`` defaults small (1e-3) so reconstruction
fidelity still dominates the loss. The embedding used downstream is always
the deterministic posterior mean (:meth:`ConvVAE.embed`), never a stochastic
sample, so it behaves like any other fixed encoder in the alignment pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

MODELS_DIR = Path("models/autoencoders")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class PixelShuffleUpsample(nn.Module):
    """2x spatial upsampling via sub-pixel convolution (PixelShuffle).

    Drop-in replacement for ``ConvTranspose2d(stride=2)``: a regular conv
    produces ``4 * out_channels`` feature maps, which ``PixelShuffle(2)``
    rearranges into ``out_channels`` maps at double the spatial resolution.
    Transposed convs are prone to checkerboard artefacts (the kernel/stride
    overlap pattern shows up as a grid in the output); sub-pixel convolution
    avoids that by construction, giving sharper decoded detail.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Channels of the input feature map.
        out_channels : int
            Channels of the (2x larger) output feature map.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``(N, C_in, H, W)`` to ``(N, C_out, 2H, 2W)``."""
        return self.shuffle(self.conv(x))


class ResidualBlock(nn.Module):
    """Standard ResNet-style residual block at a fixed spatial resolution (stride=1).

    Adds capacity/gradient flow within a single resolution stage, entirely on
    one side of the bottleneck (either inside the encoder, before ``z`` is
    formed, or inside the decoder, after ``z`` is received) — unlike a U-Net
    skip connection, it never lets information bypass the transmitted
    embedding, so it doesn't break the "only z travels over the channel"
    semantic-communication assumption.
    """

    def __init__(self, channels: int):
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels (the block preserves shape).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        # Zero-init the last BN's scale so each block starts as an identity
        # function (standard ResNet stabilisation trick) — otherwise, stacked
        # on top of an already-deep encoder/decoder plus sigma_vae's log(MSE)
        # term, this reliably diverged to NaN within the first epoch.
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``act(x + F(x))`` with the same shape as ``x``."""
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + x)


class ConvVAE(nn.Module):
    """Convolutional beta-VAE with a configurable-width encoder/decoder.

    The encoder is a stack of stride-2 conv blocks (one per entry in
    ``channel_widths``); the decoder mirrors it with :class:`PixelShuffleUpsample`
    blocks and a final bilinear resize to the exact input resolution (robust
    to input sizes, like 28, that aren't a clean power of two after
    downsampling). Optionally, a :class:`ResidualBlock` follows each stage on
    both sides for extra capacity (see ``use_residual``).
    """

    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        latent_dim: int = 256,
        channel_widths: list[int] = (64, 128, 256),
        use_residual: bool = False,
    ):
        """
        Parameters
        ----------
        in_channels : int
            1 (grayscale) or 3 (RGB).
        image_size : tuple of int
            Input/output resolution ``(H, W)``.
        latent_dim : int, optional
            Bottleneck embedding dimensionality — this is the z_tx/z_rx size
            used by the SIM alignment.
        channel_widths : list of int, optional
            Output channels of each encoder conv block, in downsampling
            order; the decoder mirrors it in reverse.
        use_residual : bool, optional
            If ``True``, add a :class:`ResidualBlock` after each downsampling/
            upsampling stage (more capacity at the same width/depth budget).
            ``False`` (default) reproduces the original plain-conv architecture.
        """
        super().__init__()
        self.in_channels = in_channels
        self.image_size = tuple(image_size)
        self.latent_dim = latent_dim
        self.channel_widths = list(channel_widths)
        self.use_residual = use_residual

        enc_layers = []
        c_in = in_channels
        for c_out in self.channel_widths:
            enc_layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
            ]
            if use_residual:
                enc_layers.append(ResidualBlock(c_out))
            c_in = c_out
        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            feat = self.encoder(torch.zeros(1, in_channels, *self.image_size))
        self._feat_shape = tuple(feat.shape[1:])  # (C, H', W')
        feat_dim = feat.numel()

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feat_dim)

        widths = list(reversed(self.channel_widths))
        dec_layers = []
        for i in range(len(widths) - 1):
            dec_layers += [
                PixelShuffleUpsample(widths[i], widths[i + 1]),
                nn.BatchNorm2d(widths[i + 1]),
                nn.SiLU(inplace=True),
            ]
            if use_residual:
                dec_layers.append(ResidualBlock(widths[i + 1]))
        dec_layers.append(PixelShuffleUpsample(widths[-1], in_channels))
        self.decoder = nn.Sequential(*dec_layers)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map images to the approximate posterior ``(mu, logvar)``.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(N, C, H, W)``.

        Returns
        -------
        mu, logvar : torch.Tensor
            Each of shape ``(N, latent_dim)``.
        """
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z = mu + sigma * eps`` via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from latent vectors.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape ``(N, C, H, W)``, values in ``[0, 1]``.
        """
        h = self.fc_decode(z).view(-1, *self._feat_shape)
        h = self.decoder(h)
        h = F.interpolate(h, size=self.image_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out_conv(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass: encode, sample, decode.

        Returns
        -------
        x_hat, mu, logvar : torch.Tensor
            Reconstruction and posterior parameters.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic embedding (posterior mean) — the z_tx/z_rx fed to SIM alignment.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(N, latent_dim)``.
        """
        mu, _ = self.encode(x)
        return mu


def _mmd(z: torch.Tensor, prior_samples: torch.Tensor) -> torch.Tensor:
    """Squared Maximum Mean Discrepancy between two sets of samples (RBF kernel).

    Used by InfoVAE-style latent regularisation: matches the *aggregate*
    distribution of a batch of encoder outputs to the prior, rather than
    pulling every individual sample's posterior towards it the way KL does
    (see :class:`ConvVAEModule`'s docstring). Zero iff the two empirical
    distributions are identical (for a characteristic kernel like RBF).

    Parameters
    ----------
    z : torch.Tensor
        Batch of encoder outputs, shape ``(N, D)``.
    prior_samples : torch.Tensor
        Batch of samples drawn from the prior, shape ``(M, D)``.

    Returns
    -------
    torch.Tensor
        Scalar MMD² estimate.
    """

    def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pairwise RBF kernel matrix ``(len(a), len(b))`` with bandwidth ``σ² = D``."""
        dim = a.size(1)
        dist_sq = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-dist_sq / (2.0 * dim))

    k_zz = _kernel(z, z).mean()
    k_pp = _kernel(prior_samples, prior_samples).mean()
    k_zp = _kernel(z, prior_samples).mean()
    return k_zz + k_pp - 2 * k_zp


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss via early VGG16 features (ImageNet-pretrained, frozen).

    Only the first two conv blocks (through ``relu2_2``) are used — deeper
    VGG layers have receptive fields tuned for ~224x224 inputs and extract
    little meaningful structure at CIFAR/MNIST scale (28-32px); the early
    layers still capture edges/texture that MSE/L1 under-weight.
    """

    def __init__(self):
        """Load the frozen, ImageNet-pretrained VGG16 trunk (through ``relu2_2``)."""
        super().__init__()
        # Imported lazily: torchvision is only needed when perceptual_weight > 0.
        import torchvision.models as tv_models

        weights = tv_models.VGG16_Weights.IMAGENET1K_V1
        vgg = tv_models.vgg16(weights=weights).features[:9].eval()  # through relu2_2
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """MSE between VGG features of ``x_hat`` and ``x`` (images in ``[0, 1]``).

        Parameters
        ----------
        x_hat : torch.Tensor
            Reconstruction, shape ``(N, C, H, W)`` with ``C`` in ``{1, 3}``.
        x : torch.Tensor
            Target image, same shape as ``x_hat``.

        Returns
        -------
        torch.Tensor
            Scalar perceptual loss.
        """
        # VGG expects RGB: replicate grayscale across the three channels.
        if x_hat.shape[1] == 1:
            x_hat = x_hat.repeat(1, 3, 1, 1)
            x = x.repeat(1, 3, 1, 1)
        # ImageNet normalisation, matching the pretrained weights.
        x_hat_n = (x_hat - self.mean) / self.std
        x_n = (x - self.mean) / self.std
        return F.mse_loss(self.vgg(x_hat_n), self.vgg(x_n))


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class ConvVAEModule(L.LightningModule):
    """Lightning wrapper around :class:`ConvVAE` for training.

    Loss is ``(mse_term + l1_weight * l1 + ssim_weight * (1 - SSIM)) + beta(epoch) * KL``.
    MSE+L1 combined tends to produce sharper reconstructions than MSE alone
    (L1's gradient doesn't vanish near-zero error the way MSE's does); the
    SSIM term additionally rewards structural/edge similarity, which pixel-wise
    losses alone under-weight. ``beta`` ramps linearly from 0 up to its target
    value over ``beta_warmup_epochs`` — letting the model first learn to
    reconstruct well before the KL term starts pulling the latent toward the
    prior avoids the KL dominating early training (which is what capped
    quality at beta=1e-3 constant). ``val_mse`` (pure pixel MSE, undiluted by
    L1/SSIM/KL/sigma-calibration) is logged separately — that's what early
    stopping/checkpointing monitor, since it's the metric that maps directly
    to PSNR.

    Two opt-in knobs address the reconstruction-vs-KL trade-off more directly
    than beta/warmup alone (both default to a no-op so existing configs are
    unaffected unless set explicitly):

    - ``free_bits`` (nats/dim, default 0): floors each latent dimension's KL
      (averaged over the batch) at this value before summing, so dimensions
      already within budget aren't penalised further. Standard KL punishes
      *every* dimension for carrying *any* sample-specific information,
      competing with reconstruction from the first gradient step; free bits
      only kicks in once a dimension exceeds the budget.
    - ``sigma_vae`` (bool, default False): replaces the raw MSE term with the
      analytically-calibrated Gaussian NLL (Rybkin et al., "Simple and
      Effective VAE Training with Calibrated Decoders"). Plain MSE implicitly
      assumes a fixed decoder variance of 1, which arbitrarily fixes the
      reconstruction/KL balance; substituting the closed-form optimal
      per-batch variance (= MSE itself) into the Gaussian NLL collapses to a
      loss proportional to ``log(MSE)``, which self-balances against KL
      instead of relying purely on a hand-tuned beta.
    - ``mmd_weight`` (float, default 0): InfoVAE-style regulariser — adds
      ``mmd_weight * MMD²(batch of mu, prior samples)`` to the loss (see
      :func:`_mmd`), *independent* of ``beta * KL`` (both can be used
      together, or ``beta=0`` to use MMD alone). Unlike KL, which pulls every
      individual sample's posterior towards the prior (competing with
      reconstruction from the first gradient step — this is why increasing
      beta alone degrades reconstruction so much), MMD only constrains the
      *aggregate* distribution of a batch to resemble the prior, which is the
      statistic that actually matters for the pilot-based whitening step —
      individual samples stay free to be as informative as they need to be.
      Ramps in over ``mmd_warmup_epochs`` for the same reason as ``beta``.

    Two more knobs target detail *late* in training, after the model has
    already learned coarse structure (both default to a no-op):

    - ``perceptual_weight`` (float, default 0): adds
      ``perceptual_weight * VGGPerceptualLoss`` (see :class:`VGGPerceptualLoss`).
      Held at 0 until ``perceptual_start_epoch``, then ramps to its target over
      ``perceptual_warmup_epochs`` — applying it from epoch 0, on a near-random
      output, wastes it on noise the same way early SSIM/MMD did; it's most
      useful once MSE/L1 have already established coarse structure.
    - ``use_cosine_schedule`` (bool, default False): cosine-anneals the Adam
      LR from ``lr`` down to ``lr * 0.01`` over ``max_epochs``. A constant LR
      keeps gradient steps coarse even late in training, when fine detail
      needs smaller, more careful updates.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        latent_dim: int = 256,
        channel_widths: list[int] = (64, 128, 256),
        beta: float = 1e-4,
        beta_warmup_epochs: int = 30,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.0,
        ssim_warmup_epochs: int = 10,
        free_bits: float = 0.0,
        sigma_vae: bool = False,
        mmd_weight: float = 0.0,
        mmd_warmup_epochs: int = 10,
        use_residual: bool = False,
        perceptual_weight: float = 0.0,
        perceptual_warmup_epochs: int = 10,
        perceptual_start_epoch: int = 0,
        use_cosine_schedule: bool = False,
        max_epochs: int = 100,
        lr: float = 1e-3,
    ):
        """
        Parameters
        ----------
        in_channels, image_size, latent_dim, channel_widths, use_residual
            Architecture arguments forwarded to :class:`ConvVAE`.
        beta : float, optional
            Target KL weight, reached after ``beta_warmup_epochs``.
        beta_warmup_epochs : int, optional
            Epochs of the linear KL-weight ramp.
        l1_weight : float, optional
            Weight of the L1 reconstruction term.
        ssim_weight, ssim_warmup_epochs : float, int, optional
            Target weight and ramp length of the ``1 - SSIM`` term.
        free_bits : float, optional
            Per-dimension KL floor in nats (0 disables).
        sigma_vae : bool, optional
            Use the calibrated-decoder ``log(MSE)`` loss instead of raw MSE.
        mmd_weight, mmd_warmup_epochs : float, int, optional
            Target weight and ramp length of the InfoVAE MMD² regulariser.
        perceptual_weight, perceptual_warmup_epochs, perceptual_start_epoch : float, int, int
            Target weight, ramp length and first active epoch of the VGG
            perceptual loss.
        use_cosine_schedule : bool, optional
            Cosine-anneal the learning rate over ``max_epochs``.
        max_epochs : int, optional
            Schedule horizon for the cosine annealing.
        lr : float, optional
            Initial Adam learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvVAE(
            in_channels, image_size, latent_dim, list(channel_widths), use_residual
        )
        # Only instantiate (and download the pretrained weights for) VGG when
        # actually requested, so perceptual_weight=0 (the default) has zero
        # extra cost.
        self.perceptual_loss = VGGPerceptualLoss() if perceptual_weight > 0 else None

    def forward(self, x):
        """Run the VAE; returns ``(x_hat, mu, logvar)`` (see :meth:`ConvVAE.forward`)."""
        return self.model(x)

    @staticmethod
    def _linear_warmup(
        target: float, warmup_epochs: int, current_epoch: int, start_epoch: int = 0
    ) -> float:
        """Linear ramp: 0 until start_epoch, then 0 -> target over warmup_epochs."""
        if current_epoch < start_epoch:
            return 0.0
        elapsed = current_epoch - start_epoch
        if warmup_epochs <= 0:
            return target
        frac = min(1.0, elapsed / warmup_epochs)
        return target * frac

    def _current_beta(self) -> float:
        """Warm-up-scheduled KL weight (``beta``) for the current epoch."""
        return self._linear_warmup(
            self.hparams.beta, self.hparams.beta_warmup_epochs, self.current_epoch
        )

    def _current_ssim_weight(self) -> float:
        """Warm-up-scheduled SSIM-term weight for the current epoch."""
        return self._linear_warmup(
            self.hparams.ssim_weight, self.hparams.ssim_warmup_epochs, self.current_epoch
        )

    def _current_mmd_weight(self) -> float:
        """Warm-up-scheduled MMD-term weight for the current epoch."""
        return self._linear_warmup(
            self.hparams.mmd_weight, self.hparams.mmd_warmup_epochs, self.current_epoch
        )

    def _current_perceptual_weight(self) -> float:
        """Warm-up-scheduled perceptual-loss weight for the current epoch."""
        return self._linear_warmup(
            self.hparams.perceptual_weight,
            self.hparams.perceptual_warmup_epochs,
            self.current_epoch,
            start_epoch=self.hparams.perceptual_start_epoch,
        )

    def _loss(self, batch):
        """Compute the full training objective and all logged components.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Single-element tuple ``(x,)`` with images ``(N, C, H, W)`` in ``[0, 1]``.

        Returns
        -------
        tuple
            ``(loss, recon, mse, kl, beta, ssim_val, mmd, mmd_w, perceptual,
            perceptual_w)``: total loss, reconstruction part, raw pixel MSE,
            KL, then each regulariser's value and its current weight.
        """
        (x,) = batch
        x_hat, mu, logvar = self.model(x)
        mse = F.mse_loss(x_hat, x, reduction="mean")
        l1 = F.l1_loss(x_hat, x, reduction="mean")
        # SSIM's local-window statistics are numerically unstable when x_hat is
        # still near-random (start of training) — ramp its weight in gradually,
        # same rationale as the beta (KL) warm-up. ssim_val is always computed
        # for logging; only its *contribution to the loss* is warmed up, and
        # since ssim_val is finite even when unweighted, `0 * finite = 0` exactly
        # (no NaN leaks through while the weight ramp is at 0).
        ssim_val = ssim(x_hat, x, data_range=1.0)
        ssim_w = self._current_ssim_weight()

        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(x_hat, x)
        else:
            perceptual = torch.zeros((), device=x.device)
        perceptual_w = self._current_perceptual_weight()

        # sigma-VAE: loss ∝ log(MSE) instead of raw MSE (see class docstring).
        # `mse` itself (undiluted) is still returned/logged for PSNR purposes.
        mse_term = 0.5 * torch.log(mse + 1e-8) if self.hparams.sigma_vae else mse
        recon = (
            mse_term
            + self.hparams.l1_weight * l1
            + ssim_w * (1 - ssim_val)
            + perceptual_w * perceptual
        )

        # Free bits: floor per-dimension KL (averaged over the batch) at
        # hparams.free_bits nats before summing. KL_i is always >= 0, so
        # free_bits=0 is an exact no-op (identical to the plain KL below).
        kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
        kl = torch.clamp(kl_per_dim, min=self.hparams.free_bits).sum()

        # MMD (InfoVAE): always computed (cheap) for logging visibility even
        # when mmd_weight=0, mirroring ssim_val above.
        mmd = _mmd(mu, torch.randn_like(mu))
        mmd_w = self._current_mmd_weight()

        beta = self._current_beta()
        loss = recon + beta * kl + mmd_w * mmd
        return loss, recon, mse, kl, beta, ssim_val, mmd, mmd_w, perceptual, perceptual_w

    def training_step(self, batch, _):
        """Optimise the total loss and log all components per epoch."""
        loss, recon, mse, kl, beta, ssim_val, mmd, mmd_w, perceptual, perceptual_w = self._loss(
            batch
        )
        self.log_dict(
            {
                "train_loss": loss,
                "train_recon": recon,
                "train_mse": mse,
                "train_kl": kl,
                "train_ssim": ssim_val,
                "train_mmd": mmd,
                "train_perceptual": perceptual,
                "beta": beta,
                "mmd_weight": mmd_w,
                "perceptual_weight": perceptual_w,
                "lr": self.optimizers().param_groups[0]["lr"],
            },
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, _):
        """Log validation losses; ``val_mse`` drives early stopping / checkpointing."""
        loss, recon, mse, kl, beta, ssim_val, mmd, mmd_w, perceptual, perceptual_w = self._loss(
            batch
        )
        self.log_dict(
            {
                "val_loss": loss,
                "val_recon": recon,
                "val_mse": mse,
                "val_kl": kl,
                "val_ssim": ssim_val,
                "val_mmd": mmd,
                "val_perceptual": perceptual,
            },
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def configure_optimizers(self):
        """Adam, optionally with per-epoch cosine annealing down to ``0.01 * lr``."""
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if not self.hparams.use_cosine_schedule:
            return opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def _to_wandb_pair(original: torch.Tensor, reconstruction: torch.Tensor, caption: str):
    """Concatenate ``original | reconstruction`` side by side into one ``wandb.Image``.

    Logging them as two separate images (even adjacent, captioned) is easy to
    misread in W&B's gallery view — the crisp originals get mistaken for the
    reconstruction quality. One combined image removes the ambiguity.
    """
    import numpy as np

    import wandb

    orig = original.detach().clamp(0, 1).cpu().numpy()
    recon = reconstruction.detach().clamp(0, 1).cpu().numpy()
    sep = np.ones((orig.shape[0], orig.shape[1], 2), dtype=orig.dtype)
    paired = np.concatenate([orig, sep, recon], axis=2)  # (C, H, 2W+2)
    arr = paired[0] if paired.shape[0] == 1 else paired.transpose(1, 2, 0)
    return wandb.Image(arr, caption=caption)


class ReconstructionImageLogger(L.pytorch.callbacks.Callback):
    """Logs a fixed set of validation images (original + reconstruction) to W&B every epoch.

    No-ops when the trainer isn't using a :class:`WandbLogger` (e.g. during
    tests or ``logger=None`` runs), so it's always safe to attach.
    """

    def __init__(self, val_images: torch.Tensor, num_images: int = 5):
        """
        Parameters
        ----------
        val_images : torch.Tensor
            Validation images ``(N, C, H, W)``; the first ``num_images`` are logged.
        num_images : int, optional
            Number of fixed images to track across epochs.
        """
        super().__init__()
        self.val_images = val_images[:num_images]

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: ConvVAEModule) -> None:
        """Reconstruct the tracked images and log ``original | reconstruction`` pairs."""
        if trainer.sanity_checking or not isinstance(trainer.logger, WandbLogger):
            return

        x = self.val_images.to(pl_module.device)
        pl_module.eval()
        with torch.no_grad():
            x_hat, _, _ = pl_module.model(x)
        pl_module.train()

        images = [
            _to_wandb_pair(orig, recon, f"val_{i}: original | reconstruction")
            for i, (orig, recon) in enumerate(zip(x, x_hat))
        ]
        trainer.logger.experiment.log(
            {"val_reconstructions": images, "epoch": trainer.current_epoch}
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ckpt_path(dataset: str, variant: str, seed: int) -> Path:
    """``models/autoencoders/{dataset}/{variant}/seed{seed}.pt``"""
    return MODELS_DIR / dataset / variant / f"seed{seed}.pt"


def save_autoencoder(
    model: ConvVAE, dataset: str, variant: str, seed: int, training_meta: dict | None = None
) -> Path:
    """Save model weights and architecture metadata to a ``.pt`` checkpoint.

    Parameters
    ----------
    model : ConvVAE
        Trained model to persist.
    dataset : str
        Dataset name, used as a directory component in the checkpoint path.
    variant : str
        Encoder variant name (``"base"`` or ``"small"``).
    seed : int
        Training seed, used as the filename stem.
    training_meta : dict, optional
        Training-time hyperparameters (``beta``, ``sigma_vae``, ``mmd_weight``,
        ...) that don't affect the architecture but are useful downstream for
        e.g. tagging result filenames by regularisation scheme (see
        :func:`regularization_tag`). Stored as-is; ``{}`` if omitted.

    Returns
    -------
    Path
        Absolute path to the written checkpoint file.
    """
    path = _ckpt_path(dataset, variant, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": model.in_channels,
            "image_size": model.image_size,
            "latent_dim": model.latent_dim,
            "channel_widths": model.channel_widths,
            "use_residual": model.use_residual,
            "training_meta": training_meta or {},
        },
        path,
    )
    return path


def load_autoencoder(dataset: str, variant: str, seed: int, device: torch.device) -> ConvVAE:
    """Load a :class:`ConvVAE` from a checkpoint written by :func:`save_autoencoder`.

    Parameters
    ----------
    dataset : str
        Dataset name used to locate the checkpoint directory.
    variant : str
        Encoder variant name used to locate the checkpoint directory.
    seed : int
        Training seed used as the checkpoint filename.
    device : torch.device
        Device on which to load the model weights.

    Returns
    -------
    ConvVAE
        Model in eval mode on ``device``.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist, with a ``just train-ae`` hint.
    """
    path = _ckpt_path(dataset, variant, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"No autoencoder at {path}. "
            f"Run: just train-ae dataset={dataset} autoencoder={variant} training.seed={seed}"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ConvVAE(
        ckpt["in_channels"],
        ckpt["image_size"],
        ckpt["latent_dim"],
        ckpt["channel_widths"],
        ckpt.get("use_residual", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_autoencoder_training_meta(dataset: str, variant: str, seed: int) -> dict:
    """Read back the ``training_meta`` dict saved by :func:`save_autoencoder`, without loading weights.

    Parameters
    ----------
    dataset : str
        Dataset name used to locate the checkpoint directory.
    variant : str
        Encoder variant name used to locate the checkpoint directory.
    seed : int
        Training seed used as the checkpoint filename.

    Returns
    -------
    dict
        The ``training_meta`` dict passed to :func:`save_autoencoder`
        (``{}`` for checkpoints saved before this field existed).
    """
    path = _ckpt_path(dataset, variant, seed)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("training_meta", {})


def regularization_tag(meta: dict) -> str:
    """Classify a training-meta dict into a short regularisation-scheme label.

    Used to tag result filenames (see :class:`semantic_alignment.evaluation.tasks.ReconstructionTask`)
    so runs with different latent regularisation don't silently overwrite
    each other.

    Parameters
    ----------
    meta : dict
        A ``training_meta`` dict as returned by :func:`load_autoencoder_training_meta`.

    Returns
    -------
    str
        One of ``"MMD-InfoVAE"``, ``"sigmaVAE"``, ``"betaVAE"``, ``"AE"``.
    """
    if meta.get("mmd_weight", 0.0) > 0:
        return "MMD-InfoVAE"
    if meta.get("beta", 0.0) > 0:
        return "sigmaVAE" if meta.get("sigma_vae", False) else "betaVAE"
    return "AE"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_autoencoder(
    images: torch.Tensor,
    in_channels: int,
    image_size: tuple[int, int],
    latent_dim: int = 256,
    channel_widths: list[int] = (64, 128, 256),
    beta: float = 1e-4,
    beta_warmup_epochs: int = 30,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.0,
    ssim_warmup_epochs: int = 10,
    free_bits: float = 0.0,
    sigma_vae: bool = False,
    mmd_weight: float = 0.0,
    mmd_warmup_epochs: int = 10,
    use_residual: bool = False,
    perceptual_weight: float = 0.0,
    perceptual_warmup_epochs: int = 10,
    perceptual_start_epoch: int = 0,
    use_cosine_schedule: bool = False,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    device: torch.device | str = "cpu",
    seed: int = 42,
    logger=None,
    gradient_clip_val: float | None = 1.0,
) -> ConvVAE:
    """Train a :class:`ConvVAE` with early stopping and return the best model.

    Parameters
    ----------
    images : torch.Tensor
        Training images of shape ``(N, C, H, W)``, values in ``[0, 1]``.
    in_channels : int
        1 (grayscale) or 3 (RGB).
    image_size : tuple of int
        ``(H, W)`` of ``images``.
    latent_dim : int, optional
        Bottleneck embedding dimensionality.
    channel_widths : list of int, optional
        Encoder conv-block widths (decoder mirrors it in reverse).
    beta : float, optional
        Target KL weight (beta-VAE), reached at the end of the warm-up.
    beta_warmup_epochs : int, optional
        Epochs over which ``beta`` ramps linearly from 0 to its target value.
        ``0`` disables warm-up (``beta`` is constant from the start).
    l1_weight : float, optional
        Weight of the L1 term added to the MSE reconstruction loss.
    ssim_weight : float, optional
        Target weight of the ``(1 - SSIM)`` term added to the reconstruction
        loss; ``0`` disables it.
    ssim_warmup_epochs : int, optional
        Epochs over which ``ssim_weight`` ramps linearly from 0 to its target
        value. SSIM's local-window statistics are numerically unstable on the
        near-random output of an untrained model — applying it at full weight
        from epoch 0 reliably produced NaN within the first epoch.
    free_bits : float, optional
        Per-dimension KL floor in nats; ``0`` (default) is a no-op (plain KL).
        Set to e.g. ``0.5`` to stop penalising a latent dimension once its
        (batch-averaged) KL drops below this budget.
    sigma_vae : bool, optional
        If ``True``, use the analytically-calibrated Gaussian NLL
        (``0.5 * log(MSE)``) instead of raw MSE for the reconstruction term's
        MSE component; ``False`` (default) keeps plain MSE.
    mmd_weight : float, optional
        InfoVAE-style MMD regularisation weight; ``0`` (default) disables it.
        Independent of ``beta``/KL — matches the aggregate batch distribution
        to the prior instead of each individual sample's posterior.
    mmd_warmup_epochs : int, optional
        Epochs over which ``mmd_weight`` ramps linearly from 0 to its target.
    use_residual : bool, optional
        If ``True``, add a residual block after each encoder/decoder stage
        (see :class:`ConvVAE`); ``False`` (default) is the plain architecture.
    perceptual_weight : float, optional
        Target weight of a VGG-feature perceptual loss (see :class:`VGGPerceptualLoss`);
        ``0`` (default) disables it (and skips loading VGG entirely).
    perceptual_warmup_epochs : int, optional
        Epochs over which ``perceptual_weight`` ramps from 0 to its target,
        starting at ``perceptual_start_epoch``.
    perceptual_start_epoch : int, optional
        Epoch at which the perceptual-loss ramp begins; ``0`` before this.
        Applying it on the near-random output of an untrained model wastes it
        on noise — most useful once MSE/L1 have already learned coarse structure.
    use_cosine_schedule : bool, optional
        If ``True``, cosine-anneal the Adam LR from ``lr`` down to ``lr * 0.01``
        over ``max_epochs``; ``False`` (default) keeps LR constant.
    lr : float, optional
        Adam learning rate.
    max_epochs : int, optional
        Maximum number of training epochs.
    patience : int, optional
        Early-stopping patience in epochs (stops when ``val_mse`` does not
        improve for this many consecutive epochs).
    batch_size : int, optional
        Mini-batch size for both train and validation loaders.
    val_fraction : float, optional
        Fraction of ``images`` held out as a validation set.
    device : torch.device or str, optional
        Target device; GPU is selected automatically when available.
    seed : int, optional
        Global seed passed to Lightning, the train/val split, and DataLoaders.
    logger : lightning logger, optional
        A Lightning-compatible logger such as ``WandbLogger``; ``None``
        disables external metric logging. When it *is* a ``WandbLogger``, 5
        fixed validation images (original | reconstruction, paired side by
        side) are logged under ``val_reconstructions`` every epoch, and final
        ``val_mse``/``val_psnr`` (from the best checkpoint, over the whole
        validation split) are logged once training completes.
    gradient_clip_val : float or None, optional
        Global gradient-norm clip applied every step; ``None`` disables it.
        Larger-capacity variants are more prone to early-training instability
        (exploding ``logvar`` in the reparameterization trick before the KL
        term has ramped up); clipping is a cheap safeguard against NaNs.

    Returns
    -------
    ConvVAE
        Best-checkpoint model in eval mode.
    """
    L.seed_everything(seed)

    dataset = TensorDataset(images.float())
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    val_images = torch.stack([val_ds[i][0] for i in range(min(5, len(val_ds)))])

    module = ConvVAEModule(
        in_channels,
        image_size,
        latent_dim,
        list(channel_widths),
        beta,
        beta_warmup_epochs,
        l1_weight,
        ssim_weight,
        ssim_warmup_epochs,
        free_bits,
        sigma_vae,
        mmd_weight,
        mmd_warmup_epochs,
        use_residual,
        perceptual_weight,
        perceptual_warmup_epochs,
        perceptual_start_epoch,
        use_cosine_schedule,
        max_epochs,
        lr,
    )

    callbacks = [
        # Monitor pure pixel MSE (val_mse) — not val_recon (MSE+L1) or val_loss
        # (+beta*KL) — since that's the metric that maps directly to PSNR and
        # isn't diluted by the L1/KL terms or the beta warm-up ramp.
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_mse",
            patience=patience,
            mode="min",
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_mse",
            mode="min",
            save_top_k=1,
        ),
        ReconstructionImageLogger(val_images),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        module = ConvVAEModule.load_from_checkpoint(best_ckpt)

    model = module.model.eval()

    if logger is not None:
        model_device = next(model.parameters()).device
        se_sum, n_pixels = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(model_device)
                x_hat, _, _ = model(x)
                se_sum += F.mse_loss(x_hat, x, reduction="sum").item()
                n_pixels += x.numel()
        val_mse = se_sum / n_pixels
        val_psnr = 10.0 * math.log10(1.0 / val_mse) if val_mse > 0 else float("inf")
        logger.log_metrics({"val_mse": val_mse, "val_psnr": val_psnr})

    return model


if __name__ == "__main__":
    import torch

    model = ConvVAE(
        in_channels=3, image_size=(32, 32), latent_dim=256, channel_widths=[64, 128, 256]
    )
    x = torch.rand(8, 3, 32, 32)
    x_hat, mu, logvar = model(x)
    z = model.embed(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvVAE   : x={x.shape} -> x_hat={x_hat.shape}, mu={mu.shape}, logvar={logvar.shape}")
    print(f"embed     : z={z.shape} (deterministic, used for SIM alignment)")
    print(f"Parameters: {n_params:,}")

    model_mnist = ConvVAE(
        in_channels=1, image_size=(28, 28), latent_dim=128, channel_widths=[32, 64, 128]
    )
    x_m = torch.rand(4, 1, 28, 28)
    x_hat_m, _, _ = model_mnist(x_m)
    print(f"ConvVAE (mnist, non-power-of-two input): x={x_m.shape} -> x_hat={x_hat_m.shape}")
