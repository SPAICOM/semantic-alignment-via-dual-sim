"""Tests for models/autoencoder.py: ConvVAE architecture, persistence, and tags."""

import pytest
import torch

import semantic_alignment.models.autoencoder as ae
from semantic_alignment.models.autoencoder import (
    ConvVAE,
    load_autoencoder,
    load_autoencoder_training_meta,
    regularization_tag,
    save_autoencoder,
)


@pytest.mark.parametrize(
    "channels,size,use_residual",
    [(3, (32, 32), False), (1, (28, 28), True)],  # 28 is not a clean power-of-two downsample
)
def test_convvae_shapes(channels, size, use_residual):
    model = ConvVAE(channels, size, latent_dim=16, channel_widths=[4, 8], use_residual=use_residual)
    x = torch.rand(3, channels, *size)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape == logvar.shape == (3, 16)
    assert 0.0 <= x_hat.min() and x_hat.max() <= 1.0


def test_embed_is_posterior_mean():
    model = ConvVAE(3, (8, 8), latent_dim=6, channel_widths=[4]).eval()
    x = torch.rand(2, 3, 8, 8)
    torch.testing.assert_close(model.embed(x), model.encode(x)[0])
    torch.testing.assert_close(model.embed(x), model.embed(x))


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "MODELS_DIR", tmp_path)
    model = ConvVAE(1, (8, 8), latent_dim=6, channel_widths=[4, 8], use_residual=True).eval()
    meta = {"beta": 0.05, "sigma_vae": True}
    path = save_autoencoder(model, "ds", "small", 7, training_meta=meta)
    assert path == tmp_path / "ds" / "small" / "seed7.pt"

    loaded = load_autoencoder("ds", "small", 7, torch.device("cpu"))
    x = torch.rand(2, 1, 8, 8)
    torch.testing.assert_close(loaded.embed(x), model.embed(x))
    assert loaded.use_residual and loaded.latent_dim == 6
    assert load_autoencoder_training_meta("ds", "small", 7) == meta


def test_load_missing_checkpoint_hints_train_ae(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="just train-ae"):
        load_autoencoder("ds", "base", 1, torch.device("cpu"))


@pytest.mark.parametrize(
    "meta,tag",
    [
        ({}, "AE"),
        ({"beta": 0.0}, "AE"),
        ({"beta": 1e-4}, "betaVAE"),
        ({"beta": 0.05, "sigma_vae": True}, "sigmaVAE"),
        ({"beta": 0.05, "sigma_vae": True, "mmd_weight": 1.0}, "MMD-InfoVAE"),
    ],
)
def test_regularization_tag(meta, tag):
    assert regularization_tag(meta) == tag
