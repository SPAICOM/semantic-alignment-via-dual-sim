"""Original alignment accuracy — A_target applied directly, no SIM, no channel."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.linear_probe import clf_predict
from semantic_alignment.utils.utils import (
    a_inv_times_b,
    complex_compressed_tensor,
    decompress_complex_tensor,
)


def run_original_accuracy(
    dataloader: DataLoader,
    A_target: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf,
    device: torch.device,
) -> float:
    """
    Apply A_target to TX embeddings and evaluate with the linear probe.

    No SIM, no MIMO channel, no noise — represents the quality of the
    alignment strategy (Linear or PPFE) in the ideal software case.
    Equivalent to *Accuracy Original No Mimo* in the reference repo.

    Returns top-1 accuracy in [0, 1].
    """
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_real_batch, labels_batch in dataloader:
            x_real_batch = x_real_batch.to(device)
            x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
            x_white = a_inv_times_b(L_in, x_complex - mu_in)
            z_hat_complex = A_target @ x_white
            z_hat = (L_out @ z_hat_complex) + mu_out
            y_hat_real = decompress_complex_tensor(z_hat, device=device).T
            preds = clf_predict(clf, y_hat_real)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
    return (np.array(all_preds) == np.array(all_labels)).mean()
