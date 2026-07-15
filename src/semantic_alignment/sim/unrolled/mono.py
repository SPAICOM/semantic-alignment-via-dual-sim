import torch

from ._base import MonoUnrolledSCA
from ._utils import _nmse, _normalize


class MonoSIMUnrolledSCA(MonoUnrolledSCA):
    """Deep unrolled SCA for a single SIM (TX or RX).

    Inherits all W buffers, step module, cascade, and innovation machinery
    from :class:`MonoUnrolledSCA`.  Adds only the K-step forward loop.

    Parameters
    ----------
    sim : MonoSIMoptimizerTorch
        ``sim.side`` selects TX (Y·W) or RX (W·Y) cascade.
    K : int  number of unrolled layers
    (all remaining kwargs forwarded to :class:`MonoUnrolledSCA`)
    """

    # __init__ inherited from MonoUnrolledSCA

    def forward(
        self,
        A,
        left_mat=None,
        right_mat=None,
        B: int = 1,
        record: bool = False,
        create_graph=None,
        return_phases: bool = False,
        step_callback=None,
        k_max: int | None = None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
    ):
        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.K
        analytic = self.innovation_mode == "analytic"
        deep = supervision == "all"
        if deep and not (0.0 < weight_ratio <= 1.0):
            raise ValueError(
                f"weight_ratio must be in (0, 1] when supervision='all'; got {weight_ratio}."
            )
        gamma = weight_ratio ** (1.0 / (k_run - 1)) if (weight_ratio < 1.0 and k_run > 1) else 1.0

        dev = next(self.buffers()).device
        A = A.to(dev)
        A2 = (A.conj() * A).sum((-2, -1)).real
        if left_mat is not None:
            left_mat = left_mat.to(dev)
        if right_mat is not None:
            right_mat = right_mat.to(dev)

        if self.warm_start_t > 0:
            phases = self._warm_start(A, left_mat, B)
        else:
            phases = [torch.zeros(B, m, device=dev) for m in self.sizes]
        v_mom = self.step.init_momentum(B, self.sizes, dev)

        curves = []

        for k in range(k_run):
            if analytic:
                _, g_list = self._innovation_analytic(A, A2, phases, left_mat, right_mat)
            else:
                _, g_list = self._innovation_autograd(
                    A, A2, phases, left_mat, right_mat, create_graph
                )

            g_list = _normalize(g_list, self.innovation_norm, self.eps)
            phases, v_mom = self.step.apply(phases, g_list, v_mom, k)

            if deep or k == k_run - 1:
                curves.append(_nmse(self._cascade(phases, left_mat, right_mat), A, A2, self.eps))

            if step_callback is not None and not self.training:
                step_callback(k, phases)

        if deep:
            if gamma < 1.0:
                w = torch.tensor(
                    [gamma ** (k_run - 1 - k) for k in range(k_run)],
                    device=dev,
                    dtype=torch.float32,
                )
                loss = (w.unsqueeze(-1) * torch.stack(curves)).sum(0).mean() / w.sum()
            else:
                loss = torch.stack(curves).mean()
        else:
            loss = curves[-1].mean() if curves else torch.tensor(0.0, device=dev)

        if return_phases:
            return loss, curves, [p.detach() for p in phases]
        return loss, curves

    @torch.no_grad()
    def eval_curve(
        self, A, left_mat=None, right_mat=None, k_max=None, supervision="final", weight_ratio=1.0
    ):
        self.eval()
        _, curves = self.forward(
            A,
            left_mat=left_mat,
            right_mat=right_mat,
            B=A.shape[0],
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            weight_ratio=weight_ratio,
        )
        return [c.mean().item() for c in curves]
