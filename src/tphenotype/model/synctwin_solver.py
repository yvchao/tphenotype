import torch
from tqdm import auto

from ..utils.decorators import numpy_io
from ..utils.utils import EPS, get_summary


class AffinitySolver(torch.nn.Module):
    def __init__(
        self,
        m,
        n,
    ):
        super().__init__()
        self.m = m
        self.n = n
        self.B = torch.nn.Parameter(torch.zeros((m, n)))

    def clone_state_dict(self):
        state_dict = {}
        for key in self.state_dict():  # pylint: disable=not-an-iterable
            state_dict[key] = self.state_dict()[key].clone()  # pylint: disable=unsubscriptable-object
        return state_dict

    @numpy_io
    def get_affinity(self, A):
        with torch.no_grad():
            B = self._get_affinity(A)
        return B

    def _get_affinity(self, A):
        F = torch.zeros_like(A)
        F[A != 1] = -1 / EPS
        B = self.B + F
        B = torch.softmax(B, dim=-1)
        return B

    def forward(self, z_test, z_corpus, A):
        B = self._get_affinity(A)

        z_hat = torch.mm(B, z_corpus)

        loss = {}
        z_error = torch.norm(z_test - z_hat, p=2, dim=1)
        z_mean = torch.mean(torch.norm(z_test, dim=1)).detach()
        loss["z_rec"] = torch.mean(z_error / (z_mean + EPS))
        return loss

    def solve(self, Z, A, max_iter=300, learning_rate=0.1, tol=1e-7, verbose=True):
        z_test, z_corpus = Z
        z_test = torch.from_numpy(z_test)
        z_corpus = torch.from_numpy(z_corpus)
        A = torch.from_numpy(A)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.9))
        best_loss = torch.inf
        best_model = {}

        with auto.trange(max_iter, position=0, leave=True, disable=not verbose) as tbar:
            for _ in tbar:
                loss = self.forward(z_test, z_corpus, A)
                # L = loss['z_rec'] + alpha * loss['B_rec'] - beta * loss['ll']
                L = loss["z_rec"]
                L.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # pyright: ignore [reportPrivateImportUsage]
                optimizer.step()

                metrics = {k: v.detach().item() for k, v in loss.items()}
                tbar.set_description(get_summary(metrics))

                current_loss = L.detach().item()
                if abs(current_loss - best_loss) < tol:
                    break

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = self.clone_state_dict()

        if best_model != {}:
            self.load_state_dict(best_model)
        return self
