from typing import Literal
import torch
import torch.nn as nn
from jaxtyping import Float


class LinOSSLayer(nn.Module):
    def __init__(
        self,
        size: int,
        use_parallel_scan: bool = False,
        solver_type: Literal["IM", "IMEX"] = "IM",
        bias: bool = True,
    ):
        super(LinOSSLayer, self).__init__()

        self.size = size

        self.glu = nn.GLU()
        self.activation = nn.GELU()

        self.C = nn.Linear(size, size, bias=bias)
        self.D = nn.Linear(size, size, bias=bias)

        # diagonal matrix A
        self.A = torch.diag(nn.Parameter(torch.randn(size)))

        self.B = nn.Linear(size, size, bias=bias)

        self.solver = self.solve_IM if solver_type == "IM" else self.solve_IMEX

    def solve_IM(
        self,
        u: Float[torch.Tensor, "B N *Q"],
        dt: float,
    ) -> Float[torch.Tensor, "B N *Q"]:
        """

        u: input tensor of shape (batch_size, time_steps, hideen_dim)
        dt: float, time step size

        """

        b, n, q = u.shape
        assert q == self.size

        S = torch.inverse(torch.eye(q) + dt**2 * self.A)
        assert S.shape == (q, q)

        M_inv: torch.Tensor = torch.block_diag(S, S)
        M_inv[0 : self.size, self.size :] += -dt * S @ self.A
        M_inv[self.size :, 0 : self.size] += dt * S
        assert M_inv.shape == (2 * q, 2 * q)

        # import numpy as np
        # np.savetxt("M_inv.txt", M_inv.detach().numpy())

        out = torch.zeros(u.shape[:-1] + (q * 2,), device=u.device)
        for i in range(n):
            u_n = u[:, i]

            # F = M_inv @ [dt*B @ u, 0]
            F = M_inv @ torch.cat([dt * self.B(u_n), torch.zeros_like(u_n)], dim=-1)
            if i == 0:
                out[:, i] = F
            else:
                out[:, i] = M_inv @ out[:, i - 1] + F

        return out[..., self.size :]

    def solve_IMEX(self, u, dt):
        raise NotImplementedError("IMEX solver not implemented")

    def forward(
        self, u: Float[torch.Tensor, "B N *Q"]
    ) -> Float[torch.Tensor, "B N *Q"]:
        """
        forcing term u
        """

        N = u.shape[1]
        dt = 1 / N

        # Step 1 solve ODE
        y = self.solver(u, dt=dt)
        # Step 2 calculate representation x
        x = self.C(y) + self.D(u)
        # Step 3 apply activation
        x = self.activation(x)

        return self.glu(torch.concat([x, x], dim=-1)) + u, y


class LinOSSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        solver_type: Literal["IM", "IMEX"] = "IM",
        use_parallel_scan: bool = False,  # TODO: implement parallel scan
    ):
        r"""
        input_dim: input dimension $q$ of each time step $u_i \in \mathbb{R}^{q}$
        output_dim: output dimension $q$ of each time step $u_i \in \mathbb{R}^{q}$
        hidden_dim: hidden dimension $m$ of each time step $u_i \in \mathbb{R}^{m}$
        """
        super(LinOSSModel, self).__init__()

        # init linOSS layers
        self.linOSS_layers = nn.ModuleList(
            [
                LinOSSLayer(
                    size=hidden_dim,
                    use_parallel_scan=use_parallel_scan,
                    solver_type=solver_type,
                )
                for _ in range(num_layers)
            ]
        )

        # init encoder and decoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, x: Float[torch.Tensor, "B N *Q"]
    ) -> Float[torch.Tensor, "B N *Q"]:
        """

        B: batch size
        N: number of time steps
        Q: input dimensions of each time step

        """

        u = self.encoder(x)

        for layer in self.linOSS_layers:
            u, y = layer(u)

        return self.decoder(u)
