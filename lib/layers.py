from typing import Literal
import torch
import torch.nn as nn
from jaxtyping import Float

import matplotlib.pyplot as plt
import numpy as np


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()

        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def __call__(self, x):
        return self.w1(x) * nn.functional.sigmoid(self.w2(x))


class LinOSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dt: float = 1.0,
        use_parallel_scan: bool = False,
        solver_type: Literal["IM", "IMEX"] = "IM",
        bias: bool = True,
    ):
        super(LinOSSBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.dt = dt

        self.glu = GLU(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        A_max = 1.0

        self.C = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.D = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(p=0.1)

        # diagonal matrix A
        self.A = torch.diag(nn.Parameter(torch.abs(torch.randn(hidden_dim) * A_max)))

        self.B = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.solver = self.solve_IM if solver_type == "IM" else self.solve_IMEX

    def solve_IM(
        self,
        u: Float[torch.Tensor, "b n m"],
        dt: float,
    ) -> Float[torch.Tensor, "b n m"]:
        """Solves the ODE $x' = Mx + F$ using implicit time integration which results from the first second order ODE: $y'' = Ay(t) + Bu(t)$.

        n: number of time steps
        m: dimension of the state space

        Detailed description in the paper: https://arxiv.org/abs/2410.03943 (Section 2.3 DISCRETIZATION)

        Args:
            u (Tensor): Input tensor of shape (n, m)
            dt (float): Time step size

        Returns:
            Tensor (n, m): Solution of the ODE
        """
        # TODO: Allow non-equidistant time steps -> dts: Float[torch.Tensor, " n"]
        # TODO: In original implementation timesteps are fed through sigmoid first

        b, n, m = u.shape

        A = nn.ReLU()(self.A)
        S = torch.inverse(torch.eye(m) + dt**2 * A)
        assert S.shape == (m, m)

        M_inv: torch.Tensor = torch.block_diag(S, S)
        M_inv[0 : self.hidden_dim, self.hidden_dim :] += -dt * S @ A
        M_inv[self.hidden_dim :, 0 : self.hidden_dim] += dt * S
        assert M_inv.shape == (2 * m, 2 * m)

        x = torch.zeros(u.shape[:-1] + (m * 2,), device=u.device)
        for i in range(n):
            u_n = u[:, i]

            # F = M_inv @ [dt*B @ u, 0]
            F = torch.cat([dt * self.B(u_n), torch.zeros_like(u_n)], dim=-1)
            if i == 0:
                x[:, i] = F @ M_inv.T
            else:
                x[:, i] = (x[:, i - 1] + F) @ M_inv.T

        return x[:, :, m:]

    def solve_IMEX(self, u, dt):
        raise NotImplementedError("IMEX solver not implemented yet")

    def forward(
        self,
        u: Float[torch.Tensor, "n m"],
        dt: float,
    ) -> Float[torch.Tensor, "n m"]:
        """Forward pass of the LinOSSBlock

        Args:
            x (Tensor): Input tensor

        Returns:
            _type_: _description_
        """

        # TODO: Possibly apply Batchnorm and Dropout

        # step 0: apply Batchnorm
        # u = self.norm(u)
        # Step 1 solve ODE
        x = self.solver(u, dt=dt)
        x = self.dropout(x)
        # Step 2 calculate representation x
        x = self.C(x) + self.D(u)
        # Step 3 apply activation
        x = self.activation(x)

        return self.dropout(self.glu(x)) + u


class LinOSSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        classification: bool = False,
        solver_type: Literal["IM", "IMEX"] = "IM",
        use_parallel_scan: bool = False,  # TODO: implement parallel scan
    ) -> None:
        """_summary_

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_dim (int): _description_
            num_layers (int): _description_
            solver_type (Literal[&quot;IM&quot;, &quot;IMEX&quot;], optional): _description_. Defaults to "IM".
            use_parallel_scan (bool, optional): _description_. Defaults to False.
        """
        super(LinOSSModel, self).__init__()

        self.classification = classification

        # init linOSS layers
        self.linOSS_layers = nn.ModuleList(
            [
                LinOSSBlock(
                    hidden_dim=hidden_dim,
                    use_parallel_scan=use_parallel_scan,
                    solver_type=solver_type,
                )
                for _ in range(num_layers)
            ]
        )

        # init encoder and decoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

        self.final_activation = nn.Tanh()

    def forward(
        self, x: Float[torch.Tensor, "b n p"], dt: float
    ) -> Float[torch.Tensor, "b n p"]:
        """

        b: batch size
        n: number of time steps
        p: dimension of the input state
        """

        u = self.encoder(x)

        for layer in self.linOSS_layers:
            u = layer(u, dt)

        u = self.decoder(u)

        if self.classification:
            u = torch.mean(u, axis=0)
            u = nn.functional.softmax(u, axis=0)
        else:
            u = self.final_activation(u)

        return u


if __name__ == "__main__":
    # Example usage
    layer = LinOSSBlock(size=1)
    x = torch.linspace(0, 1, 1000)
    u = torch.ones_like(x)

    layer.A = torch.tensor([[0.5]])

    # set parameters of B to weight=1 and bias=0
    layer.B.weight.data.fill_(1.0)
    layer.B.bias.data.fill_(0.0)

    output = layer.solve_IM(u, dt=0.1)

    z = output[:, 0].detach().numpy()
    y = output[:, 1].detach().numpy()
    y_pp = np.gradient(np.gradient(y))
    ref = 0.5 * np.gradient(y) + 1 * u.detach().numpy()

    plt.plot(y, label="y")
    plt.plot(z, label="z")
    plt.plot(np.gradient(y), label="y'", linestyle="-.")
    plt.plot(y_pp, label="y''", linestyle="--")
    plt.plot(ref, label="ref", linestyle=":")

    plt.legend()
    plt.show()
