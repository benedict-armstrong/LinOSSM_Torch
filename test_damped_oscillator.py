import torch
import numpy as np
import matplotlib.pyplot as plt
from lib.layers import LinOSSLayer
from scipy.integrate import solve_ivp
import torch.nn as nn


def analytical_solution(t, x0, v0, k1, k2, m1, m2, coupling, c1, c2):
    """Solve damped coupled oscillator using scipy's ODE solver as reference"""

    def oscillator_system(t, y):
        x1, v1, x2, v2 = y
        dx1dt = v1
        dv1dt = -k1 / m1 * x1 - coupling / m1 * (x1 - x2) - c1 / m1 * v1
        dx2dt = v2
        dv2dt = -k2 / m2 * x2 - coupling / m2 * (x2 - x1) - c2 / m2 * v2
        return [dx1dt, dv1dt, dx2dt, dv2dt]

    y0 = [x0[0], v0[0], x0[1], v0[1]]
    sol = solve_ivp(oscillator_system, [0, t[-1]], y0, t_eval=t, method="RK45")
    return sol.y


def test_linoss_damped_oscillator():
    # System parameters
    m1, m2 = 1.0, 1.5  # masses
    k1, k2 = 1.0, 1.5  # spring constants
    coupling = 0.5  # coupling strength
    c1, c2 = 0.2, 0.3  # damping coefficients
    x0 = torch.tensor([1.0, 0.0])  # initial positions
    v0 = torch.tensor([0.0, 0.0])  # initial velocities

    # Time parameters
    T = 20.0  # total simulation time
    N = 200  # number of time steps
    dt = T / N  # time step size
    t = np.linspace(0, T, N)

    # Get reference solution
    ref_sol = analytical_solution(
        t, x0.numpy(), v0.numpy(), k1, k2, m1, m2, coupling, c1, c2
    )

    # Plot analytical solution only
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, ref_sol[0], "b-", label="Mass 1 Position (x1)")
    plt.plot(t, ref_sol[2], "r-", label="Mass 2 Position (x2)")
    plt.legend()
    plt.title("Analytical Solution - Positions")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, ref_sol[1], "b-", label="Mass 1 Velocity (v1)")
    plt.plot(t, ref_sol[3], "r-", label="Mass 2 Velocity (v2)")
    plt.legend()
    plt.title("Analytical Solution - Velocities")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("analytical_solution.png")
    plt.show()

    # Setup LinOSSLayer
    size = 2  # 2D system (x1, x2)
    layer = LinOSSLayer(size=size)

    # Define A matrix for the coupled oscillator without damping
    # In approach 1, A only includes stiffness and coupling terms
    A_val = torch.zeros(size, size)
    A_val[0, 0] = (k1 + coupling) / m1
    A_val[0, 1] = -coupling / m1
    A_val[1, 0] = -coupling / m2
    A_val[1, 1] = (k2 + coupling) / m2
    layer.A = nn.Parameter(A_val)

    # Set B to identity for the external forces
    with torch.no_grad():
        for i in range(size):
            layer.B.weight[i, i] = 1.0
            layer.B.bias.zero_()

    # Set up arrays to store simulation results
    positions = torch.zeros((1, N, size))
    velocities = torch.zeros((1, N, size))

    # Initialize positions and velocities
    positions[0, 0] = x0
    velocities[0, 0] = v0

    # Note: We need to run this in eval mode
    layer.eval()

    # Step-by-step simulation with feedback
    for i in range(1, N):
        # Current velocity is used to compute damping forces (as input)
        u_damping = torch.zeros((1, 1, size))
        u_damping[0, 0, 0] = -c1 / m1 * velocities[0, i - 1, 0]
        u_damping[0, 0, 1] = -c2 / m2 * velocities[0, i - 1, 1]

        # Get current position as a single time step
        current_pos = positions[0, i - 1 : i].unsqueeze(0)

        # Solve for next position using LinOSS
        with torch.no_grad():
            next_pos = layer.solve_IM(current_pos, dt=dt, u=u_damping)
            positions[0, i] = next_pos[0, 0]

            # Update velocity using central difference (for next iteration)
            if i < N - 1:
                # For intermediate points, use central difference
                velocities[0, i] = (positions[0, i + 1] - positions[0, i - 1]) / (
                    2 * dt
                )
            else:
                # For the last point, use backward difference
                velocities[0, i] = (positions[0, i] - positions[0, i - 1]) / dt

    # Extract position data
    x1 = positions[0, :, 0].numpy()
    x2 = positions[0, :, 1].numpy()
    v1 = velocities[0, :, 0].numpy()
    v2 = velocities[0, :, 1].numpy()

    # Plot comparison results
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, ref_sol[0], "b-", label="Reference x1")
    plt.plot(t, ref_sol[2], "r-", label="Reference x2")
    plt.plot(t, x1, "b--", label="LinOSS x1")
    plt.plot(t, x2, "r--", label="LinOSS x2")
    plt.legend()
    plt.title("Positions")
    plt.xlabel("Time")

    plt.subplot(3, 1, 2)
    plt.plot(t, ref_sol[1], "b-", label="Reference v1")
    plt.plot(t, ref_sol[3], "r-", label="Reference v2")
    plt.plot(t, v1, "b--", label="LinOSS v1")
    plt.plot(t, v2, "r--", label="LinOSS v2")
    plt.legend()
    plt.title("Velocities")
    plt.xlabel("Time")

    plt.subplot(3, 1, 3)
    x1_error = np.abs(x1 - ref_sol[0])
    x2_error = np.abs(x2 - ref_sol[2])
    plt.plot(t, x1_error, "b-", label="x1 Error")
    plt.plot(t, x2_error, "r-", label="x2 Error")
    plt.legend()
    plt.title("Absolute Error")
    plt.xlabel("Time")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("damped_oscillator_approach1.png")
    plt.show()

    # Calculate system energy
    ke = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    pe = 0.5 * k1 * x1**2 + 0.5 * k2 * x2**2
    total_energy = ke + pe

    plt.figure(figsize=(10, 6))
    plt.plot(t, total_energy, "g-", label="Total Energy")
    plt.plot(t, ke, "r-", label="Kinetic Energy")
    plt.plot(t, pe, "b-", label="Potential Energy")
    plt.title("System Energy (should decrease due to damping)")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig("damped_oscillator_energy_approach1.png")
    plt.show()

    # Calculate error
    error = np.mean((x1 - ref_sol[0]) ** 2) / np.mean(ref_sol[0] ** 2)
    print(f"Relative squared error: {error:.6f}")

    return error < 0.2  # Return True if error is acceptable


if __name__ == "__main__":
    test_result = test_linoss_damped_oscillator()
    print(f"Test {'passed' if test_result else 'failed'}")
