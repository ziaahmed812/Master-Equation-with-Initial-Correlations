import numpy as np
import master_equation_initial_correlations as meic

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This optional plotting example needs matplotlib; install it if you want "
        "to run this plotting script. The solver itself does not require matplotlib."
    ) from exc


system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
bath = meic.BathParams(
    bath_type="bosonic",
    kind="ohmic",
    s=1.0,
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)

tlist = np.linspace(0.0, 5.0, 51)
e_ops = ["jx"]
paper_numerics = meic.NumericsConfig(
    coefficient_omega_max=500.0,
    correlation_omega_max=510.0,
    initial_state_omega_max=500.0,
    omega_nodes=500,
    lambda_nodes=100,
    initial_state_omega_nodes=260,
    initial_state_lambda_nodes=40,
    initial_state_zeta_nodes=40,
    coefficient_time_step=0.0025,
    correlation_tau_step=0.0025,
)

exact_wc = meic.exact.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with")
exact_woc = meic.exact.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without")
me_wc = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with", numerics=paper_numerics)
me_woc = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without", numerics=paper_numerics)

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(me_wc.times, me_wc.e_data["jx"], color="black", linewidth=2.0, label="ME with correlations")
ax.plot(me_woc.times, me_woc.e_data["jx"], color="red", linestyle="--", linewidth=2.0, label="ME without correlations")
ax.plot(
    exact_wc.times,
    exact_wc.e_data["jx"],
    linestyle="none",
    marker="o",
    markersize=6,
    markerfacecolor="none",
    markeredgecolor="blue",
    label="exact with correlations",
)
ax.plot(
    exact_woc.times,
    exact_woc.e_data["jx"],
    linestyle="none",
    marker="s",
    markersize=6,
    markerfacecolor="none",
    markeredgecolor="purple",
    label="exact without correlations",
)

ax.set_xlim(0.0, 5.0)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$j_x$")
ax.set_title("Pure dephasing benchmark, N=4")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
