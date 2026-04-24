from pathlib import Path

from master_equation_initial_correlations import load_reference_curves
from master_equation_initial_correlations.plotting import apply_paper_axes_style, plot_reference_curves


def test_plot_reference_curves_writes_eps_and_png(tmp_path: Path) -> None:
    curves = load_reference_curves("pure-dephasing-ohmic-N4")
    outputs = plot_reference_curves(curves, tmp_path)
    assert outputs["eps"].exists()
    assert outputs["png"].exists()


def test_apply_paper_axes_style_for_jx2_ticks() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    spec = apply_paper_axes_style(ax, observable="jx2", x_range=(0.0, 2.0), y_range=(0.0, 1.0))
    assert spec["x_ticks"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert spec["y_ticks"] == [0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
    plt.close(fig)
