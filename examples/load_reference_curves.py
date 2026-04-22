import master_equation_initial_correlations as meic


curves = meic.load_reference_curves("pure_dephasing_ohmic_j0p5")
print(curves.preset.label)
print(curves.correlated[:5])
