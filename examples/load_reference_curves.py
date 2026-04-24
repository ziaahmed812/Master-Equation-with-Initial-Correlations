import master_equation_initial_correlations as meic


curves = meic.load_reference_curves("pure-dephasing-ohmic-N1")
print(curves.example.label)
print(curves.correlated[:5])
