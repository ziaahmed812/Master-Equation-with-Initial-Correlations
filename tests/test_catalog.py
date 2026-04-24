from master_equation_initial_correlations.catalog import get_example, list_examples


def test_list_examples_uses_public_physical_ids() -> None:
    ids = {example.public_id for example in list_examples()}
    assert "pure-dephasing-ohmic-N1" in ids
    assert "spin-boson-ohmic-jx-N4" in ids
    assert "spin-bath-ohmic-N4" in ids


def test_get_example_supports_public_ids() -> None:
    example = get_example("pure-dephasing-ohmic-N1")
    assert example.bath == "bosonic"
    assert example.model == "pure-dephasing"
    assert example.parameters["N"] == 1
