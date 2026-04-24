from __future__ import annotations

from functools import lru_cache

from ._resources import read_json
from ._types import ReferenceExample


def _decimal_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _public_metadata(raw: dict) -> tuple[str, str, str, str]:
    family = raw["family"]
    params = raw["parameters"]
    N = int(params["N"])
    beta = float(params.get("beta", 1.0))

    if family == "pure_dephasing_ohmic":
        return f"pure-dephasing-ohmic-N{N}", "bosonic", "pure-dephasing", "ohmic"
    if family in {"beyond_pure_dephasing_ohmic", "beta_compare_ohmic"}:
        suffix = f"-beta{_decimal_token(beta)}" if family == "beta_compare_ohmic" else ""
        return f"spin-boson-ohmic-jx{suffix}-N{N}", "bosonic", "spin-boson", "ohmic"
    if family == "jx2_ohmic":
        return f"spin-boson-ohmic-jx2-N{N}", "bosonic", "spin-boson", "ohmic"
    if family == "subohmic_s0p5":
        s = _decimal_token(float(params["s"]))
        return f"spin-boson-subohmic-s{s}-N{N}", "bosonic", "spin-boson", "subohmic"
    if family == "spin_environment_ohmic":
        return f"spin-bath-ohmic-N{N}", "spin", "spin-environment", "ohmic"
    raise ValueError(f"Unsupported example family {family!r}")


def _example_from_manifest(raw: dict) -> ReferenceExample:
    public_id, bath, model, spectral = _public_metadata(raw)
    aliases = tuple(raw.get("aliases", ()))
    return ReferenceExample(
        id=raw["example_id"],
        public_id=public_id,
        bath=bath,
        model=model,
        spectral=spectral,
        paper_figure_number=int(raw["paper_figure_number"]),
        family=raw["family"],
        label=raw["label"],
        paper_asset=raw["paper_asset"],
        observable=raw["observable"],
        solver_id=raw["solver_id"],
        execution_profile=raw["execution_profile"],
        supports_exact=bool(raw["supports_exact"]),
        supports_rerun=bool(raw["supports_rerun"]),
        parameters=dict(raw["parameters"]),
        x_range=tuple(float(v) for v in raw["plot"]["x_range"]),
        y_range=tuple(float(v) for v in raw["plot"]["y_range"]),
        plot_divisor=float(raw["plot"]["plot_divisor"]),
        asset_dir=raw["paths"]["asset_dir"],
        aliases=(public_id, *aliases),
    )


@lru_cache(maxsize=1)
def _all_examples() -> tuple[ReferenceExample, ...]:
    manifest = read_json("figure_manifest.json")
    examples = tuple(_example_from_manifest(raw) for raw in manifest["examples"])
    return tuple(sorted(examples, key=lambda example: example.paper_figure_number))


@lru_cache(maxsize=1)
def _index() -> dict[str, ReferenceExample]:
    result: dict[str, ReferenceExample] = {}
    for example in _all_examples():
        result[example.id] = example
        for alias in example.aliases:
            result[alias] = example
    return result


def list_examples() -> list[ReferenceExample]:
    return list(_all_examples())


def get_example(example_id: str) -> ReferenceExample:
    try:
        return _index()[example_id]
    except KeyError as exc:
        known = ", ".join(example.public_id for example in _all_examples())
        raise KeyError(f"Unknown reference example {example_id!r}. Known examples: {known}") from exc
