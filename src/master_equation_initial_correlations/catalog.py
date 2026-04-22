from __future__ import annotations

from functools import lru_cache

from ._resources import read_json
from ._types import Preset


def _preset_from_manifest(raw: dict) -> Preset:
    return Preset(
        id=raw["preset_id"],
        figure_id=raw["figure_id"],
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
        aliases=tuple(raw.get("aliases", ())),
    )


@lru_cache(maxsize=1)
def _all_presets() -> tuple[Preset, ...]:
    manifest = read_json("figure_manifest.json")
    presets = tuple(_preset_from_manifest(raw) for raw in manifest["presets"])
    return tuple(sorted(presets, key=lambda preset: preset.paper_figure_number))


@lru_cache(maxsize=1)
def _index() -> dict[str, Preset]:
    result: dict[str, Preset] = {}
    for preset in _all_presets():
        result[preset.id] = preset
        for alias in preset.aliases:
            result[alias] = preset
    return result


def list_presets(*, include_advanced: bool = False, include_heavy: bool = False) -> list[Preset]:
    presets = list(_all_presets())
    if include_heavy:
        return presets
    if include_advanced:
        return [preset for preset in presets if preset.execution_profile != "heavy"]
    return [preset for preset in presets if preset.execution_profile == "light"]


def get_preset(preset_id: str) -> Preset:
    try:
        return _index()[preset_id]
    except KeyError as exc:
        known = ", ".join(preset.id for preset in _all_presets())
        raise KeyError(f"Unknown preset {preset_id!r}. Known presets: {known}") from exc
