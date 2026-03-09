import dataclasses
import typing
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

import numpy as np


@dataclass(frozen=True)
class DistributionConfig:
    min: float  # minimum of distribiution
    max: float  # maximum of distribution
    value: float  # constant value if distribution == 'constant'
    distribution: Literal[
        "constant",
        "log_uniform_values",
        "int_uniform",
        "uniform",
    ]  # type of distribution, in wandb-format (i.e. uniform, log_uniform_values, etc.)

    def sample(self) -> float:
        """Sample a value from this distribution, or return the constant value."""
        if self.distribution == "constant":
            return self.value

        if self.distribution == "log_uniform_values":
            return np.exp(
                np.random.uniform(low=np.log(self.min), high=np.log(self.max)),
            )
        if self.distribution == "int_uniform":
            return np.random.randint(low=self.min, high=self.max)

        return np.random.uniform(low=self.min, high=self.min)

    def to_wandb_sweep(self):
        """Serialise this distribution to the W&B sweep parameter format."""
        if self.distribution == "constant":
            return {"distribution": self.distribution, "value": self.value}

        return {
            "min": self.min,
            "max": self.max,
            "distribution": self.distribution,
        }


def dist_config_helper(
    min: float = 0.0,
    max: float = 0.0,
    value: float = 0.0,
    distribution: Literal[
        "constant",
        "log_uniform_values",
        "int_uniform",
        "uniform",
    ] = "constant",
) -> DistributionConfig:
    """Construct a DistributionConfig, nudging max above min if they are equal (W&B requirement)."""
    if min >= max:
        max += 1e-10
    return DistributionConfig(min=min, max=max, value=value, distribution=distribution)


def _is_fixed_field(cls: type, field_name: str) -> bool:
    """Return True if the field is annotated with tyro.conf.Fixed."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        return False
    annotation = hints.get(field_name)
    if annotation is None:
        return False
    if get_origin(annotation) is Annotated:
        from tyro.conf import Fixed

        _, *metadata = get_args(annotation)
        return Fixed in metadata
    return False


def _is_union_field(cls: type, field_name: str) -> bool:
    """Return True if the field is typed as a Union of dataclasses."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        return False
    annotation = hints.get(field_name)
    if annotation is None:
        return False
    # Unwrap Annotated[Union[...], ...] → Union[...]
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    origin = get_origin(annotation)
    return origin is Union or origin is UnionType


def merge_wandb_sweep_union(instances: list) -> dict[str, object]:
    """Build W&B sweep params for sweeping over multiple Union variants.

    Merges parameters from all instances (first-seen wins on conflicts);
    _type becomes a categorical {"values": [...]} sweep parameter.
    """
    merged: dict[str, object] = {}
    type_names = [type(inst).__name__ for inst in instances]

    for inst in instances:
        if hasattr(inst, "to_wandb_sweep") and callable(inst.to_wandb_sweep):
            inner = inst.to_wandb_sweep().get("parameters", {})
        else:
            inner = to_wandb_sweep_params(inst).get("parameters", {})
        for k, v in inner.items():
            if k not in merged:
                merged[k] = v

    merged["_type"] = {"values": type_names}
    return {"parameters": merged}


def to_wandb_sweep_params(obj) -> dict[str, object]:
    """Derive W&B sweep parameters from a dataclass by inspecting its fields.

    Fields annotated with tyro.conf.Fixed are excluded automatically.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"Expected a dataclass instance, got {type(obj)}")

    params: dict[str, object] = {}
    cls = type(obj)
    for field in dataclasses.fields(obj):
        if _is_fixed_field(cls, field.name):
            continue
        attr = getattr(obj, field.name)
        if hasattr(attr, "to_wandb_sweep") and callable(attr.to_wandb_sweep):
            nested = attr.to_wandb_sweep()
            if _is_union_field(cls, field.name):
                nested["parameters"]["_type"] = {"value": type(attr).__name__}
            params[field.name] = nested
        else:
            params[field.name] = {"value": attr}
    return {"parameters": params}
