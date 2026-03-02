import dataclasses
import typing
from dataclasses import dataclass
from typing import Annotated, Literal, get_args, get_origin

import equinox as eqx
import numpy as np


@dataclass(frozen=True)
class DistributionConfig:
    min: float  # minimum of distribiution
    max: float  # maximum of distribution
    value: float  # constant value if distribution == 'constant'
    distribution: Literal[
        "constant", "log_uniform_values", "int_uniform", "uniform"
    ]  # type of distribution, in wandb-format (i.e. uniform, log_uniform_values, etc.)

    def sample(self) -> float:
        if self.distribution == "constant":
            return self.value

        elif self.distribution == "log_uniform_values":
            return np.exp(
                np.random.uniform(low=np.log(self.min), high=np.log(self.max))
            )
        elif self.distribution == "int_uniform":
            return np.random.randint(low=self.min, high=self.max)

        return np.random.uniform(low=self.min, high=self.min)

    def to_wandb_sweep(self):
        if self.distribution == "constant":
            return {"distribution": self.distribution, "value": self.value}

        return {
            "min": self.min,
            "max": self.max,
            "distribution": self.distribution,
        }


# wandb cannot create sweeps if any distribution has min >= max
def dist_config_helper(
    min: float = 0.0,
    max: float = 0.0,
    value: float = 0.0,
    distribution: Literal[
        "constant", "log_uniform_values", "int_uniform", "uniform"
    ] = "constant",
) -> DistributionConfig:
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


def to_wandb_sweep_params(obj) -> dict[str, object]:
    """Derive W&B sweep parameters from a dataclass by inspecting its fields.

    Fields annotated with tyro.conf.Fixed are excluded automatically.
    No manual 'attrs' tuple is needed.
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
            params[field.name] = attr.to_wandb_sweep()
        else:
            params[field.name] = {"value": attr}
    return {"parameters": params}
