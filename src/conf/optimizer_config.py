"""Signature-derived optimizer configs for tyro subcommand dispatch.

Each supported optax callable is lifted into a dataclass whose fields are
produced from ``inspect.signature``: floats are wrapped in
``DistributionConfig`` (so W&B can sweep over them), bools/ints are
passthrough, and unparseable types (Callable, dtype, mask, etc.) are skipped.

The generated dataclasses participate in the same Union-as-subcommand pattern
used elsewhere in the config tree; ``_type`` round-tripping from W&B goes
through ``optimizer_registry``.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import warnings
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin

import optax
import tyro

from conf.config_util import DistributionConfig, dist_config_helper
from conf.optimizer_registry import register


def _union_contains(ann, target: type) -> bool:
    origin = get_origin(ann)
    if origin is Union or origin is types.UnionType:
        return any(a is target for a in get_args(ann))
    return False


def _classify(ann) -> str:
    """Return 'float', 'bool', 'int', or 'skip' for a parameter annotation."""
    if ann is float or _union_contains(ann, float):
        return "float"
    if ann is bool:
        return "bool"
    if ann is int or _union_contains(ann, int):
        return "int"
    return "skip"


class OptaxConfigMixin:
    """Shared ``build()`` for every make_optimizer_config-generated class.

    Walks dataclass fields, samples any ``DistributionConfig`` down to a scalar,
    and forwards the resulting kwargs to the underlying optax callable.
    """

    _fn: ClassVar[Callable[..., optax.GradientTransformation]]

    def resolve(self):
        """Return a copy with every DistributionConfig sampled to a constant.

        Call before ``build()`` if you also need to read individual hyperparameter
        values — prevents double-sampling that would decouple the optimizer's
        configuration from what is logged/read elsewhere.
        """
        updates: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if isinstance(v, DistributionConfig):
                updates[f.name] = dist_config_helper(
                    value=float(v.sample()),
                    distribution="constant",
                )
        return dataclasses.replace(self, **updates)  # type: ignore[arg-type]

    def build(self) -> optax.GradientTransformation:
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            kwargs[f.name] = v.sample() if isinstance(v, DistributionConfig) else v
        return type(self)._fn(**kwargs)

    def to_wandb_sweep(self) -> dict[str, Any]:
        from conf.config_util import to_wandb_sweep_params

        return to_wandb_sweep_params(self)

    def to_wandb_conf(self) -> dict[str, Any]:
        from conf.config_util import to_wandb_conf

        return to_wandb_conf(self)


def make_optimizer_config(
    fn: Callable[..., optax.GradientTransformation],
    *,
    name: str,
    defaults: dict[str, Any] | None = None,
    skip: tuple[str, ...] = (),
    extra_fields: tuple[tuple[str, type, Any], ...] = (),
) -> type:
    """Construct a dataclass whose fields mirror ``fn``'s signature.

    Args:
        fn:            The optax constructor (e.g. ``optax.sgd``).
        name:          Class name; also the W&B ``_type`` discriminator.
        defaults:      Overrides / fills for signature params.  Required for any
                       signature param that has no default of its own.
        skip:          Parameter names to omit entirely.
        extra_fields:  ``(name, type, default)`` triples appended to the dataclass
                       (rarely needed).

    Raises:
        ValueError: a signature-required parameter has no default in ``defaults``.
    """
    defaults = dict(defaults or {})
    sig = inspect.signature(fn)
    fields: list[tuple[str, type, Any]] = []

    for pname, param in sig.parameters.items():
        if pname in skip:
            continue
        kind = _classify(param.annotation)
        if kind == "skip":
            warnings.warn(
                f"{name}: skipping '{pname}' — unparseable type {param.annotation!r}",
                stacklevel=2,
            )
            continue

        sig_has_default = param.default is not inspect.Parameter.empty
        if pname in defaults:
            raw_default = defaults[pname]
        elif sig_has_default and param.default is not None:
            raw_default = param.default
        elif sig_has_default and param.default is None and kind == "float":
            raise ValueError(
                f"{name}: '{pname}' has Optional[...] default None; "
                f"supply a concrete default via defaults={{'{pname}': ...}}",
            )
        elif not sig_has_default:
            raise ValueError(
                f"{name}: required parameter '{pname}' has no default; "
                f"supply one via defaults={{'{pname}': ...}}",
            )
        else:
            raw_default = param.default

        if kind == "float":
            field_type = DistributionConfig
            field_default = dataclasses.field(
                default_factory=lambda v=float(raw_default): dist_config_helper(
                    value=v,
                    distribution="constant",
                ),
            )
        elif kind == "bool":
            field_type = bool
            field_default = bool(raw_default)
        else:  # int
            field_type = int
            field_default = int(raw_default)

        fields.append((pname, field_type, field_default))

    fields.extend(extra_fields)

    cls = dataclasses.make_dataclass(
        name,
        fields,
        bases=(OptaxConfigMixin,),
    )
    cls.__module__ = __name__
    cls.__qualname__ = name
    cls._fn = staticmethod(fn)  # type: ignore[attr-defined]
    register(cls)
    return cls


# ---------------------------------------------------------------------------
# Enumerated optimizers
# ---------------------------------------------------------------------------

SGDConfig = make_optimizer_config(
    optax.sgd,
    name="SGDConfig",
    defaults={"learning_rate": 0.1, "momentum": 0.9},
)

# eps_root > 0 puts epsilon inside the sqrt (sqrt(v + eps_root)), which keeps
# the backward pass finite when v_hat ≈ 0 — required when differentiating
# through the inner DP-SGD scan into the schedule.
AdamConfig = make_optimizer_config(
    optax.adam,
    name="AdamConfig",
    defaults={"learning_rate": 1e-3, "eps_root": 1e-8},
)

AdamWConfig = make_optimizer_config(
    optax.adamw,
    name="AdamWConfig",
    defaults={"learning_rate": 1e-3, "eps_root": 1e-8},
)


OptimizerConfig = Union[  # noqa: UP007
    Annotated[SGDConfig, tyro.conf.subcommand("sgd")],
    Annotated[AdamConfig, tyro.conf.subcommand("adam")],
    Annotated[AdamWConfig, tyro.conf.subcommand("adamw")],
]
