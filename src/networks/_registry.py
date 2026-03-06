"""Registration decorator and factory for network architectures (MLP, CNN, …).

Each concrete network class registers itself with @register(ConfigClass).
The factory build() dispatches to the correct class without an explicit if-chain.
"""

from __future__ import annotations

_REGISTRY: dict[type, type] = {}


def register(config_cls: type):
    """Class decorator — register a network class against its config type.

    Usage::

        @register(MLPConfig)
        class MLP(eqx.Module, Network): ...
    """

    def decorator(network_cls: type) -> type:
        _REGISTRY[config_cls] = network_cls
        return network_cls

    return decorator


def build(conf, input_shape: tuple[int, ...], output_shape: tuple[int, ...], key: int = 0):
    """Construct the network corresponding to *conf*.

    Args:
        conf:         A concrete network config instance (MLPConfig, CNNConfig, …).
        input_shape:  Shape of the input batch, e.g. (batch, features) or (batch, C, H, W).
        output_shape: Shape of the output batch, e.g. (batch, nclasses).
        key:          Integer PRNG seed.
    """
    cls = _REGISTRY.get(type(conf))
    if cls is None:
        known = [c.__name__ for c in _REGISTRY]
        raise ValueError(
            f"No network registered for config type '{type(conf).__name__}'. Known types: {known}",
        )
    return cls.build(conf, input_shape, output_shape, key)
