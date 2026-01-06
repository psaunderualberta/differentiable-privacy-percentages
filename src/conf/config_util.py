from dataclasses import dataclass

import equinox as eqx
import numpy as np
from tyro.conf import Fixed


@dataclass(frozen=True)
class DistributionConfig:
    min: float  # minimum of distribiution
    max: float  # maximum of distribution
    value: float  # constant value if distribution == 'constant'
    distribution: str  # type of distribution, in wandb-format (i.e. uniform, log_uniform_values, etc.)

    def sample(self) -> float:
        if self.distribution == "constant":
            return self.value

        elif self.distribution == "log_uniform_values":
            return np.exp(
                np.random.uniform(low=np.log(self.min), high=np.log(self.max))
            )

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
# >:(
def dist_config_helper(
    min: float = 0.0,
    max: float = 0.0,
    value: float = 0.0,
    distribution: str = "constant",
) -> DistributionConfig:
    if min >= max:
        max += 1e-10
    return DistributionConfig(min=min, max=max, value=value, distribution=distribution)


def to_wandb_sweep_params(self) -> dict[str, object]:
    assert hasattr(self, "attrs"), "Need to know which attrs to sweep over!"

    params = {}

    for attr_name in self.attrs:
        attr = getattr(self, attr_name)
        if hasattr(attr, "to_wandb_sweep") and callable(attr.to_wandb_sweep):
            params[attr_name] = attr.to_wandb_sweep()
        else:
            params[attr_name] = {"value": attr}

    return {"parameters": params}
