from dataclasses import dataclass

import numpy as np


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
