from abc import ABC, abstractmethod
import numpy as np
from typing import Literal

VelocityProfile = Literal["oneway", "bothway"]

DEFAULT_VELOCITY_PROFILE = {
    "freq": [0.1, 0.1],  # 0.2Hz
    "mag": [-2, 0],  # 2m/s
}


class TargetVelocityGenerator(ABC):
    def __init__(self, dim: int, freq: float, mag: float):
        assert len(freq) == dim and len(mag) == dim, (
            "Frequency and magnitude lists must match the dimension."
        )
        self.dim = dim
        self.freq = freq
        self.mag = mag
        self.velocity = np.array([0] * self.dim)

    @abstractmethod
    def get_target_velocity(self, t: float) -> np.ndarray:
        # do something
        # self.velocity = np.array([0]*self.dim)
        return self.velocity


class SinusoidalVelocityGenerator(TargetVelocityGenerator):
    def __init__(
        self,
        dim,
        freq=DEFAULT_VELOCITY_PROFILE["freq"],
        mag=DEFAULT_VELOCITY_PROFILE["mag"],
    ):
        super().__init__(dim, freq, mag)

    def get_target_velocity(self, t: float) -> np.ndarray:
        self.velocity = np.array(
            [
                self.mag[i] * np.sin(2.0 * np.pi * self.freq[i] * t)
                for i in range(self.dim)
            ]
        )
        return self.velocity


class BiasedSinusoidalVelocityGenerator(TargetVelocityGenerator):
    def __init__(
        self,
        dim,
        freq=DEFAULT_VELOCITY_PROFILE["freq"],
        mag=DEFAULT_VELOCITY_PROFILE["mag"],
        bias=None,
    ):
        super().__init__(dim, freq, mag)
        if bias is None:
            bias = mag
        assert len(bias) == dim, "Bias list must match the dimension."
        self.bias = bias

    def get_target_velocity(self, t: float) -> np.ndarray:
        self.velocity = np.array(
            [
                0.5
                * (self.mag[i] * np.sin(2.0 * np.pi * self.freq[i] * t) + self.bias[i])
                for i in range(self.dim)
            ]
        )
        return self.velocity


def get_velocity_generator(profile: VelocityProfile) -> TargetVelocityGenerator:
    match profile:
        case "oneway":
            return BiasedSinusoidalVelocityGenerator
        case "bothway":
            return SinusoidalVelocityGenerator
        case _:
            raise ValueError(f"Invalid velocity profile '{profile}'")
