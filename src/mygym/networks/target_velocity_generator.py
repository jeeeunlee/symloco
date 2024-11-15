from abc import ABC, abstractmethod
import numpy as np

DEFAULT_VELOCITY_PROFILE = {
    "freq": [0.1, 0.1],  # 0.2Hz
    "mag": [2, 0],  # 2m/s
}


class TargetVelocityGenerator(ABC):
    def __init__(self, dim):
        self.dim = dim
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
        super().__init__(dim)
        assert (
            len(freq) == dim and len(mag) == dim
        ), "Frequency and magnitude lists must match the dimension."
        self.freq = freq
        self.mag = mag

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
        bias=DEFAULT_VELOCITY_PROFILE["mag"],
    ):
        super().__init__(dim)
        assert (
            len(freq) == dim and len(mag) == dim
        ), "Frequency and magnitude lists must match the dimension."
        self.freq = freq
        self.mag = mag
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
