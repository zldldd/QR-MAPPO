"""SPAC: Staged Perturbation Adversarial Curriculum scheduler."""
from collections import deque
from typing import Optional


class SPACScheduler:
    """Adjusts environment adversary rationality based on a moving window of
    episode returns.  When recent performance improves, the adversary becomes
    more rational (harder); when performance drops, it becomes softer.
    """

    def __init__(
        self,
        window: int = 50,
        rho_high: float = 2.0,
        rho_low: float = 0.5,
        delta: float = 0.05,
        init_rho: float = 0.5,
    ):
        self.window = window
        self.rho_high = rho_high
        self.rho_low = rho_low
        self.delta = delta
        self.rho = init_rho
        self._returns: deque = deque(maxlen=window)
        self._prev_mean: Optional[float] = None

    def update(self, episode_return: float) -> float:
        self._returns.append(episode_return)
        if len(self._returns) < self.window // 2:
            return self.rho
        mean_ret = sum(self._returns) / len(self._returns)
        if self._prev_mean is not None:
            if mean_ret > self._prev_mean:
                self.rho = min(self.rho_high, self.rho + self.delta)
            elif mean_ret < self._prev_mean:
                self.rho = max(self.rho_low, self.rho - self.delta)
        self._prev_mean = mean_ret
        return self.rho

    def set_rho(self, rho: float):
        self.rho = float(rho)
