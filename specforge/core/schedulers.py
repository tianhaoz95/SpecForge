"""
Alpha schedulers for MDLM training.

Ported from dLLM framework to SpecForge architecture.
"""
import math
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Type, Union

import torch


Number = Union[int, float, torch.Tensor]


class BaseAlphaScheduler(ABC):
    """Base class for alpha schedulers with registry pattern.

    Alpha schedulers control the masking rate α(t) during diffusion training:
    - α(t): Masking rate function where t ∈ [0, 1]
    - α'(t): Derivative of α(t) with respect to t
    - weight(t): Loss weighting function w(t) = -α'(t) / (1 - α(t))
    """

    __registry__: ClassVar[Dict[str, Type["BaseAlphaScheduler"]]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register subclasses automatically."""
        super().__init_subclass__(**kwargs)
        BaseAlphaScheduler.__registry__[cls.__name__] = cls
        BaseAlphaScheduler.__registry__[cls.__name__.lower()] = cls
        # Add friendly names
        if cls.__name__ == "LinearAlphaScheduler":
            BaseAlphaScheduler.__registry__["linear"] = cls
        elif cls.__name__ == "CosineAlphaScheduler":
            BaseAlphaScheduler.__registry__["cosine"] = cls

    @abstractmethod
    def _alpha(self, t: Number) -> Number:
        """Core alpha function implementation."""
        pass

    @abstractmethod
    def _alpha_derivative(self, t: Number) -> Number:
        """Core alpha derivative implementation."""
        pass

    def alpha(self, t: Number) -> Number:
        """Masking rate α(t) ∈ [0,1].

        Args:
            t: Timestep(s) in [0, 1]

        Returns:
            Alpha value(s) corresponding to t
        """
        return self._alpha(t)

    def alpha_derivative(self, t: Number) -> Number:
        """Derivative dα/dt.

        Args:
            t: Timestep(s) in [0, 1]

        Returns:
            Alpha derivative value(s) corresponding to t
        """
        return self._alpha_derivative(t)

    def reverse_mask_prob(self, s: Number, t: Number) -> Number:
        """Reverse process probability (1 - α(s)) / (1 - α(t)).

        Used during generation to determine unmasking schedule.

        Args:
            s: Source timestep
            t: Target timestep

        Returns:
            Reverse masking probability
        """
        return (1 - self.alpha(s)) / (1 - self.alpha(t))

    def weight(self, t: Number) -> Number:
        """Loss weighting function w(t) = -α'(t) / (1 - α(t)).

        Used in scheduler-based loss weighting during training.

        Args:
            t: Timestep(s) in [0, 1]

        Returns:
            Loss weight(s) corresponding to t
        """
        alpha_val = self.alpha(t)
        alpha_deriv = self.alpha_derivative(t)

        # Add small epsilon for numerical stability
        if isinstance(alpha_val, torch.Tensor):
            denominator = (1 - alpha_val).clamp_min(1e-8)
        else:
            denominator = max(1 - alpha_val, 1e-8)

        return -alpha_deriv / denominator

    @classmethod
    def get_scheduler_class(cls, name: str) -> Type["BaseAlphaScheduler"]:
        """Get scheduler class by name.

        Args:
            name: Scheduler name (case insensitive)

        Returns:
            Scheduler class

        Raises:
            ValueError: If scheduler not found
        """
        if name not in cls.__registry__:
            available = list(cls.__registry__.keys())
            raise ValueError(f"Unknown scheduler: {name}. Available: {available}")
        return cls.__registry__[name]

    @classmethod
    def make_scheduler(cls, name: str, **kwargs) -> "BaseAlphaScheduler":
        """Create scheduler instance by name.

        Args:
            name: Scheduler name
            **kwargs: Additional arguments for scheduler

        Returns:
            Scheduler instance
        """
        scheduler_cls = cls.get_scheduler_class(name)
        return scheduler_cls(**kwargs)


class LinearAlphaScheduler(BaseAlphaScheduler):
    """Linear alpha scheduler: α(t) = 1 - t.

    Simple linear masking schedule where masking probability decreases linearly
    from 1 at t=0 to 0 at t=1.
    """

    def _alpha(self, t: Number) -> Number:
        return 1 - t

    def _alpha_derivative(self, t: Number) -> Number:
        if isinstance(t, torch.Tensor):
            return -torch.ones_like(t)
        else:
            return -1.0


class CosineAlphaScheduler(BaseAlphaScheduler):
    """Cosine alpha scheduler: α(t) = 1 - cos(π/2 * (1 - t)).

    Smoother masking schedule with more masking at early timesteps (t=0)
    and gradual transition to unmasked at t=1.
    """

    def _alpha(self, t: Number) -> Number:
        if isinstance(t, torch.Tensor):
            return 1 - torch.cos(math.pi / 2 * (1 - t))
        else:
            return 1 - math.cos(math.pi / 2 * (1 - t))

    def _alpha_derivative(self, t: Number) -> Number:
        coeff = -(math.pi / 2)
        if isinstance(t, torch.Tensor):
            return coeff * torch.sin(math.pi / 2 * (1 - t))
        else:
            return coeff * math.sin(math.pi / 2 * (1 - t))


# Factory functions for convenience
def get_alpha_scheduler_class(name: str) -> Type[BaseAlphaScheduler]:
    """Get alpha scheduler class by name."""
    return BaseAlphaScheduler.get_scheduler_class(name)


def make_alpha_scheduler(name: str, **kwargs) -> BaseAlphaScheduler:
    """Create alpha scheduler instance by name."""
    return BaseAlphaScheduler.make_scheduler(name, **kwargs)