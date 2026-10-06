"""Adaptive gradient clipping: clip at a multiple of the recent median gradient norm.

The loss is summed over tokens, so gradient norms differ by model, by orders of magnitude, and drift during training.
A single fixed clip is either inert for one model or clips every other step for another. This rule follows each
run's own scale instead:

    threshold(t) = min(max_grad_norm, multiple * median(clipped norms of the last `window` steps))

- The threshold for step t uses only steps before t: the trainer applies one global clip per optimizer step and
  reports the pre-clip norm afterwards.
- The history stores each norm clipped at its step's threshold, which limits a spike's influence: it enters the
  median at no more than that threshold. It does not freeze the threshold, which still moves as the window does.
- The history is keyed by optimizer step. A step replayed or redone after a restart replaces its own entry, so the
  history always describes the steps behind the current weights.
- During the learning-rate warm-up there is no history yet, so the threshold is `max_grad_norm`.
- `max_grad_norm` stays as a ceiling, as in ZClip's optional maximum.

Related work: ZClip (Kumar et al., 2025) sets the threshold from an exponential moving mean and standard deviation
of the norm; AutoClip (Seetharaman et al., 2020) from a percentile of the norm history; AdaGC (2025) clips each tensor
against its own moving average. A rolling median is used here because policy-gradient norms are heavy-tailed, and a
median ignores the tail that would inflate a mean and standard deviation.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class MedianGradClip:
    ceiling: float                      # max_grad_norm
    multiple: float | None = 2.0        # None: always clip at the ceiling
    window: int = 20
    warmup_steps: int = 10              # the learning-rate warm-up
    history: dict[int, float] = field(default_factory=dict)   # optimizer step -> clipped norm

    def __post_init__(self):
        if not math.isfinite(self.ceiling) or self.ceiling <= 0:
            raise ValueError("ceiling must be finite and positive")
        if self.multiple is not None and self.multiple <= 1:
            raise ValueError("multiple must be greater than 1, or None")
        if self.window < 3 or self.warmup_steps < 0:
            raise ValueError("window must be at least 3 and warmup_steps non-negative")

    def threshold(self) -> float:
        """The clip to pass with the next optimizer step."""
        if self.multiple is None or len(self.history) < max(1, self.warmup_steps):
            return self.ceiling
        recent = [self.history[step] for step in sorted(self.history)[-self.window:]]
        return min(self.ceiling, self.multiple * statistics.median(recent))

    def observe(self, step: int, grad_norm: float | None, threshold: float) -> None:
        """Record optimizer step `step`'s pre-clip norm and the threshold it was clipped at."""
        if grad_norm is None or not math.isfinite(grad_norm) or grad_norm <= 0:
            # No usable norm says nothing about the run's scale; a replayed step's earlier entry no longer applies.
            self.history.pop(step, None)
            return
        self.history[step] = min(grad_norm, threshold)

    @classmethod
    def replay(cls, steps: Iterable[tuple[int, float | None, float]], **settings) -> MedianGradClip:
        """Rebuild the state on resume from (optimizer step, pre-clip norm, threshold) records, oldest first."""
        clip = cls(**settings)
        for step, grad_norm, threshold in steps:
            clip.observe(step, grad_norm, threshold)
        return clip
