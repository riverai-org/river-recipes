import dataclasses
import math
import unittest

from text_to_sql_recipe.clipping import MedianGradClip


def run(clip, norms, first_step=1):
    """Drive the clip like the trainer: threshold first, then the step's pre-clip norm."""
    applied = []
    for step, g in enumerate(norms, first_step):
        t = clip.threshold()
        applied.append((step, g, t))
        clip.observe(step, g, t)
    return applied


class MedianGradClipTest(unittest.TestCase):
    def test_ceiling_during_warmup(self):
        clip = MedianGradClip(ceiling=20000.0, warmup_steps=10)
        thresholds = [t for _, _, t in run(clip, [3000.0] * 10)]
        self.assertEqual(thresholds, [20000.0] * 10)

    def test_multiple_of_recent_median_after_warmup(self):
        clip = MedianGradClip(ceiling=20000.0, multiple=2.0, window=20, warmup_steps=10)
        run(clip, [3000.0] * 10)
        self.assertEqual(clip.threshold(), 6000.0)

    def test_never_above_ceiling(self):
        clip = MedianGradClip(ceiling=20000.0, multiple=2.0, warmup_steps=3)
        run(clip, [15000.0] * 5)
        self.assertEqual(clip.threshold(), 20000.0)

    def test_spike_is_clipped_and_does_not_raise_the_threshold(self):
        clip = MedianGradClip(ceiling=20000.0, multiple=2.0, window=20, warmup_steps=10)
        run(clip, [1000.0] * 20)
        step, g, t = run(clip, [10000.0], first_step=21)[0]
        self.assertGreater(g, t)
        self.assertEqual(t, 2000.0)
        self.assertEqual(clip.history[step], 2000.0)
        self.assertEqual(clip.threshold(), 2000.0)

    def test_follows_a_slow_drift(self):
        clip = MedianGradClip(ceiling=1e9, multiple=2.0, window=20, warmup_steps=10)
        norms = [1000.0 * 1.01 ** i for i in range(200)]  # about 7x over 200 steps
        clipped = [g for _, g, t in run(clip, norms) if g > t]
        self.assertEqual(clipped, [])

    def test_fixed_when_multiple_unset(self):
        clip = MedianGradClip(ceiling=20000.0, multiple=None)
        thresholds = {t for _, _, t in run(clip, [1000.0] * 40)}
        self.assertEqual(thresholds, {20000.0})

    def test_ignores_non_finite_norms(self):
        clip = MedianGradClip(ceiling=20000.0, warmup_steps=2)
        run(clip, [1000.0, math.inf, math.nan, 0.0, 1000.0])
        self.assertEqual(sorted(clip.history), [1, 5])
        self.assertEqual(clip.threshold(), 2000.0)

    def test_threshold_uses_the_last_window_of_steps(self):
        clip = MedianGradClip(ceiling=1e9, multiple=2.0, window=5, warmup_steps=0)
        run(clip, [float(i) for i in range(1, 50)])
        self.assertEqual(clip.threshold(), 2.0 * 47.0)  # median of steps 45-49

    def test_a_replayed_or_redone_step_replaces_its_own_entry(self):
        clip = MedianGradClip(ceiling=20000.0, multiple=2.0, window=20, warmup_steps=2)
        run(clip, [1000.0, 1000.0, 9000.0])
        clip.observe(3, 1000.0, 20000.0)  # step 3 redone after a restart from step 2
        self.assertEqual(clip.history, {1: 1000.0, 2: 1000.0, 3: 1000.0})
        clip.observe(3, math.nan, 20000.0)  # and a step whose norm is unusable leaves no entry
        self.assertEqual(sorted(clip.history), [1, 2])

    def test_replay_rebuilds_the_same_state(self):
        settings = {"ceiling": 20000.0, "multiple": 2.0, "window": 20, "warmup_steps": 10}
        live = MedianGradClip(**settings)
        steps = run(live, [3000.0 + 37.0 * (i % 9) for i in range(60)])
        resumed = MedianGradClip.replay(steps, **settings)
        self.assertEqual(dataclasses.asdict(resumed), dataclasses.asdict(live))

    def test_rejects_bad_settings(self):
        for bad in ({"ceiling": 0.0}, {"ceiling": 1.0, "multiple": 1.0}, {"ceiling": 1.0, "window": 2}):
            with self.assertRaises(ValueError):
                MedianGradClip(**bad)


if __name__ == "__main__":
    unittest.main()
