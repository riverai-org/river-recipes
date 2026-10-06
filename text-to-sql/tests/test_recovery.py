import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from text_to_sql_recipe import recovery


class Transient(Exception):
    pass


class Fatal(Exception):
    pass


def classify(error):
    return "serving" if isinstance(error, Transient) else None


class Harness:
    """A run directory with one saved trainer state, and a scripted `run` that raises or returns."""

    def __init__(self, directory, model_step=29):
        self.root = Path(directory)
        self.state = self.root / "epochs/001/trainer.json"
        self.state.parent.mkdir(parents=True)
        self.save(model_step)
        self.calls = 0
        self.sleeps = []

    def save(self, model_step):
        self.state.write_text(json.dumps({"model_step": model_step, "base": {"path": "checkpoint"}}))

    def recover(self, script):
        def run():
            self.calls += 1
            return script(self.calls)

        return recovery.run_with_recovery(run, self.root, "run-1", classify, sleep=self.sleeps.append)

    def events(self):
        return json.loads((self.root / recovery.RECEIPT).read_text())["recoveries"]


class RunWithRecoveryTest(unittest.TestCase):
    def test_recovers_and_leaves_saved_state_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            h = Harness(directory, model_step=57)
            saved = h.state.read_text()

            def script(call):
                if call == 1:
                    raise Transient("inference_worker_retry_exhausted: worker failed after 10 attempts")
                return "completed"

            self.assertEqual(h.recover(script), "completed")
            self.assertEqual(h.calls, 2)
            self.assertEqual(h.sleeps, [recovery.RECOVERY_DELAYS_SECONDS[0]])
            self.assertEqual(h.state.read_text(), saved)
            [event] = h.events()
            self.assertEqual((event["kind"], event["model_step"], event["attempt_at_step"]), ("serving", 57, 1))

    def test_unclassified_errors_propagate_without_a_receipt(self):
        for error in (Fatal("bad provenance"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                h = Harness(directory)

                def script(call, error=error):
                    raise error

                with self.assertRaises(type(error)):
                    h.recover(script)
                self.assertEqual(h.calls, 1)
                self.assertFalse((h.root / recovery.RECEIPT).exists())

    def test_before_any_training_the_run_restarts_from_the_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def run():
                calls.append(1)
                if len(calls) == 1:
                    raise Transient("inference_worker_retry_exhausted: failed during the base evaluation")
                return "completed"

            self.assertEqual(recovery.run_with_recovery(run, root, "run-1", classify, sleep=lambda _: None),
                             "completed")
            [event] = json.loads((root / recovery.RECEIPT).read_text())["recoveries"]
            self.assertEqual((event["model_step"], event["trainer_state"]), (0, None))

    def test_steps_reported_past_the_saved_state_no_longer_block_a_recovery(self):
        # The next attempt reconciles the run directory with the saved state (see ReconcileTest).
        with tempfile.TemporaryDirectory() as directory:
            h = Harness(directory, model_step=28)
            (h.root / "steps.jsonl").write_text(json.dumps({"batch": 29, "model_step": 29}) + "\n")

            def script(call):
                if call == 1:
                    raise Transient("unavailable")
                return "completed"

            self.assertEqual(h.recover(script), "completed")
            [event] = h.events()
            self.assertEqual(event["model_step"], 28)

    def test_a_torn_final_step_record_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            path.write_text(json.dumps({"model_step": 1}) + "\n" + '{"model_st')
            self.assertEqual(recovery.recorded_steps(path), [{"model_step": 1}])
            path.write_text('{"model_st\n' + json.dumps({"model_step": 1}) + "\n")
            with self.assertRaises(json.JSONDecodeError):
                recovery.recorded_steps(path)

    def test_a_torn_step_log_is_repaired_before_the_next_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            path.write_text(json.dumps({"model_step": 1}) + "\n" + '{"model_st')
            quarantine = recovery.repair_step_log(path)
            self.assertEqual(quarantine.read_text(), '{"model_st')
            self.assertEqual(path.read_text(), json.dumps({"model_step": 1}) + "\n")
            with path.open("a") as stream:  # the next records append cleanly
                stream.write(json.dumps({"model_step": 2}) + "\n" + json.dumps({"model_step": 3}) + "\n")
            self.assertEqual([r["model_step"] for r in recovery.recorded_steps(path)], [1, 2, 3])
            self.assertIsNone(recovery.repair_step_log(path))

    def test_a_complete_final_record_only_gains_its_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            path.write_text(json.dumps({"model_step": 1}) + "\n" + json.dumps({"model_step": 2}))
            self.assertIsNone(recovery.repair_step_log(path))
            self.assertEqual([r["model_step"] for r in recovery.recorded_steps(path)], [1, 2])
            self.assertTrue(path.read_text().endswith("\n"))
            self.assertEqual(list(Path(directory).glob("*.torn-*")), [])
            self.assertIsNone(recovery.repair_step_log(Path(directory) / "missing.jsonl"))

    def test_attempts_without_progress_back_off_then_stop_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            h = Harness(directory)
            attempts = len(recovery.RECOVERY_DELAYS_SECONDS)

            def script(call):
                raise Transient("scan_window_expired: pending for over 2 hours")

            with self.assertRaises(Transient):
                h.recover(script)
            self.assertEqual(h.calls, attempts + 1)
            self.assertEqual(h.sleeps, list(recovery.RECOVERY_DELAYS_SECONDS))
            # A restarted driver inherits the attempts already spent at this step.
            with self.assertRaises(Transient):
                h.recover(script)
            self.assertEqual(h.calls, attempts + 2)
            self.assertEqual(len(h.events()), attempts)

    def test_progress_resets_the_attempts_at_a_step(self):
        with tempfile.TemporaryDirectory() as directory:
            h = Harness(directory, model_step=30)
            failures = len(recovery.RECOVERY_DELAYS_SECONDS) + 2

            def script(call):
                if call > failures:
                    return "completed"
                if call > 1:
                    h.save(30 + call - 1)  # each restart commits one step before failing again
                raise Transient("inference_worker_retry_exhausted: failed")

            self.assertEqual(h.recover(script), "completed")
            self.assertEqual(set(h.sleeps), {recovery.RECOVERY_DELAYS_SECONDS[0]})
            self.assertEqual(len(h.events()), failures)
            self.assertEqual({event["attempt_at_step"] for event in h.events()}, {1})

    def test_total_recoveries_are_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            h = Harness(directory, model_step=0)

            def script(call):
                h.save(call)
                raise Transient("inference_worker_retry_exhausted: failed")

            with self.assertRaises(Transient):
                h.recover(script)
            self.assertEqual(h.calls, recovery.RECOVERY_LIMIT + 1)
            self.assertEqual(len(h.events()), recovery.RECOVERY_LIMIT)

    def test_rejects_a_receipt_from_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            h = Harness(directory)
            (h.root / recovery.RECEIPT).write_text(json.dumps({"run_id": "other", "recoveries": []}))
            with self.assertRaises(ValueError):
                h.recover(lambda call: "completed")
            self.assertEqual(h.calls, 0)


def write_state(root, epoch, n, model_step):
    state = root / "epochs" / f"{epoch:03d}" / "trainer.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"n": n, "model_step": model_step, "base": {"path": "checkpoint"}}))


def write_lines(path, records, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)


class ReconcileTest(unittest.TestCase):
    """A hard kill can leave reported steps the trainer never saved; every start sets them aside."""

    def test_progress_comes_from_the_newest_saved_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(recovery.saved_progress(root, 33), (0, 0))
            write_state(root, 1, 33, 33)
            write_state(root, 2, 4, 37)
            self.assertEqual(recovery.saved_progress(root, 33), (37, 37))

    def test_everything_past_the_saved_progress_is_set_aside(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_state(root, 1, 2, 2)
            write_lines(root / recovery.STEP_LOG, [{"batch": b, "model_step": b} for b in (1, 2, 3, 4)])
            write_lines(root / recovery.OPTIMIZER_LOG, [{"model_step": s, "grad_norm": 1.0, "grad_clip_norm": 9.0}
                                                        for s in (1, 2, 3, 4)], tail='{"model_st')
            for batch in (0, 2, 4):
                (root / "pending-evaluations").mkdir(exist_ok=True)
                (root / "pending-evaluations" / f"{batch:04d}.json").write_text("{}")
                (root / f"checkpoint-{batch:04d}.json").write_text("{}")
                (root / "eval" / f"validation-step-{batch:04d}").mkdir(parents=True)
            (root / "collapse-state.json").write_text(json.dumps({"batch": 5, "streak": 1}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recovery.reconcile(root, 33), (2, 2))
            self.assertEqual([r["batch"] for r in recovery.recorded_steps(root / recovery.STEP_LOG)], [1, 2])
            self.assertEqual([r["model_step"] for r in recovery.recorded_steps(root / recovery.OPTIMIZER_LOG)], [1, 2])
            self.assertEqual(sorted(p.name for p in (root / "pending-evaluations").iterdir()), ["0000.json", "0002.json"])
            self.assertEqual(sorted(p.name for p in root.glob("checkpoint-*.json")),
                             ["checkpoint-0000.json", "checkpoint-0002.json"])
            self.assertEqual(sorted(p.name for p in (root / "eval").iterdir()),
                             ["validation-step-0000", "validation-step-0002"])
            self.assertFalse((root / "collapse-state.json").exists())
            [aside] = (root / recovery.ROLLED_BACK).iterdir()
            self.assertTrue((aside / "eval" / "validation-step-0004").is_dir())
            self.assertIn('"batch": 3', (aside / recovery.STEP_LOG).read_text())
            # Nothing is moved when the directory already matches the saved progress.
            with contextlib.redirect_stdout(io.StringIO()) as out:
                recovery.reconcile(root, 33)
            self.assertEqual(out.getvalue(), "")

    def test_a_collapse_state_one_batch_ahead_is_kept(self):
        # The guard records the batch it is about to train before the trainer saves the previous one.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_state(root, 1, 2, 2)
            (root / "collapse-state.json").write_text(json.dumps({"batch": 3, "streak": 1}))
            recovery.reconcile(root, 33)
            self.assertTrue((root / "collapse-state.json").exists())

    def test_the_optimizer_log_keeps_the_latest_record_of_a_step(self):
        with tempfile.TemporaryDirectory() as directory:
            log = recovery.OptimizerLog(Path(directory) / recovery.OPTIMIZER_LOG)
            log.record(1, {"grad_norm": 100.0, "grad_norm_finite": 1.0}, 20000.0)
            log.record(2, {"grad_norm": float("nan"), "grad_norm_finite": 0.0}, 20000.0)
            log.record(1, {"grad_norm": 120.0, "grad_norm_finite": 1.0}, 20000.0)  # replayed after a resume
            entries = recovery.OptimizerLog(log.path).load(through_step=2)
            self.assertEqual([(e["model_step"], e["grad_norm"]) for e in entries], [(1, 120.0), (2, None)])
            self.assertEqual(recovery.OptimizerLog(log.path).load(through_step=1)[0]["grad_norm"], 120.0)


class ServingFailureTest(unittest.TestCase):
    def test_classification(self):
        try:
            import river_client as river
        except ImportError:
            self.skipTest("river_client is not installed")
        recoverable = [
            river.RiverError("inference_worker_retry_exhausted: Inference worker failed after 10 attempts"),
            river.RiverError("inference_processor_lost: worker was lost 3 times"),
            river.RiverError("scan_window_expired: pending for over 2 hours"),
            river.RiverConnectionError("connection refused", status_code="UNAVAILABLE"),
            river.RiverConnectionError("bad gateway", status_code="502"),
            river.SessionHeartbeatError("Session heartbeat lost after 300.0s", status_code="UNAVAILABLE"),
            river.CapacityError("No capacity available"),
        ]
        fatal = [
            river.RiverError("invalid_input: prompt exceeds the context window"),
            river.RiverError("internal_error: Non-finite global gradient norm"),
            river.RiverError("Invalid sampling snapshot: unknown or duplicate handle"),
            river.RiverConnectionError("server error", status_code="INTERNAL"),
            river.AuthenticationError("invalid API key"),
            ValueError("inference_worker_retry_exhausted: not a River error"),
        ]
        for error in recoverable:
            with self.subTest(error=str(error)):
                self.assertTrue(recovery.is_serving_failure(error))
        for error in fatal:
            with self.subTest(error=str(error)):
                self.assertFalse(recovery.is_serving_failure(error))


if __name__ == "__main__":
    unittest.main()
