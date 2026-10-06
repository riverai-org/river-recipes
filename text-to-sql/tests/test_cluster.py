import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_to_sql_recipe import cluster


class JobManifestTest(unittest.TestCase):
    def spec(self, **kw):
        base = {"name": "glm-recipe", "image": "example.com/text-to-sql-recipe:0.1", "recipe": "glm-5.3-flash",
                    "volume_claim": "text-to-sql"}
        return cluster.JobSpec(**{**base, **kw})

    def container(self, manifest):
        return manifest["spec"]["template"]["spec"]["containers"][0]

    def test_manifest_takes_every_cluster_value_from_the_caller(self):
        m = cluster.manifest(self.spec(namespace="ml", api_key_secret="k", volume_claim="runs",
                                       node_selector={"pool": "cpu"}, overrides=["training.seed=7"]))
        pod = m["spec"]["template"]["spec"]
        container = pod["containers"][0]
        self.assertEqual(m["metadata"]["namespace"], "ml")
        self.assertEqual(container["env"][0]["valueFrom"]["secretKeyRef"], {"name": "k", "key": "RIVER_API_KEY"})
        self.assertEqual(pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "runs")
        self.assertEqual(pod["nodeSelector"], {"pool": "cpu"})
        self.assertEqual(container["command"][-2:], ["--set", "training.seed=7"])
        self.assertNotIn("gpu", json.dumps(m).lower())   # the driver needs no accelerator

    def test_the_run_name_is_the_job_name_so_a_restarted_pod_resumes(self):
        command = cluster.manifest(self.spec())["spec"]["template"]["spec"]["containers"][0]["command"]
        self.assertEqual(command[command.index("--run-name") + 1], "glm-recipe")

    def test_a_volume_claim_is_required(self):
        with self.assertRaisesRegex(ValueError, "volume claim"):
            cluster.manifest(self.spec(volume_claim=""))

    def test_data_runs_and_database_copies_live_on_the_volume(self):
        env = {e["name"]: e.get("value") for e in self.container(cluster.manifest(self.spec()))["env"]}
        self.assertEqual((env["TEXT_TO_SQL_DATA_DIR"], env["TEXT_TO_SQL_OUTPUT_DIR"],
                          env["TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT"]),
                         ("/workspace/data", "/workspace/runs", "/workspace/sqlite-snapshots"))

    def test_a_finished_job_is_kept_for_a_week(self):
        self.assertEqual(cluster.manifest(self.spec())["spec"]["ttlSecondsAfterFinished"], 7 * 24 * 3600)

    def test_resources_follow_the_verifier_workers_unless_given(self):
        def resources(**kw):
            return self.container(cluster.manifest(self.spec(**kw)))["resources"]
        disk = {"ephemeral-storage": "24Gi"}
        self.assertEqual(resources(), {"requests": {"cpu": "4", "memory": "24Gi", **disk},
                                       "limits": {"cpu": "4", "memory": "24Gi"}})
        verieql = {"uses_verieql": True, "verifier_workers": 16, "verieql_manifest": "/m.json",
                       "verieql_manifest_sha256": "ab" * 32}
        self.assertEqual(resources(**verieql), {"requests": {"cpu": "12", "memory": "48Gi", **disk},
                                                "limits": {"cpu": "20", "memory": "48Gi"}})
        self.assertEqual(resources(**verieql, cpu="32", memory="96Gi"),
                         {"requests": {"cpu": "32", "memory": "96Gi", **disk},
                          "limits": {"cpu": "32", "memory": "96Gi"}})
        # Workers count only when the recipe uses VeriEQL.
        self.assertEqual(resources(verifier_workers=16)["limits"]["memory"], "24Gi")

    def test_the_data_job_runs_setup_data_into_the_volume(self):
        m = cluster.data_manifest(self.spec(recipe=None, name="fill-data"))
        container = self.container(m)
        self.assertEqual(container["command"],
                         ["python", "-m", "text_to_sql_recipe", "setup", "data", "--data-dir", "/workspace/data"])
        self.assertEqual(container["env"], [])   # no API key: it only downloads
        self.assertEqual(m["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"],
                         "text-to-sql")
        self.assertEqual(m["metadata"]["labels"]["app.kubernetes.io/component"], "setup-data")

    def test_invalid_names_are_rejected(self):
        with self.assertRaises(ValueError):
            cluster.manifest(self.spec(name="Bad_Name"))

    def test_a_verieql_recipe_needs_its_runtime_manifest(self):
        with self.assertRaisesRegex(ValueError, "VeriEQL"):
            cluster.manifest(self.spec(uses_verieql=True))
        with self.assertRaisesRegex(ValueError, "VeriEQL"):
            cluster.manifest(self.spec(uses_verieql=True, verieql_manifest="/opt/verieql/manifest.json"))
        m = cluster.manifest(self.spec(uses_verieql=True, verieql_manifest="/opt/verieql/manifest.json",
                                       verieql_manifest_sha256="ab" * 32))
        env = {e["name"]: e.get("value") for e in m["spec"]["template"]["spec"]["containers"][0]["env"]}
        self.assertEqual(env["TEXT_TO_SQL_VERIEQL_MANIFEST"], "/opt/verieql/manifest.json")
        self.assertEqual(env["TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256"], "ab" * 32)

    def test_render_job_reads_the_verifier_from_the_recipe(self):
        import contextlib
        import io

        from text_to_sql_recipe import __main__
        base = ["render-job", "--name", "glm-recipe", "--image", "example.com/r:0.1", "--volume-claim", "runs"]
        # A recipe that uses VeriEQL needs its manifest: a usage error (exit 2) naming it, not a traceback.
        err = io.StringIO()
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(err):
            __main__.main([*base, "--recipe", "glm-5.3-flash"])
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("VeriEQL", err.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            __main__.main([*base, "--recipe", "glm-5.3-flash", "--set", 'reward.verifier="none"'])
        self.assertNotIn("TEXT_TO_SQL_VERIEQL_MANIFEST", out.getvalue())

    def test_render_job_reports_bad_input_as_usage_errors(self):
        import contextlib
        import io

        from text_to_sql_recipe import __main__
        base = ["render-job", "--name", "r", "--image", "example.com/r:0.1", "--volume-claim", "runs"]
        for argv, message in (
            ([*base, "--recipe", "glm-5.3-flash", "--set", 'reward.verifier="none"', "--node-selector", "pool"],
             "KEY=VALUE"),
            ([*base, "--recipe", "glm-5.3-flash", "--set", 'reward.verifier="none"', "--set",
              'data.train="no-such-dataset"'], "data.train must name a train split"),
            ([*base, "--recipe", "no-such-preset"], "no preset of that name"),
            (base, "--recipe is required"),
            (["render-job", "--name", "r", "--image", "x", "--recipe", "glm-5.3-flash"], "--volume-claim"),
        ):
            with self.subTest(argv=argv):
                err = io.StringIO()
                with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(err):
                    __main__.main(argv)
                self.assertEqual(stop.exception.code, 2)
                self.assertIn(message, err.getvalue())

    def test_render_job_setup_data_needs_no_recipe(self):
        import contextlib
        import io

        from text_to_sql_recipe import __main__
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            __main__.main(["render-job", "--setup-data", "--name", "fill-data", "--image", "example.com/r:0.1",
                           "--volume-claim", "runs"])
        self.assertIn('"setup",', out.getvalue())


if __name__ == "__main__":
    unittest.main()
