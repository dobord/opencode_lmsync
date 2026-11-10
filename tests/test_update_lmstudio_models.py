import importlib.util
import unittest
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "update_lmstudio_models.py"
SPEC = importlib.util.spec_from_file_location("update_lmstudio_models", SCRIPT_PATH)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class ContextResolutionTests(unittest.TestCase):
    def test_loaded_variant_context_is_associated_with_model_key(self):
        response = {
            "models": [
                {
                    "key": "gemma-4-12b-it@iq3_xs",
                    "max_context_length": 131072,
                    "loaded_instances": [],
                },
                {
                    "key": "gemma-4-12b-it@iq4_xs",
                    "max_context_length": 131072,
                    "loaded_instances": [
                        {
                            "id": "gemma-4-12b-it",
                            "config": {"context_length": 38400},
                        }
                    ],
                },
            ]
        }

        server_models, loaded_instances, _ = sync.parse_models_response(response)

        self.assertEqual(loaded_instances, {"gemma-4-12b-it@iq4_xs": 38400})
        self.assertEqual(
            sync.get_context_for_model(
                "http://lmstudio:1234",
                "gemma-4-12b-it@iq4_xs",
                server_models,
                {},
                loaded_instances,
            ),
            38400,
        )
        self.assertEqual(
            sync.get_context_for_model(
                "http://lmstudio:1234",
                "gemma-4-12b-it@iq3_xs",
                server_models,
                {},
                loaded_instances,
            ),
            131072,
        )


RESPONSE = {"models": [{"key": "remote", "max_context_length": 8192,
                        "capabilities": {"trained_for_tool_use": True, "vision": True, "reasoning": True},
                        "loaded_instances": [{"config": {"context_length": 4096}}]}]}


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / name for name in ("config.json", "auth.json", "cache.json")]
        self.args = [item for flag, path in zip(("--config-path", "--auth-path", "--cache-path"), self.paths) for item in (flag, str(path))]
        self.write(0, {})
        self.write(1, {})
        self.write(2, {})

    def write(self, index, data):
        self.paths[index].write_text(json.dumps(data), encoding="utf-8")

    def read(self, index=0):
        return json.loads(self.paths[index].read_text(encoding="utf-8"))

    def snapshots(self):
        return [p.read_bytes() if p.exists() else None for p in self.paths]

    def run_sync(self, major=2, extra=(), response=RESPONSE):
        with patch.object(sync, "detect_opencode_version", return_value=major), patch.object(sync, "request_models", return_value=response), patch.dict(os.environ, {}, clear=True):
            return sync.main(self.args + list(extra))

    def provider(self, native=True, models=None):
        return {"settings" if native else "options": {"baseURL": "http://fixture:1234/v1"}, "models": models or {}}

    def test_gate_precedes_all_reads_and_no_writes_unknown(self):
        for version in ("3.0.0", "unknown", "", "2.1", "noise 2.0.0"):
            before = self.snapshots()
            with self.subTest(version=version), patch.object(sync.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, version, "")) as cli, patch.object(sync, "file_snapshot") as read:
                self.assertEqual(sync.main(self.args), 2)
                read.assert_not_called()
                cli.assert_called_once_with(["opencode", "--version"], shell=False, timeout=10, capture_output=True, text=True, check=True)
            self.assertEqual(before, self.snapshots())

    def test_gate_missing_timeout_nonzero(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("opencode", 10), subprocess.CalledProcessError(1, "opencode")):
            with self.subTest(error=type(error).__name__), patch.object(sync.subprocess, "run", side_effect=error), patch.object(sync, "file_snapshot") as read:
                self.assertEqual(sync.main(self.args + ["--dry-run"]), 2)
                read.assert_not_called()

    def test_versions(self):
        for text, major in (("1.2.3", 1), ("opencode 2.0.0", 2), ("v2.0.0-beta.1+build", 2),
                            ("opencode v2.2.5\n", 2), ("opencode v2.0.0-dev-20261110\r\n", 2)):
            with patch.object(sync.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, text, "")):
                self.assertEqual(sync.detect_opencode_version("fixture"), major)

    def test_native_v2_alias_preserves_settings_variants_output(self):
        model = {"modelID": "remote", "name": "Alias", "limit": {"output": 123, "input": 321}, "variants": [{"id": "fast", "settings": {"foo": "bar"}}], "capabilities": {"reasoning": {"supported": True}}}
        self.write(0, {"model": "lmstudio_work/alias", "permissions": [], "providers": {"lmstudio_work": self.provider(models={"alias": model})}})
        self.assertEqual(self.run_sync(), 0)
        cfg = self.read()
        result = cfg["providers"]["lmstudio_work"]["models"]
        self.assertEqual(set(result), {"alias"})
        self.assertEqual(result["alias"]["limit"], {"context": 4096, "output": 123, "input": 321})
        self.assertEqual(result["alias"]["variants"], model["variants"])
        self.assertEqual(result["alias"]["capabilities"], {"reasoning": {"supported": True}, "tools": True, "input": ["text", "image"], "output": ["text"]})
        self.assertFalse({"id", "type", "reasoning", "attachment", "tool_call"}.intersection(result["alias"]))
        self.assertEqual(cfg["model"], "lmstudio_work/alias")

    def test_legacy_v1_and_v2(self):
        for major in (1, 2):
            with self.subTest(major=major):
                self.write(0, {"provider": {"lmstudio_suffix": self.provider(False, {"alias": {"id": "remote"}})}})
                self.assertEqual(self.run_sync(major), 0)
                result = self.read()["provider"]["lmstudio_suffix"]
                self.assertEqual(set(result["models"]), {"alias"})
                self.assertTrue(result["models"]["alias"]["tool_call"])
                self.assertNotIn("capabilities", result["models"]["alias"])
                self.assertNotIn("type", result)
                self.assertNotIn("id", result)

    def test_mixed_top_level_maps(self):
        self.write(0, {"provider": {"lmstudio_old": self.provider(False)}, "providers": {"lmstudio_new": self.provider()}})
        self.assertEqual(self.run_sync(), 0)
        cfg = self.read()
        self.assertIn("id", cfg["provider"]["lmstudio_old"]["models"]["remote"])
        self.assertIn("modelID", cfg["providers"]["lmstudio_new"]["models"]["remote"])

    def test_v1_rejects_native_and_conflicting_ids(self):
        cases = [({"providers": {}}, 1), ({"agents": {}}, 1), ({"skills": []}, 1),
                 ({"mcp": {"servers": {}}}, 1), ({"compaction": {"keep": {"tokens": 1000}}}, 1),
                 ({"model": {"providerID": "lmstudio", "model": "remote"}}, 1),
                 ({"provider": {"lmstudio": {}}, "providers": {"lmstudio": {}}}, 2),
                 ({"providers": {"lmstudio": {"options": {}}}}, 2),
                 ({"providers": {"lmstudio": {"models": {"a": {"id": "remote"}}}}}, 2),
                 ({"providers": {"lmstudio": {"models": {"a": {"variants": {"high": {}}}}}}}, 2),
                 ({"provider": {"lmstudio": {"models": {"a": {"variants": [{"id": "high"}]}}}}}, 2),
                 ({"provider": {"lmstudio": {"models": {"a": {"modelID": "remote"}}}}}, 2)]
        for cfg, major in cases:
            with self.subTest(cfg=cfg):
                self.write(0, cfg)
                before = self.snapshots()
                self.assertEqual(self.run_sync(major), 2)
                self.assertEqual(before, self.snapshots())

    def test_corrupt_and_invalid_roots_fail_before_fetch_or_write(self):
        for index in range(3):
            for raw in (b"broken", b"[]", b"null", b'{/* comment */ "x":1}', b'{"x":1,}', b'{"x":NaN}', b'{"x":1,"x":2}'):
                with self.subTest(index=index, raw=raw):
                    for i in range(3):
                        self.write(i, {})
                    self.paths[index].write_bytes(raw)
                    before = self.snapshots()
                    with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models") as request:
                        self.assertEqual(sync.main(self.args + ["--add-connection", "fixture", "--no-autoload"]), 2)
                        request.assert_not_called()
                    self.assertEqual(before, self.snapshots())

    def test_invalid_nested_inputs(self):
        for index, data in ((0, {"providers": []}), (0, {"providers": {"lmstudio": {"models": []}}}),
                            (0, {"providers": {"lmstudio": {"models": {"x": {"limit": 1}}}}}),
                            (1, {"other": "secret"}), (2, {"fixture": {"remote": "4096"}})):
            for i in range(3):
                self.write(i, {})
            self.write(index, data)
            before = self.snapshots()
            self.assertEqual(self.run_sync(extra=["--add-connection", "fixture", "--no-autoload"]), 2)
            self.assertEqual(before, self.snapshots())

    def test_new_connections_both_versions_no_autoload(self):
        for major in (1, 2):
            for path in self.paths:
                path.unlink(missing_ok=True)
            self.assertEqual(self.run_sync(major, ["--add-connection", "fixture", "--api-key", "fixture-secret", "--no-autoload"]), 0)
            cfg = self.read()
            section = "providers" if major == 2 else "provider"
            provider = cfg[section]["lmstudio_fixture_1234"]
            self.assertEqual(provider["models"], {})
            self.assertFalse({"id", "type", "_provider_key", "_added_by_cli"}.intersection(provider))
            self.assertEqual(self.read(1), {"lmstudio_fixture_1234": {"type": "api", "key": "fixture-secret"}})
            self.assertEqual(provider["settings" if major == 2 else "options"]["baseURL"], "http://fixture:1234/v1")

    def test_add_avoids_ids_in_both_maps(self):
        self.write(0, {"provider": {"lmstudio_fixture_1234": self.provider(False)}})
        self.assertEqual(self.run_sync(extra=["--add-connection", "fixture", "--no-autoload"]), 0)
        self.assertIn("lmstudio_fixture_1234_2", self.read()["providers"])

    def test_add_never_overwrites_or_reuses_orphaned_credentials(self):
        for major in (1, 2):
            for extra in ([], ["--api-key", "new-secret"]):
                self.write(0, {})
                auth = {"lmstudio_fixture_1234": {"type": "api", "key": "old-secret"},
                        "lmstudio_fixture_1234_2": {"type": "oauth", "access": "old-access"}}
                self.write(1, auth)
                self.assertEqual(self.run_sync(major, ["--add-connection", "fixture", "--no-autoload", *extra]), 0)
                section = "providers" if major == 2 else "provider"
                self.assertEqual(set(self.read()[section]), {"lmstudio_fixture_1234_3"})
                for key, value in auth.items():
                    self.assertEqual(self.read(1)[key], value)
                if not extra:
                    self.assertEqual(self.read(1), auth)

    def test_url_keeps_explicit_scheme_authority_and_rejects_path_prefix(self):
        for raw, expected in (("fixture", "http://fixture:1234/v1"),
                              ("https://fixture/v1", "https://fixture/v1"),
                              ("http://fixture/v1", "http://fixture/v1"),
                              ("https://[::1]:444/v1", "https://[::1]:444/v1")):
            self.assertEqual(sync.build_lmstudio_config_base_url(raw), expected)
        self.assertEqual(sync.get_base_url_from_conn({"settings": {"baseURL": "https://fixture/v1"}}), "https://fixture")
        self.write(0, {"providers": {"lmstudio": {"settings": {"baseURL": "https://fixture/tenant/v1", "apiKey": "secret"}}}})
        before = self.snapshots()
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models") as request:
            self.assertEqual(sync.main(self.args), 2)
            request.assert_not_called()
        self.assertEqual(self.snapshots(), before)

    def test_hardlink_inputs_rejected_before_reads(self):
        self.paths[1].unlink()
        try:
            os.link(self.paths[0], self.paths[1])
        except OSError:
            self.skipTest("Hardlinks unavailable")
        before = self.snapshots()
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "file_snapshot") as read:
            self.assertEqual(sync.main(self.args + ["--add-connection", "fixture", "--no-autoload"]), 2)
            read.assert_not_called()
        self.assertEqual(self.snapshots(), before)

    def test_dry_run_never_writes(self):
        before = self.snapshots()
        self.assertEqual(self.run_sync(extra=["--add-connection", "fixture", "--api-key", "secret", "--dry-run"]), 0)
        self.assertEqual(before, self.snapshots())

    def test_cache_only_does_not_rewrite_config_or_auth(self):
        self.write(0, {"providers": {"lmstudio": self.provider(models={"remote": {"limit": {"context": 4096, "output": 0}}})}})
        response = {"models": [{"key": "remote", "max_context_length": 8192, "loaded_instances": [{"config": {"context_length": 4096}}]}]}
        before = self.snapshots()
        with patch.object(sync.os, "replace", wraps=sync.os.replace) as replace:
            self.assertEqual(self.run_sync(response=response), 0)
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(replace.call_args.args[1], self.paths[2])
        after = self.snapshots()
        self.assertEqual(before[:2], after[:2])
        self.assertNotEqual(before[2], after[2])

    def test_idempotence(self):
        self.write(0, {"providers": {"lmstudio": self.provider()}})
        self.assertEqual(self.run_sync(), 0)
        before = self.snapshots()
        with patch.object(sync.os, "replace") as replace:
            self.assertEqual(self.run_sync(), 0)
            replace.assert_not_called()
        self.assertEqual(before, self.snapshots())

    def test_concurrent_change_to_any_input_aborts_all_writes(self):
        for index in range(3):
            for i in range(3):
                self.write(i, {})
            self.write(0, {"providers": {"lmstudio": self.provider()}})
            before = self.snapshots()
            def request(*_):
                self.paths[index].write_text('{"external": {}}', encoding="utf-8")
                return RESPONSE
            with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=request), patch.object(sync.os, "replace") as replace:
                self.assertEqual(sync.main(self.args), 2)
                replace.assert_not_called()
            after = self.snapshots()
            for i in range(3):
                if i != index:
                    self.assertEqual(before[i], after[i])

    def test_unique_temps_and_permissions_preserved(self):
        path = self.paths[0]
        os.chmod(path, 0o600)
        snapshot = {p: sync.file_snapshot(p) for p in self.paths}
        expected = stat.S_IMODE(path.stat().st_mode)
        with patch.object(sync.os, "replace", wraps=sync.os.replace) as replace:
            sync.write_changes([(path, {"new": True})], snapshot)
            temp = Path(replace.call_args.args[0])
            self.assertEqual(temp.parent, path.parent)
            self.assertTrue(temp.name.startswith(path.name + "."))
            self.assertNotEqual(temp, path.with_suffix(".tmp"))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_post_replace_external_change_is_not_adopted_as_baseline(self):
        self.write(0, {"providers": {"lmstudio": self.provider()}})
        before = self.snapshots()
        real_replace = sync.os.replace
        calls = []
        def replace(source, target):
            calls.append(target)
            real_replace(source, target)
            self.paths[0].write_bytes(b'{"external":true}')
        with patch.object(sync.os, "replace", side_effect=replace):
            self.assertEqual(self.run_sync(), 2)
        self.assertEqual(calls, [self.paths[0]])
        self.assertEqual(self.paths[0].read_bytes(), b'{"external":true}')
        self.assertEqual(self.snapshots()[1:], before[1:])

    def test_external_change_after_final_replace_returns_failure(self):
        snapshots = {p: sync.file_snapshot(p) for p in self.paths}
        real_replace = sync.os.replace
        def replace(source, target):
            real_replace(source, target)
            target.write_bytes(b'{"external":true}')
        with patch.object(sync.os, "replace", side_effect=replace), self.assertRaises(sync.SyncError):
            sync.write_changes([(self.paths[0], {"ours": True})], snapshots)
        self.assertEqual(self.paths[0].read_bytes(), b'{"external":true}')

    def test_processing_failure_after_first_provider_no_partial_save(self):
        self.write(0, {"providers": {"lmstudio_a": self.provider(), "lmstudio_b": self.provider()}})
        before = self.snapshots()
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=[RESPONSE, ValueError("network failed")]):
            self.assertEqual(sync.main(self.args), 2)
        self.assertEqual(before, self.snapshots())

    def qwen_response(self):
        return {"models": [
            {"key": "qwen3.8-27b@iq3_xs", "max_context_length": 262144},
            {"key": "qwen3.8-27b@iq3_xxs", "max_context_length": 262144,
             "loaded_instances": [{"id": "qwen3.8-27b@iq3_xxs", "config": {"context_length": 135168}}],
             "capabilities": {"vision": True, "trained_for_tool_use": True}},
            {"key": "qwen3.8-27b@iq4_xs", "max_context_length": 262144}]}

    def test_qwen_xxs_sync_survives_offline_second_server_only_when_explicit(self):
        for native in (False, True):
            for offline_first in (False, True):
                with self.subTest(native=native, offline_first=offline_first):
                    for i in range(3):
                        self.write(i, {})
                    section = "providers" if native else "provider"
                    online = self.provider(native, {"qwen3.8-27b@iq3_xs": {}, "qwen3.8-27b@iq4_xs": {}})
                    offline = self.provider(native, {"keep": {"limit": {"output": 99}}})
                    pairs = [("lmstudio_online", online), ("lmstudio_offline", offline)]
                    if offline_first:
                        pairs.reverse()
                    self.write(0, {section: dict(pairs), "model": "unrelated/default"})
                    self.write(2, {"offline": {"keep": 1234}})
                    before = self.snapshots()
                    responses = [self.qwen_response(), sync.ServerUnavailable("offline")]
                    if offline_first:
                        responses.reverse()
                    with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=responses):
                        self.assertEqual(sync.main(self.args), 2)
                    self.assertEqual(before, self.snapshots())
                    with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=responses), self.assertLogs(level="INFO") as logs:
                        self.assertEqual(sync.main(self.args + ["--skip-unavailable"]), 0)
                    self.assertIn("lmstudio_offline", str(logs.output))
                    cfg = self.read()
                    self.assertEqual(cfg[section]["lmstudio_offline"], offline)
                    self.assertEqual(cfg["model"], "unrelated/default")
                    models = cfg[section]["lmstudio_online"]["models"]
                    self.assertEqual(models["qwen3.8-27b@iq3_xxs"]["limit"]["context"], 135168)
                    self.assertEqual(models["qwen3.8-27b@iq3_xs"]["limit"]["context"], 262144)
                    self.assertEqual(models["qwen3.8-27b@iq4_xs"]["limit"]["context"], 262144)
                    self.assertEqual(self.read(2)["offline"], {"keep": 1234})
                    self.assertEqual(before[1], self.snapshots()[1])
                    after = self.snapshots()
                    with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=responses), patch.object(sync.os, "replace") as replace:
                        self.assertEqual(sync.main(self.args + ["--skip-unavailable"]), 0)
                        replace.assert_not_called()
                    self.assertEqual(after, self.snapshots())

    def test_skip_unavailable_all_offline_never_writes(self):
        self.write(0, {"providers": {"lmstudio_a": self.provider(), "lmstudio_b": self.provider()}})
        before = self.snapshots()
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=sync.ServerUnavailable("offline")):
            self.assertEqual(sync.main(self.args + ["--skip-unavailable"]), 2)
        self.assertEqual(before, self.snapshots())

    def test_skip_unavailable_does_not_skip_invalid_response_or_auth_errors(self):
        self.write(0, {"providers": {"lmstudio_a": self.provider(), "lmstudio_b": self.provider()}})
        for second in (sync.SyncError("LM Studio вернул HTTP 401"), {"models": [None]}):
            before = self.snapshots()
            with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=[RESPONSE, second]):
                self.assertEqual(sync.main(self.args + ["--skip-unavailable"]), 2)
            self.assertEqual(before, self.snapshots())

    def test_selected_connections_only(self):
        self.write(0, {"providers": {"lmstudio_a": self.provider(), "lmstudio_b": self.provider(), "lmstudio_c": self.provider()}})
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", return_value=RESPONSE) as request:
            self.assertEqual(sync.main(self.args + ["--connection", "lmstudio_a", "--connection", "lmstudio_c"]), 0)
            self.assertEqual(request.call_count, 2)
        providers = self.read()["providers"]
        self.assertEqual(providers["lmstudio_b"], self.provider())
        self.assertIn("remote", providers["lmstudio_a"]["models"])
        self.assertIn("remote", providers["lmstudio_c"]["models"])

    def test_invalid_connection_selection_fails_before_network_or_writes(self):
        for extra in (["--connection", "typo"], ["--connection", "openai"],
                      ["--connection", "lmstudio", "--add-connection", "fixture"]):
            self.write(0, {"providers": {"lmstudio": self.provider(), "openai": {}}})
            before = self.snapshots()
            with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models") as request:
                self.assertEqual(sync.main(self.args + extra), 2)
                request.assert_not_called()
            self.assertEqual(before, self.snapshots())

    def test_partial_dry_run_never_writes(self):
        self.write(0, {"providers": {"lmstudio_a": self.provider(), "lmstudio_b": self.provider()}})
        before = self.snapshots()
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", side_effect=[RESPONSE, sync.ServerUnavailable("offline")]):
            self.assertEqual(sync.main(self.args + ["--skip-unavailable", "--dry-run"]), 0)
        self.assertEqual(before, self.snapshots())

    def test_transport_errors_are_distinct_from_http_and_malformed_json(self):
        for error in (sync.urllib.error.URLError("PRIVATE"), TimeoutError("PRIVATE"), ConnectionResetError("PRIVATE"),
                      sync.urllib.error.HTTPError("http://PRIVATE", 401, "PRIVATE", {}, None), ValueError("PRIVATE")):
            expected = sync.ServerUnavailable if isinstance(error, (sync.urllib.error.URLError, OSError)) and not isinstance(error, sync.urllib.error.HTTPError) else sync.SyncError
            with self.subTest(error=type(error).__name__), patch.object(sync.urllib.request.OpenerDirector, "open", side_effect=error):
                with self.assertRaises(sync.SyncError) as caught:
                    sync.request_models("http://fixture", "PRIVATE")
                self.assertIs(type(caught.exception), expected)
                self.assertNotIn("PRIVATE", str(caught.exception))

    def test_credential_no_arbitrary_or_host_substring_fallback(self):
        auth = {"openai": {"type": "api", "key": "NEVER"}, "fixture.evil:1234": {"type": "api", "key": "NEVER2"}}
        self.assertIsNone(sync.extract_token_from_auth(auth, {}, "http://fixture:1234", "lmstudio_fixture"))
        auth["fixture"] = {"type": "api", "key": "NEVER_HOST_ONLY"}
        self.assertIsNone(sync.extract_token_from_auth(auth, {}, "https://fixture", "lmstudio_fixture"))
        auth["fixture:1234"] = {"type": "api", "key": "correct"}
        self.assertEqual(sync.extract_token_from_auth(auth, {}, "http://fixture:1234", "lmstudio_fixture"), "correct")
        auth["lmstudio_fixture"] = {"type": "api", "key": "preferred"}
        self.assertEqual(sync.extract_token_from_auth(auth, {}, "http://fixture:1234", "lmstudio_fixture"), "preferred")

    def test_credential_from_settings_env_and_logs_redacted(self):
        self.write(0, {"providers": {"lmstudio": {"settings": {"baseURL": "http://fixture:1234", "apiKey": "{env:FIXTURE_KEY}"}}}})
        self.write(1, {"openai": {"type": "api", "key": "UNRELATED_SECRET"}})
        with patch.object(sync, "detect_opencode_version", return_value=2), patch.object(sync, "request_models", return_value=RESPONSE) as request, patch.dict(os.environ, {"FIXTURE_KEY": "RIGHT_SECRET"}, clear=True), self.assertLogs(level="INFO") as logs:
            self.assertEqual(sync.main(self.args + ["--verbose"]), 0)
            self.assertEqual(request.call_args.args[1], "RIGHT_SECRET")
        self.assertNotIn("RIGHT_SECRET", str(logs.output))
        self.assertNotIn("UNRELATED_SECRET", str(logs.output))

    def test_unexpected_exception_cannot_log_secret(self):
        before = self.snapshots()
        with patch.object(sync, "detect_opencode_version", side_effect=ValueError("DO_NOT_LOG_SECRET")), self.assertLogs(level="ERROR") as logs:
            self.assertEqual(sync.main(self.args), 2)
        self.assertNotIn("DO_NOT_LOG_SECRET", str(logs.output))
        self.assertEqual(before, self.snapshots())

    def test_missing_env_key_and_oauth_not_used(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(sync.SyncError):
                sync.extract_token_from_conn({"settings": {"apiKey": "{env:MISSING}"}})
        self.assertIsNone(sync.extract_token_from_auth({"lmstudio": {"type": "oauth", "access": "NEVER"}}, {}, None, "lmstudio"))

    def test_write_failure_returns_nonzero_and_cleans_staging(self):
        self.write(0, {"providers": {"lmstudio": self.provider()}})
        before = self.snapshots()
        with patch.object(sync.os, "replace", side_effect=OSError("private details")):
            self.assertEqual(self.run_sync(), 2)
        self.assertEqual(before, self.snapshots())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_later_replace_failure_is_not_reported_success(self):
        self.write(0, {"providers": {"lmstudio": self.provider()}})
        before = self.snapshots()
        real_replace = sync.os.replace
        def replace(source, target):
            if target == self.paths[2]:
                raise OSError("fixture write failure")
            return real_replace(source, target)
        with patch.object(sync.os, "replace", side_effect=replace):
            self.assertEqual(self.run_sync(), 2)
        self.assertNotEqual(before[0], self.snapshots()[0])
        self.assertEqual(before[2], self.snapshots()[2])
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_redirect_is_refused_without_reusing_bearer(self):
        handler = sync.NoRedirect()
        with self.assertRaises(sync.SyncError):
            handler.redirect_request(None, None, 302, "fixture", {}, "http://unrelated.invalid")

    def test_false_vision_and_tool_metadata_native(self):
        self.write(0, {"providers": {"lmstudio": self.provider(models={"a": {"modelID": "remote"}, "b": {"modelID": "remote", "limit": {"output": 7}}})}})
        response = {"models": [{"key": "remote", "max_context_length": 4096, "capabilities": {"vision": False, "trained_for_tool_use": False, "reasoning": False}}]}
        self.assertEqual(self.run_sync(response=response), 0)
        models = self.read()["providers"]["lmstudio"]["models"]
        self.assertEqual(set(models), {"a", "b"})
        for info in models.values():
            self.assertEqual(info["capabilities"], {"tools": False, "input": ["text"], "output": ["text"]})
        self.assertEqual(models["b"]["limit"]["output"], 7)

    def test_no_fuzzy_context_match(self):
        self.assertIsNone(sync.get_context_for_model("host", "model", {}, {"host": {"model-other": 2048}}, {"model-other": 4096}))

    def test_discovery_does_not_overwrite_conflicting_alias(self):
        self.write(0, {"providers": {"lmstudio": self.provider(models={"remote": {"modelID": "different-remote", "name": "Keep this alias"}})}})
        before = self.snapshots()
        self.assertEqual(self.run_sync(), 2)
        self.assertEqual(before, self.snapshots())

    def test_cache_only_with_absent_models_keeps_config_bytes(self):
        self.write(0, {"providers": {"lmstudio": {"settings": {"baseURL": "http://fixture:1234"}}}})
        before = self.snapshots()
        response = {"models": [], "loaded_instances": [{"id": "remote", "config": {"context_length": 4096}}]}
        self.assertEqual(self.run_sync(response=response), 0)
        self.assertEqual(before[0], self.snapshots()[0])
        self.assertEqual(self.read(2), {"http://fixture:1234": {"remote": 4096}})

    def test_cli_subprocess_with_real_http_fixture(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner):
                if inner.path != "/api/v1/models" or inner.headers.get("Authorization") != "Bearer fixture-key":
                    inner.send_error(400)
                    return
                inner.send_response(200)
                inner.end_headers()
                inner.wfile.write(json.dumps(RESPONSE).encode())

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        fake = self.root / ("opencode.cmd" if os.name == "nt" else "opencode")
        fake.write_text("@echo off\necho 2.0.0\n" if os.name == "nt" else "#!/bin/sh\nprintf '2.0.0\\n'\n", encoding="utf-8")
        if os.name != "nt":
            fake.chmod(0o700)
        env = dict(os.environ, LM_API_TOKEN="fixture-key")
        result = subprocess.run([sys.executable, str(SCRIPT_PATH), *self.args, "--opencode-bin", str(fake),
                                 "--add-connection", "http://127.0.0.1:%d" % server.server_port], env=env, capture_output=True, text=True, timeout=30, shell=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        provider = next(iter(self.read()["providers"].values()))
        self.assertEqual(provider["models"]["remote"]["modelID"], "remote")
        self.assertEqual(provider["models"]["remote"]["limit"]["context"], 4096)
        self.assertNotIn("fixture-key", result.stdout + result.stderr)

    def test_cli_subprocess_partial_sync_with_real_disconnected_server(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner):
                if inner.server.offline:
                    return  # Close without a response: a real transport error.
                inner.send_response(200)
                inner.end_headers()
                inner.wfile.write(json.dumps(self.qwen_response()).encode())

            def log_message(self, *_):
                pass

        providers = {}
        for name, offline in (("lmstudio_online", False), ("lmstudio_offline", True)):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.offline = offline
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            providers[name] = {"settings": {"baseURL": "http://127.0.0.1:%d/v1" % server.server_port}, "models": {}}
        self.write(0, {"providers": providers})
        before = self.snapshots()
        fake = self.root / ("opencode.cmd" if os.name == "nt" else "opencode")
        fake.write_text("@echo off\necho 2.0.0\n" if os.name == "nt" else "#!/bin/sh\nprintf '2.0.0\\n'\n", encoding="utf-8")
        if os.name != "nt":
            fake.chmod(0o700)
        command = [sys.executable, str(SCRIPT_PATH), *self.args, "--opencode-bin", str(fake)]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        strict = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(strict.returncode, 2, strict.stderr)
        self.assertIn("lmstudio_offline", strict.stderr)
        self.assertEqual(before, self.snapshots())
        partial = subprocess.run(command + ["--skip-unavailable"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(partial.returncode, 0, partial.stderr)
        self.assertIn("lmstudio_offline", partial.stderr)
        result = self.read()["providers"]
        self.assertEqual(result["lmstudio_online"]["models"]["qwen3.8-27b@iq3_xxs"]["limit"]["context"], 135168)
        self.assertEqual(result["lmstudio_offline"], providers["lmstudio_offline"])
        self.assertEqual(before[1], self.snapshots()[1])


if __name__ == "__main__":
    unittest.main()
