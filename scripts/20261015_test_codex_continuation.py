#!/usr/bin/env python3
"""Offline queue tests. No Codex invocation, model usage or systemd mutation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    'continuation', Path(__file__).with_name('20261015_codex_continuation.py'))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class QueueTests(unittest.TestCase):
    def test_quota_refresh_releases_old_deadline(self):
        state = {'retry_at': 1100, 'next_at': 9000}
        old = {'secondary': {'used_percent': 91, 'resets_at': 8940}}
        fresh = {'secondary': {'used_percent': 2, 'resets_at': 10000}}
        self.assertEqual(driver.launch_deadline(state, old, 1200), 9000)
        self.assertEqual(driver.launch_deadline(state, fresh, 1200), 1100)
        # An account refresh must preserve a genuine retry delay.
        state['retry_at'] = 2000
        self.assertEqual(driver.launch_deadline(state, fresh, 1200), 2000)

    def test_extended_budget_and_prompt(self):
        config = json.loads(driver.CONFIG.read_text())
        state = {'attempts': 11, 'resource_refusals': 2, 'startup_failures': 1,
                 'completed': ['geometry', 'foot_motion', 'state_frames', 'motion_transfer']}
        progress = driver.queue_progress(config, state)
        self.assertEqual(progress['sessions_remaining'], 10)
        self.assertEqual(progress['pending'][0], 'foot_model_transfer')
        for step in config['steps']:
            prompt = driver.worker_prompt(config, step)
            self.assertIn('without RL-based robustification', prompt)
            self.assertIn(step['task'], prompt)

    def test_quota(self):
        self.assertEqual(driver.quota_wait(None, 1000), 0)
        self.assertEqual(driver.quota_wait({'primary': {
            'used_percent': 80, 'resets_at': 1200}}, 1000), 1260)
        self.assertEqual(driver.quota_wait({'primary': {
            'used_percent': 99, 'resets_at': 900}}, 1000), 0)
        self.assertEqual(driver.quota_wait({
            'primary': {'used_percent': 85, 'resets_at': 1200},
            'secondary': {'used_percent': 90, 'resets_at': 2200}}, 1000), 2260)

    def test_result_gate(self):
        original = driver.ROOT
        with tempfile.TemporaryDirectory() as path:
            driver.ROOT = Path(path)
            try:
                step = {'id': 'probe', 'required_artifacts': ['result.txt']}
                def fresh():
                    return {'status': 'running', 'completed': []}
                result = {'status': 'checkpoint_complete'}
                failed = driver.reconcile(fresh(), step, result, 0, 1000)
                self.assertEqual(failed['completed'], [])
                self.assertIn('Missing artifacts', failed['last_error'])
                (driver.ROOT / 'result.txt').write_text('Measured result')
                self.assertEqual(driver.reconcile(fresh(), step, result, 0, 1000)['completed'], ['probe'])
                failed = driver.reconcile(fresh(), step, result, 1, 1000)
                self.assertEqual(failed['completed'], [])
                self.assertEqual(failed['next_at'], 22600)
                self.assertEqual(driver.reconcile(fresh(), step, {'status': 'blocked'}, 0, 1000)['status'], 'blocked')
                self.assertEqual(driver.reconcile(fresh(), step, {'status': 'continue'}, 0, 1000)['completed'], [])
            finally:
                driver.ROOT = original

    def test_save(self):
        with tempfile.TemporaryDirectory() as path:
            dst = Path(path) / 'state.json'
            driver.save(dst, {'checkpoint': 1})
            self.assertEqual(json.loads(dst.read_text()), {'checkpoint': 1})
            self.assertFalse(dst.with_suffix('.tmp').exists())

    def test_resource_refusal(self):
        state = {'status': 'running', 'attempts': 2, 'completed': [], 'resource_refusals': 1}
        step = {'id': 'geometry', 'required_artifacts': []}
        result = driver.reconcile(state, step, {}, 3, 1000, guard_refused=True)
        self.assertEqual(result['next_at'], 1900)
        self.assertEqual(result['resource_refusals'], 2)
        self.assertEqual(result['attempts'] - result['resource_refusals'], 0)
        self.assertEqual(result['completed'], [])

    def test_cleanup_only_own_attempt(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            for pid, marker in [('101', b'CMPC_CONTINUATION_TAG=our-attempt\0'),
                                ('102', b'CMPC_CONTINUATION_TAG=another-attempt\0')]:
                entry = root / pid
                entry.mkdir()
                (entry / 'environ').write_bytes(marker)
                (entry / 'cgroup').write_text('0::/user.slice/app.slice/run-test.scope\n')
            with mock.patch.object(driver, 'Path', return_value=root), \
                 mock.patch.object(driver.os, 'kill') as kill, \
                 mock.patch.object(driver.subprocess, 'run') as run:
                self.assertEqual(driver.cleanup_tagged('our-attempt'), 1)
                kill.assert_called_once_with(101, driver.signal.SIGTERM)
                run.assert_called_once_with(['systemctl', '--user', '--no-block',
                                             'stop', 'run-test.scope'], check=False)


if __name__ == '__main__':
    unittest.main()
