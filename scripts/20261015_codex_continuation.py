#!/usr/bin/env python3
"""Run a finite CMPC research queue from systemd, with checkpoint and quota gates.

This driver starts fresh Codex sessions. It never resumes the owner's live chat.
Research directions are in the versioned handoff note; runtime state and CLI logs
are under studies/runs/20261015-cmpc-continuation. No network quota polling.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'studies/runs/20261015-cmpc-continuation'
NOTE = ROOT / 'docs/handoff/20261015-cmpc-research.md'
CONFIG = ROOT / 'docs/handoff/20261015-cmpc-queue.json'
SCHEMA = ROOT / 'docs/handoff/20261015-cmpc-result-schema.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def latest_usage():
    """Read only rate-limit telemetry from recent local sessions."""
    db = Path.home() / '.codex/state_5.sqlite'
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as c:
            paths = [r[0] for r in c.execute(
                'select rollout_path from threads where model_provider=? '
                'order by updated_at desc limit 8', ('openai',))]
    except sqlite3.Error:
        return None
    found = []
    for path in paths:
        try:
            with open(path, 'rb') as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - 1024 * 1024))
                lines = f.read().splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if b'"rate_limits"' not in line:
                continue
            try:
                record = json.loads(line)
                limits = record.get('payload', {}).get('rate_limits')
                if limits and limits.get('limit_id') == 'codex':
                    found.append((record.get('timestamp', ''), limits))
                    break
            except (ValueError, TypeError):
                continue
    return max(found, key=lambda x: x[0])[1] if found else None


def quota_wait(limits, now):
    deadlines = []
    for name, ceiling in [('primary', 80), ('secondary', 90)]:
        window = (limits or {}).get(name) or {}
        reset = window.get('resets_at') or 0
        if window.get('used_percent', 0) >= ceiling and reset + 60 > now:
            deadlines.append(reset + 60)
    return max(deadlines, default=0)


def reconcile(state, step, result, rc, now, guard_refused=False):
    """A failed process or missing artifact cannot advance the queue."""
    state['last_returncode'] = rc
    state['last_result'] = result
    state['next_at'] = now + 15 * 60
    if guard_refused:
        state['resource_refusals'] = state.get('resource_refusals', 0) + 1
        state['last_error'] = 'Resguard refused the launch; no Codex session started'
        return state
    status = result.get('status') if isinstance(result, dict) else None
    if rc == 0 and status == 'checkpoint_complete':
        missing = [p for p in step['required_artifacts']
                   if not (ROOT / p).is_file() or (ROOT / p).stat().st_size == 0]
        if missing:
            state['last_error'] = 'Missing artifacts: ' + ', '.join(missing)
        else:
            state['completed'].append(step['id'])
            state.pop('last_error', None)
    elif rc == 0 and status == 'blocked':
        state['status'] = 'blocked'
    elif rc != 0 or status != 'continue':
        state['last_error'] = 'Worker did not return a valid checkpoint result'
        # Account limits and transient errors receive a long backoff even if
        # the CLI failed before writing fresh quota telemetry.
        state['next_at'] = now + 6 * 3600
    return state


def cleanup_tagged(tag):
    """Stop this attempt's children, including separate resguard scopes."""
    if not tag:
        return 0
    marker = ('CMPC_CONTINUATION_TAG=' + tag).encode()
    pids, scopes = [], set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if marker not in (entry / 'environ').read_bytes().split(b'\0'):
                continue
            pids.append(int(entry.name))
            for part in (entry / 'cgroup').read_text().replace('\n', '/').split('/'):
                if part.startswith('run-') and part.endswith('.scope'):
                    scopes.add(part)
        except (OSError, ValueError):
            continue
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if scopes:
        subprocess.run(['systemctl', '--user', '--no-block', 'stop', *sorted(scopes)], check=False)
    return len(pids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['status', 'tick', 'cleanup'])
    args = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    state_path = RUN / 'state.json'
    with (RUN / 'job.lock').open('a') as lock:
        if args.command == 'tick':
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('A continuation session is already running.', flush=True)
                return
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'status': 'armed', 'completed': [], 'attempts': 0,
            'next_at': config['not_before_epoch']}
        if args.command == 'cleanup':
            count = cleanup_tagged(state.get('active_tag'))
            if state['status'] == 'running':
                state['status'] = 'stopped'
                state['last_error'] = 'Service stopped before checkpoint completion'
                save(state_path, state)
            print(f'Cleaned up {count} processes belonging to this attempt.', flush=True)
            return
        limits = latest_usage()
        now = time.time()
        if args.command == 'status':
            print(json.dumps({'state': state, 'quota': limits,
                              'quota_wait_until': quota_wait(limits, now)}, indent=2))
            return
        if state['status'] in ('done', 'blocked', 'stopped'):
            return
        pending = [s for s in config['steps'] if s['id'] not in state['completed']]
        if not pending:
            state['status'] = 'done'
            save(state_path, state)
            print('All three research checkpoints are complete.', flush=True)
            return
        if state['attempts'] - state.get('resource_refusals', 0) >= config['max_launches']:
            state['status'] = 'stopped'
            save(state_path, state)
            print('Launch budget exhausted; review the saved checkpoint.', flush=True)
            return
        deadline = max(state['next_at'], quota_wait(limits, now))
        if now < deadline:
            state['status'] = 'waiting'
            state['next_at'] = deadline
            save(state_path, state)
            print('Waiting until ' + datetime.fromtimestamp(deadline).astimezone().isoformat(), flush=True)
            return
        binaries = sorted((Path.home() / '.vscode/extensions').glob('openai.chatgpt-*/bin/linux-x86_64/codex'),
                          key=lambda p: p.stat().st_mtime)
        if not binaries:
            raise RuntimeError('No installed Codex CLI')
        guard = Path.home() / '.claude/bin/resguard.sh'
        check = subprocess.run([str(guard), 'status'], capture_output=True, text=True)
        print(check.stdout, flush=True)
        if check.returncode:
            state['next_at'] = now + 15 * 60
            save(state_path, state)
            return
        step = pending[0]
        state.update(status='running', current_step=step['id'],
                     attempts=state['attempts'] + 1,
                     last_started=datetime.now(timezone.utc).isoformat())
        tag = f"attempt-{state['attempts']:02d}-{step['id']}"
        state['active_tag'] = 'cmpc-20261015-' + tag
        save(state_path, state)
        final_path = RUN / (tag + '-result.json')
        prompt = (
            'The owner authorized this finite continuation on 2026-10-15. '
            'You are a fresh research session, separate from the owner chat. '
            f'Read {NOTE} and its Read first files, then work ONLY checkpoint '
            f"{step['id']}: {step['task']}\n"
            'Complete a useful measurement within 35 minutes, with at most 20 minutes '
            'of serial numerical work. At 30 minutes save a checkpoint. '
            'Read the local quota with scripts/20261015_codex_continuation.py status '
            'after 10 minutes and before another experiment; at 80% five-hour usage '
            'save progress and return continue. Never spawn agents or queue other runs. '
            'Use resguard for every mesh or numerical job and report its events. '
            'Do not change the runner, queue config, limits, timer, certified defaults '
            'or unrelated dirty files. Update the note State and write the required '
            'result artifacts. Commit only your files; do not push. '
            'Return the supplied JSON result schema. checkpoint_complete requires '
            'the measurement and artifacts, continue means unfinished work can resume, '
            'blocked means an owner answer is required. Your output is a durable '
            'checkpoint, so state what ran and what remains.'
        )
        # Pilot: CLI 183 MiB + geometry probe 1131 MiB; x1.3 = 1708 MiB.
        # Heavy child jobs still need their own measured resguard scope.
        command = [str(guard), 'run', '--mem', '1.8G', '--cpu', '200', '--',
                   'timeout', '--signal=TERM', '--kill-after=30s', '40m',
                   'taskset', '-c', '18', str(binaries[-1]), 'exec',
                   '--cd', str(ROOT), '-m', config['model'],
                   '--sandbox', 'danger-full-access', '-c', 'approval_policy="never"',
                   '--json', '--color', 'never', '--output-schema', str(SCHEMA),
                   '--output-last-message', str(final_path), '-']
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CMPC_CONTINUATION_TAG=state['active_tag'])
        with (RUN / (tag + '.jsonl')).open('w') as output:
            proc = subprocess.run(command, input=prompt, text=True, stdout=output,
                                  stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            output.flush()
            os.fsync(output.fileno())
        try:
            result = json.loads(final_path.read_text())
        except (OSError, ValueError):
            result = {}
        with (RUN / (tag + '.jsonl')).open() as output:
            first_line = output.readline()
        guard_refused = proc.returncode == 3 and first_line.startswith('[resguard] refusing to launch:')
        state = reconcile(state, step, result, proc.returncode, time.time(), guard_refused)
        if len(state['completed']) == len(config['steps']):
            state['status'] = 'done'
        elif state['status'] == 'running':
            state['status'] = 'waiting'
        state['next_at'] = max(state['next_at'], quota_wait(latest_usage(), time.time()))
        save(state_path, state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
