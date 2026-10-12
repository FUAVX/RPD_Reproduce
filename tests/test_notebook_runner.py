"""Verify live output capture and bounded cleanup without model dependencies."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.notebook_runner import NotebookRunner


class NotebookRunnerTests(unittest.TestCase):
    def test_stdout_stderr_stream_before_exit_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            runner = NotebookRunner(temp, heartbeat_seconds=0.1)
            errors = []
            def task():
                try:
                    runner.run('output', [sys.executable, '-u', '-c',
                        "import sys,time; sys.stdout.write('progress-no-newline'); sys.stdout.flush(); "
                        "sys.stderr.write('stderr-marker\\n'); sys.stderr.flush(); time.sleep(2)"], 5)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=task)
            thread.start()
            deadline = time.monotonic() + 1.5
            path = Path(temp)/'output.log'
            while time.monotonic() < deadline:
                if path.exists() and 'stderr-marker' in path.read_text():
                    break
                time.sleep(0.02)
            self.assertTrue(thread.is_alive(), 'Output must persist while the child still runs')
            self.assertIn('progress-no-newline', path.read_text())
            self.assertIn('stderr-marker', path.read_text())
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertIn('WAIT output', (Path(temp)/'launcher.log').read_text())
            self.assertEqual(json.loads((Path(temp)/'runner_status.json').read_text())['status'], 'complete')

    def test_timeout_records_failure_and_stops_process(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            runner = NotebookRunner(temp)
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run('stuck_import', [sys.executable, '-u', '-c',
                    "import time; print('import-start',flush=True); time.sleep(60)"], 0.4)
            state = json.loads((Path(temp)/'runner_status.json').read_text())
            self.assertEqual(state['stage'], 'stuck_import')
            self.assertEqual(state['status'], 'failed')
            self.assertIn('import-start', (Path(temp)/'stuck_import.log').read_text())
            with self.assertRaises(ProcessLookupError):
                __import__('os').kill(state['pid'], 0)

    def test_startup_timeout_then_release_on_marker(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            runner = NotebookRunner(temp)
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run('ddp_stuck', [sys.executable, '-u', '-c',
                    "import time; print('initializing',flush=True); time.sleep(60)"], None,
                    startup_timeout_seconds=0.4, startup_marker='training-started')
            self.assertFalse(runner.state['startup_complete'])
            runner.run('ddp_ready', [sys.executable, '-u', '-c',
                "import time; print('training-started',flush=True); time.sleep(0.8)"], None,
                startup_timeout_seconds=0.4, startup_marker='training-started')
            self.assertTrue(runner.state['startup_complete'])
            self.assertEqual(runner.state['status'], 'complete')

    def test_nonzero_exit_and_capture(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            runner = NotebookRunner(temp)
            result = runner.run('revision', [sys.executable, '-c', "print('abc123')"], 5, capture=True)
            self.assertEqual(result.strip(), 'abc123')
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run('failure', [sys.executable, '-u', '-c',
                    "import sys; print('real-error',file=sys.stderr); sys.exit(7)"], 5)
            self.assertIn('real-error', (Path(temp)/'failure.log').read_text())
            self.assertEqual(json.loads((Path(temp)/'runner_status.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
