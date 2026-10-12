"""Standard-library process logging, embedded in the shared Kaggle notebook."""
import codecs
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


class NotebookRunner:
    def __init__(self, directory, heartbeat_seconds=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.heartbeat_seconds = heartbeat_seconds
        self.state = {'status': 'ready'}
        self.note('LOGGER READY: ' + str(self.directory.resolve()))
        self._status()

    def _now(self):
        return datetime.now(timezone.utc).isoformat(timespec='seconds')

    def _status(self):
        temporary = self.directory / 'runner_status.json.tmp'
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(self.directory / 'runner_status.json')

    def _write(self, text, process_log=None):
        # Persist first, then display: notebook output capture must not lose the log.
        with (self.directory / 'launcher.log').open('a', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
        if process_log is not None:
            process_log.write(text)
            process_log.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    def note(self, message):
        self._write(f'[{self._now()}] {message}\n')

    def fail(self, stage, error):
        self.state.update(stage=stage, status='failed', ended_at=self._now(), error=str(error))
        self._status()
        self.note(f'FAILED {stage}: {error}')

    @staticmethod
    def _stop(process):
        # Terminate the entire session (including torchrun ranks), not only the parent.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The parent can exit while a descendant ignores SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)

    def run(self, stage, command, timeout_seconds, filename=None, env=None, capture=False,
            startup_timeout_seconds=None, startup_marker=None):
        if startup_timeout_seconds is not None and not startup_marker:
            raise ValueError("A startup deadline requires an output marker.")
        started = time.monotonic()
        self.state = dict(stage=stage, status='running', started_at=self._now(),
                          timeout_seconds=timeout_seconds, output_bytes=0,
                          startup_timeout_seconds=startup_timeout_seconds,
                          startup_complete=startup_timeout_seconds is None)
        self._status()
        self.note(f'START {stage}; timeout={timeout_seconds}s; command={command}')
        child_env = os.environ.copy()
        child_env.update(PYTHONUNBUFFERED='1', GIT_TERMINAL_PROMPT='0')
        if env:
            child_env.update(env)
        process = None
        collected = []
        try:
            with (self.directory / (filename or stage + '.log')).open('a', encoding='utf-8') as process_log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=0, start_new_session=True, env=child_env)
                self.state['pid'] = process.pid
                self._status()
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                startup_tail = ''
                next_heartbeat = started + self.heartbeat_seconds
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map() or process.poll() is None:
                        elapsed = time.monotonic() - started
                        if timeout_seconds is not None and elapsed >= timeout_seconds:
                            raise subprocess.TimeoutExpired(command, timeout_seconds)
                        if (startup_timeout_seconds is not None and not self.state['startup_complete']
                                and elapsed >= startup_timeout_seconds):
                            raise subprocess.TimeoutExpired(command, startup_timeout_seconds)
                        for key, _ in selector.select(timeout=0.2):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            self.state['output_bytes'] += len(chunk)
                            self.state['last_output_at'] = self._now()
                            output = decoder.decode(chunk)
                            self._write(output, process_log)
                            if startup_timeout_seconds is not None and not self.state['startup_complete']:
                                if startup_marker in startup_tail + output:
                                    self.state.update(startup_complete=True, startup_finished_at=self._now())
                                    self._status()
                                    self.note(f'STARTUP CONFIRMED {stage}: {startup_marker}')
                                startup_tail = (startup_tail + output)[-len(startup_marker):]
                            if capture:
                                collected.append(output)
                        if time.monotonic() >= next_heartbeat:
                            self.state['elapsed_seconds'] = round(time.monotonic() - started, 1)
                            self._status()
                            # This heartbeat only reports the launcher is waiting; not training progress.
                            self.note(f'WAIT {stage}: {self.state["elapsed_seconds"]}s, '
                                      f'bytes={self.state["output_bytes"]}, '
                                      f'last_output={self.state.get("last_output_at", "none")}')
                            next_heartbeat = time.monotonic() + self.heartbeat_seconds
                    output = decoder.decode(b'', final=True)
                    if output:
                        self._write(output, process_log)
                        if capture:
                            collected.append(output)
                returncode = process.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                if not self.state["startup_complete"]:
                    raise RuntimeError(f"{stage} exited without its required startup marker: {startup_marker}")
        except BaseException as error:
            self.state["elapsed_seconds"] = round(time.monotonic() - started, 1)
            if process is not None:
                self._stop(process)
            self.fail(stage, error)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
        self.state.update(status='complete', ended_at=self._now(), returncode=0,
                          elapsed_seconds=round(time.monotonic() - started, 1))
        self._status()
        self.note(f'END {stage}: {self.state["elapsed_seconds"]}s; exit=0')
        return ''.join(collected) if capture else None
