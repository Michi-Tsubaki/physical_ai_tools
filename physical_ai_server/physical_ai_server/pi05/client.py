"""Local process boundary between the recorder's LeRobot and the Pi05 runtime."""

from collections import deque
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess

import torch


class Pi05Runtime:
    def __init__(self, stop_event=None):
        python = Path(os.environ.get(
            'PHYSICAL_AI_PI05_PYTHON', '~/.local/share/physical-ai-tools/pi05/bin/python'
        )).expanduser()
        if not python.is_file():
            raise RuntimeError(f'Missing Pi05 environment at {python}. See the Pi05 setup in README.md')
        parent, child = socket.socketpair()
        with parent, child:
            self.process = subprocess.Popen(
                [str(python), '-I', str(Path(__file__).with_name('worker.py')), str(child.fileno())],
                pass_fds=(child.fileno(),),
            )
            self.connection = Connection(parent.detach())
        self.stop_event = stop_event

    def call(self, operation, **data):
        try:
            self.connection.send((operation, data))
            while not self.connection.poll(0.25):
                if self.stop_event is not None and self.stop_event.is_set():
                    raise InterruptedError('Pi05 training stopped')
            ok, result = self.connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise RuntimeError('Pi05 worker exited. Check its output and installed dependencies') from exc
        if not ok:
            raise RuntimeError(result)
        return result

    def close(self):
        self.connection.close()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def cpu_batch(batch):
    return {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key.startswith('observation.') or key in ('action', 'task')}


class Pi05Inference:
    def __init__(self, path, device):
        self.runtime = Pi05Runtime()
        self.actions = deque()
        try:
            self.config = self.runtime.call('load', source=path, device=device)
        except BaseException:
            self.close()
            raise

    def reset(self):
        self.actions.clear()

    def select_action(self, observation):
        if not self.actions:
            batch = cpu_batch(observation)
            batch.setdefault('task', [''])
            self.actions.extend(self.runtime.call('predict', batch=batch))
        return torch.from_numpy(self.actions.popleft()).unsqueeze(0)

    def close(self):
        self.runtime.close()
