"""Exercise real TensorBoard writes while rejecting optional TensorFlow/JAX imports."""

import importlib.abc
import sys
import tempfile
from pathlib import Path

from utils.tensorboard_compat import use_tensorboard_without_tensorflow


class RejectOptionalFrameworks(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tensorflow', 'jax', 'jaxlib'):
            self.attempts.append(fullname)
            raise RuntimeError('Simulated broken optional TensorFlow/JAX installation')
        return None


def test_real_tensorboard_logging_without_tensorflow():
    blocker = RejectOptionalFrameworks()
    sys.meta_path.insert(0, blocker)
    try:
        use_tensorboard_without_tensorflow()
        use_tensorboard_without_tensorflow()  # Safe to call more than once.
        from pytorch_lightning.loggers import TensorBoardLogger
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.compat import tf

        # Access a real API to resolve TensorBoard's lazy backend before checking it.
        assert tf.io.gfile.__name__ == 'tensorboard.compat.tensorflow_stub.io.gfile'
        assert tf.__name__ == 'tensorboard.compat.tensorflow_stub'
        with tempfile.TemporaryDirectory() as directory:
            logger = TensorBoardLogger(directory, name='lightning_logs', version=0, default_hp_metric=False)
            logger.log_metrics({'val_weed_iou': 0.25, 'dpp/down1/alpha/mean': 1.0}, step=1)
            logger.save()  # Reproduces the lazy initialization that failed in Kaggle.
            logger.finalize('success')
            events = EventAccumulator(logger.log_dir).Reload()
            assert events.Scalars('val_weed_iou')[0].value == 0.25
            assert events.Scalars('dpp/down1/alpha/mean')[0].value == 1.0
            assert list(Path(logger.log_dir).glob('events.out.tfevents.*'))
        assert not blocker.attempts, blocker.attempts
    finally:
        sys.meta_path.remove(blocker)
    print('PASS: real TensorBoardLogger write/flush/read without TensorFlow/JAX imports')


if __name__ == '__main__':
    test_real_tensorboard_logging_without_tensorflow()
