"""Use TensorBoard's supported no-TensorFlow compatibility path for PyTorch.

TensorBoard otherwise probes any installed TensorFlow, which can pull in an
incompatible optional JAX/NumPy stack on Kaggle. Only TensorBoard's backend
selection is affected; TensorFlow/JAX installations are not modified.
Reference: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/__init__.py
"""

import sys
import types


def use_tensorboard_without_tensorflow():
    """Register TensorBoard's `compat.notf` marker before importing SummaryWriter."""
    name = 'tensorboard.compat.notf'
    sys.modules.setdefault(name, types.ModuleType(name))
