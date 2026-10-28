"""Select TensorBoard's no-TensorFlow backend without altering TF/JAX installs."""
import sys
import types

def use_tensorboard_without_tensorflow():
    sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))
