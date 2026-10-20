# Phase 3 TensorBoard no-TensorFlow startup guard.
import os
import sys
import types
if os.environ.get('P3_TENSORBOARD_NO_TF') == '1':
    sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))
