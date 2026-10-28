"""Run one project script with consistent IEEE FP32 flags on Colab A100."""
import runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
from utils.ndc_colab import configure_fp32
configure_fp32()
if len(sys.argv)<2:raise SystemExit('Usage: ndc_colab_exec.py project_script.py [args]')
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
