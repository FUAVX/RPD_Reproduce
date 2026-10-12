"""Regression: both timestamped rank log and direct training.log update live."""
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multi_metric_train import TeeStream


class TrainingLogTests(unittest.TestCase):
    def test_writes_flush_to_both_logs_each_time(self):
        console, rank_log, latest_log = io.StringIO(), io.StringIO(), io.StringIO()
        stream = TeeStream(console, [rank_log, latest_log])
        self.assertEqual(stream.write('epoch 0\n'), 8)
        for output in (console, rank_log, latest_log):
            self.assertEqual(output.getvalue(), 'epoch 0\n')
        stream.write('epoch 1\n')
        stream.flush()
        for output in (console, rank_log, latest_log):
            self.assertEqual(output.getvalue(), 'epoch 0\nepoch 1\n')


if __name__ == '__main__':
    unittest.main()
