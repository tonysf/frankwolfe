"""Check duplicate-launch prevention and failure preservation without workers."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pot_step_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignSafetyTests(unittest.TestCase):
    def make_out(self, root):
        out = Path(root)
        (out / 'runs').mkdir()
        (out / 'logs').mkdir()
        campaign.write(out / 'config.json', dict(runs=[dict(id='full_gradient')]))
        return out

    def test_existing_start_receipt_blocks_second_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = self.make_out(temporary)
            campaign.write(out / 'started.json', {'pid': 123})
            original = (out / 'started.json').read_bytes()
            with patch.object(campaign, 'validate'), patch.object(campaign, 'execute_run') as worker:
                with self.assertRaises(FileExistsError):
                    campaign.campaign(out)
                worker.assert_not_called()
            self.assertEqual(original, (out / 'started.json').read_bytes())

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = self.make_out(temporary)
            previous = out / 'runs' / 'full_gradient'
            previous.mkdir()
            (previous / 'progress.json').write_text('{"status":"running"}')
            with patch.object(campaign, 'validate'), patch.object(campaign, 'execute_run') as worker:
                with self.assertRaisesRegex(ValueError, 'Existing runs'):
                    campaign.campaign(out)
                worker.assert_not_called()
            self.assertEqual(json.loads((previous / 'progress.json').read_text())['status'], 'running')

    def test_report_failure_does_not_mark_campaign_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = self.make_out(temporary)
            result = dict(status='completed', id='full_gradient')
            with patch.object(campaign, 'validate'), patch.object(campaign, 'execute_run', return_value=result), \
                    patch.object(campaign.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
                with self.assertRaisesRegex(ValueError, 'final validation failed'):
                    campaign.campaign(out)
            state = campaign.read(out / 'campaign.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['runs'], [result])
            self.assertFalse((out / 'completed.json').exists())
            self.assertTrue((out / 'started.json').exists())


if __name__ == '__main__':
    unittest.main()
