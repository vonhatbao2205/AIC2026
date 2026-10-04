import unittest

from tools.filter_benchmark_failures import exclusion_reasons


class FailureFilterTests(unittest.TestCase):
    def test_wrong_answer_is_not_an_execution_error(self):
        self.assertEqual(exclusion_reasons({'metrics': {'r1': 0, 'r5': 0, 'agent_failed': False, 'failed': False, 'fallback': False}}), [])

    def test_fallback_is_an_execution_error_even_when_agents_completed(self):
        self.assertEqual(exclusion_reasons({'metrics': {'agent_failed': False, 'failed': False, 'fallback': True}}), ['fallback'])

    def test_archived_legacy_quota_is_an_error(self):
        self.assertEqual(exclusion_reasons({'metrics': {}, 'legacy_quota_interrupted': True}), ['quota_interrupted'])


if __name__ == '__main__':
    unittest.main()
