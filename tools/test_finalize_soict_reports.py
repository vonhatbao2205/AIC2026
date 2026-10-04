import unittest

from tools.finalize_soict_reports import original_split


class SplitLineageTests(unittest.TestCase):
    def test_fresh_retry_executed_as_test_returns_to_original_dev(self):
        queries={'q':{'split':'dev'}}
        self.assertEqual(original_split({'query_id':'q','split':'test'},queries),'dev')

    def test_mislabelled_original_split_is_rejected(self):
        queries={'q':{'split':'test'}}
        with self.assertRaisesRegex(ValueError,'provenance conflicts'):
            original_split({'query_id':'q','split':'test','original_split':'dev'},queries)


if __name__=='__main__':
    unittest.main()
