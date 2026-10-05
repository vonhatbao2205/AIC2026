import unittest

from tools.benchmark_cv_report import matched, mcnemar, paired, ranking_metrics


class CVDiagnosticTests(unittest.TestCase):
    def test_cohort_rejects_missing_duplicate_or_quota_pairs(self):
        rows = [{"query_id": "q", "variant": v} for v in ("C", "C+V", "E")]
        self.assertEqual(set(matched(rows, ["q"])["q"]), {"C", "C+V", "E"})
        for invalid in (rows[:-1], rows + [rows[0]],
                        [{**rows[0], "quota_interrupted": True}, *rows[1:]]):
            with self.assertRaises(ValueError):
                matched(invalid, ["q"])

    def test_exact_mcnemar_uses_only_discordant_queries(self):
        self.assertEqual(mcnemar([1]*4+[0], [0]*4+[1]),
                         {"a_only": 4, "b_only": 1, "p_exact": .375})
        self.assertEqual(mcnemar([1, 0], [1, 0])["p_exact"], 1)

    def test_paired_resampling_keeps_the_same_query_on_both_sides(self):
        result = paired([1, 0], [1, 0], [[0, 0], [1, 1], [0, 1]])
        self.assertEqual(result, {"n": 2, "delta": 0, "ci95": [0, 0]})

    def test_same_run_ranking_still_requires_qa_answer(self):
        query = {"query_type": "QA", "targets": [{"video_id": "v", "start_s": 10,
                    "end_s": 10, "answers": ["red"]}]}
        candidate = {"video_id": "v", "pts_time": 10, "answer": "blue"}
        self.assertEqual(ranking_metrics([candidate], query, 1)["r1"], 0)
        self.assertEqual(ranking_metrics([{**candidate, "answer": "red"}], query, 1)["r1"], 1)


if __name__ == "__main__":
    unittest.main()
