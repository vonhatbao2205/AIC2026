import unittest

from tools.benchmark_video_report import grouped_rank, paired, group_solve, build_levels
from benchmarks.agent.metrics import first_rank, video_rank


def frame(video, time, **fields):
    return {"video_id": video, "time": time, **fields}


class VideoGroupingTests(unittest.TestCase):
    def setUp(self):
        self.query = {"query_type": "T-KIS", "targets": [{"video_id": "right", "start_s": 10, "end_s": 10}]}

    def test_correct_frame_later_in_leading_video_counts_at_one(self):
        ranking = [frame("right", 2), frame("wrong", 10), frame("right", 10)]
        self.assertEqual(first_rank(ranking, self.query, 1), 3)
        self.assertEqual(grouped_rank(ranking, self.query, 1), 1)

    def test_duplicates_do_not_take_video_slots(self):
        ranking = [frame("wrong", 10)] * 6 + [frame("right", 10)]
        self.assertEqual(grouped_rank(ranking, self.query, 1), 2)

    def test_video_identity_alone_does_not_meet_group_moment(self):
        ranking = [frame("right", 2)]
        self.assertEqual(video_rank(ranking, self.query), 1)
        self.assertIsNone(grouped_rank(ranking, self.query, 1))
        self.assertIsNone(grouped_rank([], self.query, 1))

    def test_qa_answer_is_required_only_for_strict_grouping(self):
        query = {"query_type": "QA", "targets": [{**self.query["targets"][0], "answers": ["accepted"]}]}
        ranking = [frame("right", 10, answer="incorrect")]
        self.assertEqual(grouped_rank(ranking, query, 1), 1)
        self.assertIsNone(grouped_rank(ranking, query, 1, strict=True))
        ranking.append(frame("right", 10, answer="accepted"))
        self.assertEqual(grouped_rank(ranking, query, 1, strict=True), 1)

    def test_trake_requires_ordered_complete_sequence(self):
        targets = [{"video_id": "right", "start_s": t, "end_s": t} for t in [10, 20]]
        query = {"query_type": "TRAKE", "targets": targets, "sequences": [targets]}
        ranking = [frame("right", 10, event=1)]
        self.assertIsNone(grouped_rank(ranking, query, 1))
        ranking.append(frame("right", 20, event=2))
        self.assertEqual(grouped_rank(ranking, query, 1), 1)
        ranking[1]["time"] = 5
        self.assertIsNone(grouped_rank(ranking, query, 1))

    def test_paired_cohort_does_not_pool_partial_variants(self):
        def record(qid, variant):
            return {"query_id": qid, "variant": variant, "agent_failed": False,
                    **{f"{k}_rank": 1 for k in ['strict', 'video', 'group_moment', 'group_strict']}}
        rows = [record("complete", "A"), record("complete", "D"), record("partial", "A")]
        result = paired(rows, ["A", "D"])
        self.assertEqual(result["query_ids"], ["complete"])
        self.assertEqual(result["summary"]["A"]["queries"], 1)

    def test_cumulative_solve_uses_group_coverage_and_excludes_incomplete_query(self):
        rows = []
        for level in [1, 2]:
            rows.append({'query_id': 'complete', 'variant': 'A', 'query_type': 'T-KIS',
                         'hint_stage': 'h1' if level == 1 else 'full', 'hint_level': level, 'hint_count': 2,
                         'strict_rank': 8, 'group_moment_rank': 1 if level == 1 else None,
                         'group_strict_rank': 1 if level == 1 else None, 'video_rank': 1, 'agent_failed': False})
        rows.append({**rows[0], 'query_id': 'incomplete'})
        result = group_solve(rows, ['A'], {'complete': 2, 'incomplete': 2})
        self.assertEqual(result['query_ids'], ['complete'])
        moment = result['summary']['A']['moment']
        self.assertEqual(moment['early_solve_rate'], 1)
        self.assertEqual(moment['regression_rate'], 1)
        levels = build_levels(rows, ['A'])
        self.assertEqual(levels['h1']['paired']['summary']['A']['queries'], 2)
        self.assertEqual(levels['full']['paired']['summary']['A']['queries'], 1)


if __name__ == "__main__":
    unittest.main()
