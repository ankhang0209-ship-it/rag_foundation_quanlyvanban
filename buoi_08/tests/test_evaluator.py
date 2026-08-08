"""
Unit tests cho Evaluator Metrics (Recall@K, MRR@K, nDCG@K) và Report Generation - Buổi 08.
Kiểm thử 100% offline với ví dụ tính tay nhỏ.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import evaluate


class TestEvaluatorMetrics(unittest.TestCase):
    """
    Tập hợp 5 unit test kiểm thử toán học metrics và report generator cho evaluate.py.
    """

    def test_01_calculate_recall(self):
        """
        Test 1: Recall@K với ví dụ tính tay.
        relevant = ['c2', 'c4'] (2 chunks).
        retrieved = ['c1', 'c2', 'c3'] (top 3 chứa 1 relevant 'c2').
        Recall@3 = 1 / 2 = 0.5.
        """
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c2", "c4"]
        recall = evaluate.calculate_recall(retrieved, relevant, k=3)
        self.assertEqual(recall, 0.5)

        # Trọng số k = 1: top 1 ['c1'] không chứa 'c2' hay 'c4' -> Recall@1 = 0.0
        recall_k1 = evaluate.calculate_recall(retrieved, relevant, k=1)
        self.assertEqual(recall_k1, 0.0)

    def test_02_calculate_mrr(self):
        """
        Test 2: MRR@K với ví dụ thứ hạng.
        relevant = ['c1'].
        retrieved = ['c3', 'c1', 'c2'].
        'c1' xuất hiện ở vị trí rank #2 (1-indexed) -> MRR = 1/2 = 0.5.
        """
        retrieved = ["c3", "c1", "c2"]
        relevant = ["c1"]
        mrr = evaluate.calculate_mrr(retrieved, relevant, k=3)
        self.assertEqual(mrr, 0.5)

        # Khi k = 1 -> 'c1' không nằm trong top 1 -> MRR = 0.0
        mrr_k1 = evaluate.calculate_mrr(retrieved, relevant, k=1)
        self.assertEqual(mrr_k1, 0.0)

    def test_03_calculate_ndcg(self):
        """
        Test 3: nDCG@K chuẩn xác theo Discounted Cumulative Gain.
        relevant = ['c1'].
        retrieved = ['c1', 'c2', 'c3'].
        'c1' nằm vị trí 1: DCG = 1/log2(2) = 1.0. IDCG = 1.0 -> nDCG = 1.0.
        """
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c1"]
        ndcg = evaluate.calculate_ndcg(retrieved, relevant, k=3)
        self.assertEqual(ndcg, 1.0)

        # Trường hợp match ở rank #2:
        # retrieved = ['c2', 'c1', 'c3'], relevant = ['c1']
        # DCG = 1/log2(3) = 0.6309. IDCG = 1.0 -> nDCG = 0.6309.
        ndcg_r2 = evaluate.calculate_ndcg(["c2", "c1", "c3"], ["c1"], k=3)
        self.assertAlmostEqual(ndcg_r2, 1.0 / math_log2(3), places=4)

    @patch("evaluate.advanced_rag.query_advanced_rag")
    def test_04_evaluation_report_schema(self, mock_query):
        """
        Test 4: Báo cáo JSON xuất ra chứa đúng cấu trúc schema và ghi nhận warning needs_human_review.
        """
        mock_query.return_value = {
            "status": "retrieval_only",
            "evidence": [{"chunk_id": "c1"}],
            "trace": {"latency_ms": {"total": 5.0}},
        }

        # Tạo file questions tạm thời có needs_human_review=True
        eval_sample = [
            {
                "query_id": "Q01",
                "question": "Câu hỏi test 1",
                "relevant_chunk_ids": ["c1"],
                "scope": "in_scope",
                "needs_human_review": True,
            }
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f_q:
            json.dump(eval_sample, f_q)
            eval_file_path = Path(f_q.name)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f_out:
            out_file_path = Path(f_out.name)

        try:
            report = evaluate.run_evaluation(
                eval_file=eval_file_path,
                output_report=out_file_path,
                k=2,
                modes=["bm25", "semantic"],
            )

            self.assertTrue(report["needs_human_review_warning"])
            self.assertIsNotNone(report["warning_message"])
            self.assertIn("metrics_by_mode", report)
            self.assertIn("bm25", report["metrics_by_mode"])
            self.assertIn("semantic", report["metrics_by_mode"])

        finally:
            eval_file_path.unlink(missing_ok=True)
            out_file_path.unlink(missing_ok=True)

    @patch("evaluate.advanced_rag.query_advanced_rag")
    def test_05_no_generation_called(self, mock_query):
        """
        Test 5: run_evaluation truyền call_generation=False đảm bảo 0 calls generation.
        """
        mock_query.return_value = {
            "status": "retrieval_only",
            "evidence": [],
            "trace": {"latency_ms": {"total": 1.0}},
        }

        eval_sample = [
            {"query_id": "Q01", "question": "test no gen", "relevant_chunk_ids": ["c1"]}
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f_q:
            json.dump(eval_sample, f_q)
            eval_file_path = Path(f_q.name)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f_out:
            out_file_path = Path(f_out.name)

        try:
            evaluate.run_evaluation(eval_file=eval_file_path, output_report=out_file_path, k=1, modes=["bm25"])
            self.assertEqual(mock_query.call_count, 1)
            call_kwargs = mock_query.call_args.kwargs
            self.assertFalse(call_kwargs.get("call_generation", True))
        finally:
            eval_file_path.unlink(missing_ok=True)
            out_file_path.unlink(missing_ok=True)


def math_log2(x: float) -> float:
    import math
    return math.log2(x)


if __name__ == "__main__":
    unittest.main()
