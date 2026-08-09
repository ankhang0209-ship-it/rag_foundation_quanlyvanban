"""
Unit tests cho Evaluator Module (Buổi 09).
Thuần Python 100% offline, 0 network, 0 Gemini API calls.
"""

import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path

BUOI_09_DIR = Path(__file__).resolve().parent.parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import evaluate


class TestEvaluatorModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            "multi_query_count": 2,
            "multi_query_max_chars": 300,
            "multi_query_temperature": 0.2,
            "multi_query_original_weight": 1.5,
            "multi_query_variant_weight": 1.0,
            "multi_query_rrf_k": 60,
            "per_query_candidates": 5,
            "parent_max_chars": 6000,
            "parent_score_child_limit": 3,
            "parent_rrf_k": 60,
            "parent_candidates": 5,
            "final_parent_top_k": 2,
            "total_context_max_chars": 16000,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 768,
            "generation_model": "gemini-3.5-flash-lite",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_min_score": 0.50,
            "rerank_device": "auto",
            "api_key": "",
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_recall_at_k(self):
        """Test 1: Công thức Recall@K."""
        gold = ["p1", "p2"]
        retrieved = ["p1", "p3", "p4"]
        self.assertEqual(evaluate.calculate_recall_at_k(retrieved, gold), 0.5)

    def test_02_mrr_at_k(self):
        """Test 2: Công thức MRR@K."""
        gold = ["p2"]
        retrieved = ["p1", "p2", "p3"]
        self.assertEqual(evaluate.calculate_mrr_at_k(retrieved, gold), 0.5)  # Rank 2 -> 1/2 = 0.5

    def test_03_ndcg_at_k(self):
        """Test 3: Công thức nDCG@K."""
        gold = ["p1"]
        retrieved = ["p1", "p2"]
        self.assertEqual(evaluate.calculate_ndcg_at_k(retrieved, gold), 1.0)  # Rank 1 hit -> 1.0

    def test_04_offline_evaluate_retrieval_modes(self):
        """Test 4: Chạy evaluate_retrieval_modes offline tạo file báo cáo hợp lệ."""
        rep_dir = self.temp_dir / "reports"
        q_file = self.temp_dir / "test_questions.json"

        test_questions = [
            {
                "question_id": "Q01",
                "question": "Test question",
                "question_type": "exact",
                "relevant_child_ids": ["hierarchical_TT_39_2016_NHNN_008"],
                "relevant_parent_ids": ["parent_TT_39_2016_NHNN_Điều_7._Điều_kiện_vay_vốn_w1"],
                "needs_human_review": True,
                "notes": "Test note",
            }
        ]

        with open(q_file, "w", encoding="utf-8") as f:
            json.dump(test_questions, f, ensure_ascii=False)

        report = evaluate.evaluate_retrieval_modes(
            config=self.config,
            questions_file=q_file,
            reports_dir=rep_dir,
            query_generator_fn=lambda q, c: [{"text": "V1", "focus": "paraphrase"}],
            hybrid_retriever_fn=lambda q, k: [{"child_id": "hierarchical_TT_39_2016_NHNN_008", "text": "T1", "source": "doc.pdf", "page_start": 1, "page_end": 1}],
            reranker_fn=lambda pairs: [2.0],
        )

        self.assertIn("summary_per_mode", report)
        self.assertTrue((rep_dir / "evaluation_report.json").exists())
        self.assertTrue((rep_dir / "latest_report.json").exists())


if __name__ == "__main__":
    unittest.main()
