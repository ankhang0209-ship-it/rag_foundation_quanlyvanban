"""
Unit tests cho Cross-Encoder Reranking stage - Buổi 08.
Kiểm thử 100% offline sử dụng Fake Reranker Function / Mocking (Không tải model từ Hugging Face).
"""

import math
import unittest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Import module từ rag_foundation/buoi_08/advanced_rag.py
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import advanced_rag


class TestCrossEncoderReranker(unittest.TestCase):
    """
    Tập hợp 10 unit test cho CrossEncoderReranker và search_hybrid_rerank.
    """

    def setUp(self):
        """
        Khởi tạo sample fused candidates cho testing.
        """
        self.sample_fused_candidates = [
            {
                "chunk_id": "chunk_01",
                "text": "Nội dung chunk 01 cơ cấu nợ",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "fused_rank": 1,
                "rrf_score": 0.032,
            },
            {
                "chunk_id": "chunk_02",
                "text": "Nội dung chunk 02 không liên quan",
                "source": "TT_06_2023_NHNN.pdf",
                "page_start": 2,
                "page_end": 2,
                "fused_rank": 2,
                "rrf_score": 0.025,
            },
            {
                "chunk_id": "chunk_03",
                "text": "Nội dung chunk 03 quy định lãi suất",
                "source": "TT_39_2016_NHNN.pdf",
                "page_start": 3,
                "page_end": 3,
                "fused_rank": 3,
                "rrf_score": 0.020,
            },
        ]

    @patch("advanced_rag.get_reranker_model")
    def test_01_lazy_loading(self, mock_get_model):
        """
        Test 1: Lazy loading - Không gọi get_reranker_model khi import, status hoặc khi dùng fake reranker fn.
        """
        # status read-only
        advanced_rag.get_status(strategy="hierarchical")
        mock_get_model.assert_not_called()

        # rerank với fake reranker fn
        fake_fn = lambda q, texts: [1.0] * len(texts)
        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_fn)
        reranker.rerank("câu hỏi", self.sample_fused_candidates, top_k=2)
        mock_get_model.assert_not_called()

    def test_02_one_pair_per_candidate(self):
        """
        Test 2: Một cặp (question, text) được truyền chính xác cho từng candidate.
        """
        captured_pairs = []

        def fake_rerank_fn(query, texts):
            captured_pairs.extend([(query, t) for t in texts])
            return [0.5] * len(texts)

        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_rerank_fn)
        reranker.rerank("cơ cấu nợ", self.sample_fused_candidates, top_k=3)

        self.assertEqual(len(captured_pairs), len(self.sample_fused_candidates))
        self.assertEqual(captured_pairs[0], ("cơ cấu nợ", self.sample_fused_candidates[0]["text"]))

    def test_03_batching_preserves_count(self):
        """
        Test 3: Batch processing trả đúng số lượng scores bằng với số lượng input candidates.
        """
        fake_fn = lambda q, texts: [float(i) for i in range(len(texts))]
        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_fn)
        results = reranker.rerank("test batch", self.sample_fused_candidates, top_k=10)
        self.assertEqual(len(results), 3)

    def test_04_sigmoid_score_correctness(self):
        """
        Test 4: Sigmoided score trong [0,1] chính xác số học: 1 / (1 + exp(-logit)).
        """
        fake_fn = lambda q, texts: [2.0, -1.0, 0.0]
        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_fn)
        results = reranker.rerank("cơ cấu nợ", self.sample_fused_candidates, top_k=3)

        item_pos2 = next(r for r in results if r["rerank_raw_score"] == 2.0)
        expected_sig = round(1.0 / (1.0 + math.exp(-2.0)), 6)
        self.assertEqual(item_pos2["rerank_score"], expected_sig)

    def test_05_sort_and_tie_break(self):
        """
        Test 5: Sắp xếp theo rerank_score giảm dần -> fused_rank tăng dần -> chunk_id.
        """
        candidates_tie = [
            {"chunk_id": "chunk_02", "text": "A", "fused_rank": 2},
            {"chunk_id": "chunk_01", "text": "B", "fused_rank": 1},
        ]
        # Cho 2 chunk có cùng logit 1.0
        fake_fn = lambda q, texts: [1.0, 1.0]
        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_fn)
        results = reranker.rerank("câu hỏi", candidates_tie, top_k=2)

        # Cùng rerank_score, chunk_01 có fused_rank 1 nhỏ hơn nên được xếp #1
        self.assertEqual(results[0]["chunk_id"], "chunk_01")
        self.assertEqual(results[1]["chunk_id"], "chunk_02")

    def test_06_rank_change_correctness(self):
        """
        Test 6: rank_change = fused_rank - rerank_rank được tính toán chính xác.
        """
        # Cho chunk_03 (fused_rank = 3) có logit cao nhất (5.0) -> vươn lên rerank_rank #1
        fake_fn = lambda q, texts: [1.0, 0.0, 5.0]
        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_fn)
        results = reranker.rerank("cơ cấu nợ", self.sample_fused_candidates, top_k=3)

        top_1 = results[0]
        self.assertEqual(top_1["chunk_id"], "chunk_03")
        self.assertEqual(top_1["rerank_rank"], 1)
        # fused_rank (3) - rerank_rank (1) = +2
        self.assertEqual(top_1["rank_change"], 2)

    @patch("advanced_rag.search_hybrid")
    def test_07_rerank_limited_candidates(self, mock_hybrid):
        """
        Test 7: Chỉ rerank tối đa min(RERANK_CANDIDATES, union_count) candidates đầu tiên.
        """
        fused_10 = [{"chunk_id": f"chunk_{i:02d}", "text": f"text {i}", "fused_rank": i} for i in range(1, 11)]
        mock_hybrid.return_value = {
            "results": fused_10,
            "bm25_candidate_count": 10,
            "semantic_candidate_count": 10,
            "union_count": 10,
            "overlap_count": 0,
            "fused_count": 10,
            "latency_ms": {"bm25_ms": 1.0, "semantic_ms": 1.0, "fusion_ms": 1.0, "total_ms": 3.0},
        }

        captured_rerank_count = []

        def fake_rerank_fn(query, texts):
            captured_rerank_count.append(len(texts))
            return [1.0] * len(texts)

        # Cấu hình RERANK_CANDIDATES = 4
        override_cfg = {"rerank_candidates": 4, "final_top_k": 2}
        with patch("advanced_rag.load_config", return_value={**advanced_rag.load_config(), **override_cfg}):
            advanced_rag.search_hybrid_rerank("test limit", top_k=2, strategy="hierarchical", reranker_fn=fake_rerank_fn)

        # Chỉ 4 candidates đầu tiên được đưa vào Reranker
        self.assertEqual(captured_rerank_count[0], 4)

    @patch("advanced_rag.search_hybrid")
    def test_08_returns_only_final_top_k(self, mock_hybrid):
        """
        Test 8: Kết quả trả về sau Reranking chỉ cắt lấy đúng FINAL_TOP_K.
        """
        mock_hybrid.return_value = {
            "results": self.sample_fused_candidates,
            "bm25_candidate_count": 3,
            "semantic_candidate_count": 3,
            "union_count": 3,
            "overlap_count": 0,
            "fused_count": 3,
            "latency_ms": {"bm25_ms": 1.0, "semantic_ms": 1.0, "fusion_ms": 1.0, "total_ms": 3.0},
        }
        fake_fn = lambda q, texts: [1.0] * len(texts)
        trace = advanced_rag.search_hybrid_rerank("test topk", top_k=2, strategy="hierarchical", reranker_fn=fake_fn)

        self.assertEqual(len(trace["results"]), 2)
        self.assertEqual(trace["final_count"], 2)

    @patch("advanced_rag.get_reranker_model")
    def test_09_model_download_failure_not_silent(self, mock_get_model):
        """
        Test 9: Lỗi khi tải hoặc khởi tạo model không được silent fallback âm thầm.
        """
        mock_get_model.side_effect = RuntimeError("reranker_unavailable: Lỗi kết nối mạng")

        reranker = advanced_rag.CrossEncoderReranker()
        with self.assertRaises(RuntimeError) as ctx:
            reranker.rerank("câu hỏi", self.sample_fused_candidates, top_k=2)
        self.assertIn("reranker_unavailable", str(ctx.exception))

    def test_10_offline_testing_without_network_or_download(self):
        """
        Test 10: Kiểm thử hoàn toàn offline sử dụng fake reranker callable, không có network call hay download model.
        """
        fake_fn = lambda q, texts: [float(len(t)) for t in texts]
        reranker = advanced_rag.CrossEncoderReranker(reranker_fn=fake_fn)
        results = reranker.rerank("offline test", self.sample_fused_candidates, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("rerank_rank", results[0])
        self.assertIn("rank_change", results[0])


if __name__ == "__main__":
    unittest.main()
