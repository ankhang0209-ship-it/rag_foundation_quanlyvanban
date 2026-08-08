"""
Unit tests cho Reciprocal Rank Fusion (RRF) và Hybrid Search Pipeline - Buổi 08.
Kiểm thử offline 100%, không load reranker model và không gọi LLM generation.
"""

import unittest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Import module từ rag_foundation/buoi_08/advanced_rag.py
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import advanced_rag


class TestRRFFusionAndHybrid(unittest.TestCase):
    """
    Tập hợp 10 unit test cho RRF Fusion và Hybrid Search Engine.
    """

    def setUp(self):
        """
        Khởi tạo danh sách sample BM25 và Semantic candidates.
        """
        self.bm25_candidates = [
            {
                "chunk_id": "chunk_01",
                "text": "Nội dung chunk 01",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "bm25_rank": 1,
                "bm25_score": 5.0,
            },
            {
                "chunk_id": "chunk_02",
                "text": "Nội dung chunk 02",
                "source": "TT_06_2023_NHNN.pdf",
                "page_start": 2,
                "page_end": 2,
                "bm25_rank": 2,
                "bm25_score": 3.5,
            },
        ]

        self.semantic_candidates = [
            {
                "chunk_id": "chunk_02",
                "text": "Nội dung chunk 02",
                "source": "TT_06_2023_NHNN.pdf",
                "page_start": 2,
                "page_end": 2,
                "semantic_rank": 1,
                "semantic_distance": 0.15,
            },
            {
                "chunk_id": "chunk_03",
                "text": "Nội dung chunk 03",
                "source": "TT_39_2016_NHNN.pdf",
                "page_start": 3,
                "page_end": 3,
                "semantic_rank": 2,
                "semantic_distance": 0.25,
            },
        ]

    def test_01_rrf_formula_arithmetic_correctness(self):
        """
        Test 1: Công thức RRF chính xác từng con số: 1/(60+1) + 1/(60+2) = 1/61 + 1/62.
        """
        # chunk_02 xuất hiện ở BM25 rank 2 và Semantic rank 1 (với k=60, w_bm25=1.0, w_sem=1.0)
        results = advanced_rag.rrf_fusion(
            self.bm25_candidates, self.semantic_candidates, k=60, bm25_weight=1.0, semantic_weight=1.0
        )
        item_c02 = next(r for r in results if r["chunk_id"] == "chunk_02")

        expected_score = round(1.0 / (60 + 2) + 1.0 / (60 + 1), 6)  # 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        self.assertEqual(item_c02["rrf_score"], expected_score)

    def test_02_candidate_overlap_no_duplicates(self):
        """
        Test 2: Candidate trùng lặp giữa 2 nhánh (overlap) hợp nhất thành 1 record duy nhất.
        """
        results = advanced_rag.rrf_fusion(self.bm25_candidates, self.semantic_candidates, k=60)
        chunk_ids = [r["chunk_id"] for r in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(len(results), 3)  # chunk_01, chunk_02, chunk_03

    def test_03_bm25_only_candidate_retained(self):
        """
        Test 3: Candidate chỉ có trong BM25 vẫn được giữ lại với semantic_rank = None.
        """
        results = advanced_rag.rrf_fusion(self.bm25_candidates, self.semantic_candidates, k=60)
        item_c01 = next(r for r in results if r["chunk_id"] == "chunk_01")
        self.assertIsNotNone(item_c01["bm25_rank"])
        self.assertIsNone(item_c01["semantic_rank"])
        self.assertEqual(item_c01["matched_by"], ["bm25"])

    def test_04_semantic_only_candidate_retained(self):
        """
        Test 4: Candidate chỉ có trong Semantic vẫn được giữ lại với bm25_rank = None.
        """
        results = advanced_rag.rrf_fusion(self.bm25_candidates, self.semantic_candidates, k=60)
        item_c03 = next(r for r in results if r["chunk_id"] == "chunk_03")
        self.assertIsNone(item_c03["bm25_rank"])
        self.assertIsNotNone(item_c03["semantic_rank"])
        self.assertEqual(item_c03["matched_by"], ["semantic"])

    def test_05_weight_zero_disables_branch(self):
        """
        Test 5: Trọng số weight = 0 sẽ triệt tiêu hoàn toàn đóng góp của nhánh tương ứng.
        """
        results = advanced_rag.rrf_fusion(
            self.bm25_candidates, self.semantic_candidates, k=60, bm25_weight=1.0, semantic_weight=0.0
        )
        item_c03 = next(r for r in results if r["chunk_id"] == "chunk_03")
        # semantic_weight = 0 làm điểm rrf_score của chunk_03 bằng 0
        self.assertEqual(item_c03["rrf_score"], 0.0)

    def test_06_tie_break_deterministic(self):
        """
        Test 6: Sắp xếp tie-break ổn định khi hai candidates cùng điểm RRF score.
        """
        bm25_tie = [
            {"chunk_id": "chunk_B", "text": "Same", "source": "s.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 4.0},
            {"chunk_id": "chunk_A", "text": "Same", "source": "s.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 4.0},
        ]
        results = advanced_rag.rrf_fusion(bm25_tie, [], k=60)
        self.assertEqual(results[0]["chunk_id"], "chunk_A")
        self.assertEqual(results[1]["chunk_id"], "chunk_B")

    def test_07_metadata_mismatch_fails(self):
        """
        Test 7: Báo lỗi ValueError nếu cùng chunk_id nhưng metadata (text/source) giữa 2 nhánh bị mâu thuẫn.
        """
        mismatched_semantic = [
            {
                "chunk_id": "chunk_01",
                "text": "TEXT KHÁC NHAU CỐ TÌNH SAU",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "semantic_rank": 1,
                "semantic_distance": 0.1,
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            advanced_rag.rrf_fusion(self.bm25_candidates, mismatched_semantic, k=60)
        self.assertIn("Metadata mismatch", str(ctx.exception))

    @patch("advanced_rag.search_semantic")
    @patch("advanced_rag.search_bm25")
    @patch("rag.load_chunks")
    def test_08_trace_counts_correctness(self, mock_load, mock_bm25, mock_sem):
        """
        Test 8: Trace object trả về đúng các con số đếm ứng viên (bm25_count, semantic_count, union, overlap).
        """
        mock_load.return_value = (self.bm25_candidates, {})
        mock_bm25.return_value = self.bm25_candidates
        mock_sem.return_value = self.semantic_candidates

        trace = advanced_rag.search_hybrid(question="thử trace", top_k=2, strategy="hierarchical")

        self.assertEqual(trace["bm25_candidate_count"], 2)
        self.assertEqual(trace["semantic_candidate_count"], 2)
        self.assertEqual(trace["union_count"], 3)
        self.assertEqual(trace["overlap_count"], 1)  # chunk_02 xuất hiện cả 2 bên
        self.assertIn("latency_ms", trace)

    @patch("advanced_rag.search_semantic")
    @patch("advanced_rag.search_bm25")
    @patch("rag.load_chunks")
    def test_09_hybrid_calls_each_retriever_once(self, mock_load, mock_bm25, mock_sem):
        """
        Test 9: Hybrid Search gọi mỗi bộ retriever (BM25 và Semantic) đúng 1 lần.
        """
        mock_load.return_value = (self.bm25_candidates, {})
        mock_bm25.return_value = self.bm25_candidates
        mock_sem.return_value = self.semantic_candidates

        advanced_rag.search_hybrid(question="thử call count", top_k=2, strategy="hierarchical")

        mock_bm25.assert_called_once()
        mock_sem.assert_called_once()

    @patch("advanced_rag.CrossEncoderReranker")
    @patch("google.genai.Client")
    def test_10_no_reranker_or_generation_loaded(self, mock_genai, mock_reranker):
        """
        Test 10: Giai đoạn Hybrid RRF không tải CrossEncoderReranker và không gọi LLM generation.
        """
        advanced_rag.rrf_fusion(self.bm25_candidates, self.semantic_candidates, k=60)
        mock_reranker.assert_not_called()
        mock_genai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
