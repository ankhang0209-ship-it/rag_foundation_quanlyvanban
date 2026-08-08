"""
Unit tests cho Answer Pipeline, Gating, Grounding, Citations & Compare - Buổi 08.
Kiểm thử 100% offline sử dụng Mocks (Gemini API & Reranker).
"""

import unittest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import advanced_rag


class TestAnswerPipeline(unittest.TestCase):
    """
    Tập hợp 10 unit test kiểm thử Answer Pipeline & Compare.
    """

    def setUp(self):
        """
        Khởi tạo dữ liệu sample chunks cho testing.
        """
        self.sample_chunks = [
            {
                "chunk_id": "chunk_01",
                "strategy": "hierarchical",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung quy định cơ cấu lại thời hạn trả nợ cho khách hàng gặp khó khăn.",
                "metadata": {"section": "Điều 1"},
            },
            {
                "chunk_id": "chunk_02",
                "strategy": "hierarchical",
                "source": "TT_06_2023_NHNN.pdf",
                "page_start": 2,
                "page_end": 3,
                "text": "Nội dung quy định về hoạt động cho vay của tổ chức tín dụng.",
                "metadata": {"section": "Điều 2"},
            },
            {
                "chunk_id": "chunk_03",
                "strategy": "hierarchical",
                "source": "TT_39_2016_NHNN.pdf",
                "page_start": 5,
                "page_end": 5,
                "text": "Nội dung lãi suất cho vay và phí dịch vụ ngân hàng.",
                "metadata": {"section": "Điều 5"},
            },
        ]

    @patch("advanced_rag.search_semantic")
    @patch("advanced_rag.search_hybrid_rerank")
    def test_01_gating_per_mode(self, mock_rerank, mock_semantic):
        """
        Test 1: Gating theo đúng từng mode.
        - semantic: distance <= MAX_DISTANCE (0.45).
        - hybrid_rerank: rerank_score >= RERANK_MIN_SCORE (0.50).
        """
        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "t1", "semantic_rank": 1, "semantic_distance": 0.30},
            {"chunk_id": "c2", "source": "s2.pdf", "page_start": 2, "page_end": 2, "text": "t2", "semantic_rank": 2, "semantic_distance": 0.60},
        ]

        res_sem = advanced_rag.query_advanced_rag(
            question="cơ cấu nợ",
            mode="semantic",
            strategy="hierarchical",
            gen_fn=lambda p: "Trả lời [E1]",
        )
        self.assertTrue(res_sem["evidence"][0]["accepted"])
        self.assertFalse(res_sem["evidence"][1]["accepted"])

        mock_rerank.return_value = {
            "bm25_candidate_count": 2,
            "semantic_candidate_count": 2,
            "overlap_count": 1,
            "union_count": 3,
            "reranked_candidate_count": 2,
            "latency_ms": {"bm25_ms": 1.0, "semantic_ms": 1.0, "fusion_ms": 1.0, "rerank_ms": 2.0},
            "results": [
                {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "t1", "fused_rank": 1, "rerank_score": 0.85, "rerank_raw_score": 1.5, "rerank_rank": 1, "rank_change": 0},
                {"chunk_id": "c2", "source": "s2.pdf", "page_start": 2, "page_end": 2, "text": "t2", "fused_rank": 2, "rerank_score": 0.20, "rerank_raw_score": -1.3, "rerank_rank": 2, "rank_change": 0},
            ],
        }

        res_rr = advanced_rag.query_advanced_rag(
            question="cơ cấu nợ",
            mode="hybrid_rerank",
            strategy="hierarchical",
            gen_fn=lambda p: "Trả lời [E1]",
        )
        self.assertTrue(res_rr["evidence"][0]["accepted"])
        self.assertFalse(res_rr["evidence"][1]["accepted"])

    @patch("advanced_rag.search_semantic")
    def test_02_rejected_evidence_excluded(self, mock_semantic):
        """
        Test 2: Bằng chứng bị reject (accepted=False) không đi vào prompt context.
        """
        captured_prompts = []

        def mock_gen(prompt):
            captured_prompts.append(prompt)
            return "Trả lời [E1]"

        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Nội dung được chấp nhận", "semantic_rank": 1, "semantic_distance": 0.20},
            {"chunk_id": "c2", "source": "s2.pdf", "page_start": 2, "page_end": 2, "text": "Nội dung bị loại bỏ do distance cao", "semantic_rank": 2, "semantic_distance": 0.80},
        ]

        res = advanced_rag.query_advanced_rag(
            question="kiểm tra context",
            mode="semantic",
            gen_fn=mock_gen,
        )

        self.assertEqual(len(captured_prompts), 1)
        prompt_text = captured_prompts[0]
        self.assertIn("Nội dung được chấp nhận", prompt_text)
        self.assertNotIn("Nội dung bị loại bỏ do distance cao", prompt_text)

    @patch("advanced_rag.search_hybrid_rerank")
    def test_03_trace_counts_and_timings(self, mock_rerank):
        """
        Test 3: Trace counts và timings chứa đủ key bắt buộc theo hợp đồng.
        """
        mock_rerank.return_value = {
            "bm25_candidate_count": 10,
            "semantic_candidate_count": 10,
            "overlap_count": 4,
            "union_count": 16,
            "reranked_candidate_count": 10,
            "latency_ms": {"bm25_ms": 2.0, "semantic_ms": 3.0, "fusion_ms": 1.0, "rerank_ms": 5.0},
            "results": [
                {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "t1", "fused_rank": 1, "rerank_score": 0.90, "rerank_raw_score": 2.0, "rerank_rank": 1, "rank_change": 0},
            ],
        }

        res = advanced_rag.query_advanced_rag(
            question="cơ cấu nợ",
            mode="hybrid_rerank",
            gen_fn=lambda p: "Trả lời [E1]",
        )

        trace = res["trace"]
        self.assertEqual(trace["bm25_candidates"], 10)
        self.assertEqual(trace["semantic_candidates"], 10)
        self.assertEqual(trace["overlap"], 4)
        self.assertEqual(trace["union"], 16)
        self.assertEqual(trace["reranked"], 10)
        self.assertEqual(trace["accepted"], 1)
        self.assertTrue(trace["generation_called"])

        lat = trace["latency_ms"]
        for key in ["bm25", "semantic", "fusion", "rerank", "generation", "total"]:
            self.assertIn(key, lat)

    @patch("advanced_rag.search_semantic")
    def test_04_citation_mapping(self, mock_semantic):
        """
        Test 4: Nhãn trích dẫn [E1], [E2] được bóc tách và ánh xạ chính xác sang metadata thật.
        """
        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "TT_02.pdf", "page_start": 1, "page_end": 2, "text": "Text 1", "semantic_rank": 1, "semantic_distance": 0.20},
            {"chunk_id": "c2", "source": "TT_06.pdf", "page_start": 5, "page_end": 5, "text": "Text 2", "semantic_rank": 2, "semantic_distance": 0.25},
        ]

        gen_output = "Cơ cấu nợ được áp dụng theo quy định [E1]. Phí dịch vụ cho vay theo [E2]."

        res = advanced_rag.query_advanced_rag(
            question="quy định",
            mode="semantic",
            gen_fn=lambda p: gen_output,
        )

        self.assertEqual(res["status"], "answered")
        citations = res["citations"]
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["label"], "[E1]")
        self.assertEqual(citations[0]["source"], "TT_02.pdf")
        self.assertEqual(citations[0]["chunk_id"], "c1")
        self.assertEqual(citations[1]["label"], "[E2]")
        self.assertEqual(citations[1]["source"], "TT_06.pdf")

    @patch("advanced_rag.search_semantic")
    def test_05_invalid_citation_handling(self, mock_semantic):
        """
        Test 5: Nhãn trích dẫn giả (vd: [E99]) bị bóc khỏi answer và ghi nhận warning.
        """
        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Text 1", "semantic_rank": 1, "semantic_distance": 0.20},
        ]

        gen_output = "Thông tin hợp lệ [E1] nhưng thông tin giả bị bịa [E99]."

        res = advanced_rag.query_advanced_rag(
            question="test fake label",
            mode="semantic",
            gen_fn=lambda p: gen_output,
        )

        self.assertNotIn("[E99]", res["answer"])
        self.assertIn("[E1]", res["answer"])
        self.assertTrue(any("Loại bỏ nhãn trích dẫn không hợp lệ: [E99]" in w for w in res["warnings"]))

    @patch("advanced_rag.search_semantic")
    def test_06_generation_called_at_most_once(self, mock_semantic):
        """
        Test 6: Lệnh query chỉ gọi generation tối đa 1 lần.
        """
        mock_gen = MagicMock(return_value="Trả lời thành công [E1]")
        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Text 1", "semantic_rank": 1, "semantic_distance": 0.20},
        ]

        res = advanced_rag.query_advanced_rag(
            question="test call count",
            mode="semantic",
            gen_fn=mock_gen,
        )

        self.assertEqual(mock_gen.call_count, 1)

    @patch("advanced_rag.query_advanced_rag")
    def test_07_compare_no_generation(self, mock_query):
        """
        Test 7: Function compare_retrieval_modes gọi generation 0 lần.
        """
        mock_query.return_value = {
            "evidence": [],
            "trace": {"latency_ms": {"total": 1.0}},
        }

        res = advanced_rag.compare_retrieval_modes(question="test compare", strategy="hierarchical")

        self.assertEqual(mock_query.call_count, 4)
        for call_args in mock_query.call_args_list:
            self.assertFalse(call_args.kwargs.get("call_generation", True))

    @patch("advanced_rag.search_semantic")
    def test_08_insufficient_evidence_status(self, mock_semantic):
        """
        Test 8: Khi không có evidence nào đạt gating -> status='insufficient_evidence', generation không bị gọi.
        """
        mock_gen = MagicMock()
        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Text 1", "semantic_rank": 1, "semantic_distance": 0.90},
        ]

        res = advanced_rag.query_advanced_rag(
            question="out of scope",
            mode="semantic",
            gen_fn=mock_gen,
        )

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["answer"], "Không có đủ bằng chứng phù hợp trong tài liệu để trả lời câu hỏi.")
        mock_gen.assert_not_called()
        self.assertFalse(res["trace"]["generation_called"])

    @patch("advanced_rag.search_hybrid_rerank")
    def test_09_reranker_unavailable_status(self, mock_rerank):
        """
        Test 9: Khi Reranker bị lỗi khởi tạo/kết nối -> status='reranker_unavailable'.
        """
        mock_rerank.side_effect = RuntimeError("Không kết nối được model reranker")

        res = advanced_rag.query_advanced_rag(
            question="lỗi reranker",
            mode="hybrid_rerank",
        )

        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertEqual(res["answer"], "")
        self.assertTrue(any("reranker_unavailable" in w for w in res["warnings"]))

    @patch("advanced_rag.search_semantic")
    def test_10_schema_compliance_all_statuses(self, mock_semantic):
        """
        Test 10: Mọi status trả về đều tuân thủ đầy đủ JSON Schema chuẩn.
        """
        mock_semantic.return_value = [
            {"chunk_id": "c1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Text 1", "semantic_rank": 1, "semantic_distance": 0.20},
        ]

        res = advanced_rag.query_advanced_rag(
            question="schema test",
            mode="semantic",
            gen_fn=lambda p: "Trả lời [E1]",
        )

        required_keys = ["status", "mode", "question", "answer", "evidence", "citations", "warnings", "trace"]
        for key in required_keys:
            self.assertIn(key, res)

        ev = res["evidence"][0]
        ev_keys = [
            "source", "page_start", "page_end", "chunk_id", "text",
            "bm25_rank", "bm25_score", "semantic_rank", "semantic_distance",
            "rrf_score", "fused_rank", "rerank_raw_score", "rerank_score",
            "rerank_rank", "rank_change", "accepted"
        ]
        for key in ev_keys:
            self.assertIn(key, ev)


if __name__ == "__main__":
    unittest.main()
