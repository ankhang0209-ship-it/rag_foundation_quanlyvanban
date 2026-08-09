"""
Unit tests cho Cross-Query RRF Fusion & Per-Query Hybrid Child Search (Buổi 09).
Bao gồm 12 test cases độc lập 100% offline sử dụng dependency injection (fake query generator & fake hybrid retriever).
"""

import unittest
import sys
from pathlib import Path

BUOI_09_DIR = Path(__file__).resolve().parent.parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag


class TestCrossQueryRRFFusion(unittest.TestCase):
    def setUp(self):
        self.config = {
            "multi_query_count": 3,
            "multi_query_max_chars": 300,
            "multi_query_temperature": 0.2,
            "multi_query_original_weight": 1.5,
            "multi_query_variant_weight": 1.0,
            "multi_query_rrf_k": 60,
            "per_query_candidates": 12,
            "parent_max_chars": 6000,
            "parent_score_child_limit": 3,
            "parent_rrf_k": 60,
            "parent_candidates": 10,
            "final_parent_top_k": 3,
            "total_context_max_chars": 16000,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 768,
            "generation_model": "gemini-3.5-flash-lite",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_min_score": 0.50,
            "rerank_device": "auto",
            "api_key": "",
        }
        hierarchical_rag._MULTI_QUERY_CACHE.clear()

    def test_01_cross_query_rrf_arithmetic_correctness(self):
        """Test 1: Kiểm tra tính toán số học chính xác của công thức Cross-Query RRF."""
        # Child 1 xuất hiện ở Q0 (rank 1, weight 1.5) và Q1 (rank 3, weight 1.0)
        # MQ-RRF Score = 1.5 / (60 + 1) + 1.0 / (60 + 3) = 1.5 / 61 + 1.0 / 63 = 0.02459016 + 0.01587301 = 0.040463
        query_set = [
            {"query_id": "Q0", "text": "Q0 text", "origin": "original"},
            {"query_id": "Q1", "text": "Q1 text", "origin": "generated"},
        ]
        per_query_results = {
            "Q0": {
                "status": "success",
                "results": [
                    {"child_id": "c1", "text": "Text 1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}
                ]
            },
            "Q1": {
                "status": "success",
                "results": [
                    {"child_id": "c2", "text": "Text 2", "source": "s1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                    {"child_id": "c3", "text": "Text 3", "source": "s1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 2},
                    {"child_id": "c1", "text": "Text 1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 3},
                ]
            }
        }
        merged, _ = hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)
        c1 = [item for item in merged if item["child_id"] == "c1"][0]
        expected_score = round(1.5 / (60 + 1) + 1.0 / (60 + 3), 6)
        self.assertAlmostEqual(c1["multi_query_rrf_score"], expected_score, places=5)

    def test_02_original_vs_variant_weights(self):
        """Test 2: Trọng số Q0 (1.5) lớn hơn trọng số Variant (1.0) khi cùng rank."""
        query_set = [
            {"query_id": "Q0", "text": "Q0", "origin": "original"},
            {"query_id": "Q1", "text": "Q1", "origin": "generated"},
        ]
        per_query_results = {
            "Q0": {"status": "success", "results": [{"child_id": "c_orig", "text": "Text", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
            "Q1": {"status": "success", "results": [{"child_id": "c_var", "text": "Text Var", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
        }
        merged, _ = hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)
        self.assertEqual(merged[0]["child_id"], "c_orig")  # c_orig xếp trên do có weight 1.5 > 1.0

    def test_03_deduplicate_union(self):
        """Test 3: Hợp nhất danh sách child hits theo child_id không trùng lặp."""
        query_set = [{"query_id": "Q0", "origin": "original"}, {"query_id": "Q1", "origin": "generated"}]
        per_query_results = {
            "Q0": {"status": "success", "results": [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
            "Q1": {"status": "success", "results": [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
        }
        merged, _ = hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)
        self.assertEqual(len(merged), 1)

    def test_04_missing_query_contribution(self):
        """Test 4: Candidate chỉ xuất hiện ở 1 query vẫn được giữ lại với đầy đủ đóng góp."""
        query_set = [{"query_id": "Q0", "origin": "original"}, {"query_id": "Q1", "origin": "generated"}]
        per_query_results = {
            "Q0": {"status": "success", "results": [{"child_id": "c0_only", "text": "T0", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
            "Q1": {"status": "success", "results": []},
        }
        merged, _ = hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["support_query_count"], 1)
        self.assertEqual(merged[0]["support_query_ids"], ["Q0"])

    def test_05_support_query_count_and_ids(self):
        """Test 5: Đếm chính xác support_query_count và danh sách support_query_ids theo thứ tự."""
        query_set = [
            {"query_id": "Q0", "origin": "original"},
            {"query_id": "Q1", "origin": "generated"},
            {"query_id": "Q2", "origin": "generated"},
        ]
        per_query_results = {
            "Q0": {"status": "success", "results": [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
            "Q1": {"status": "success", "results": []},
            "Q2": {"status": "success", "results": [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
        }
        merged, _ = hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)
        self.assertEqual(merged[0]["support_query_count"], 2)
        self.assertEqual(merged[0]["support_query_ids"], ["Q0", "Q2"])

    def test_06_metadata_mismatch_fails(self):
        """Test 6: Xung đột metadata cùng child_id giữa các query ném lỗi ValueError."""
        query_set = [{"query_id": "Q0", "origin": "original"}, {"query_id": "Q1", "origin": "generated"}]
        per_query_results = {
            "Q0": {"status": "success", "results": [{"child_id": "c1", "text": "Text A", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
            "Q1": {"status": "success", "results": [{"child_id": "c1", "text": "Text B", "source": "s.pdf", "page_start": 1, "page_end": 1}]},
        }
        with self.assertRaises(ValueError):
            hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)

    def test_07_deterministic_tie_break(self):
        """Test 7: Tie-break sắp xếp ổn định khi hai child cùng điểm RRF score."""
        query_set = [{"query_id": "Q1", "origin": "generated"}]
        per_query_results = {
            "Q1": {
                "status": "success",
                "results": [
                    {"child_id": "c_b", "text": "Text B", "source": "s.pdf", "page_start": 1, "page_end": 1},
                    {"child_id": "c_a", "text": "Text A", "source": "s.pdf", "page_start": 1, "page_end": 1},
                ]
            }
        }
        # Sắp xếp tie-break theo child_id tăng dần
        merged, _ = hierarchical_rag.cross_query_rrf_fusion(per_query_results, query_set, config=self.config)
        self.assertEqual(merged[0]["child_id"], "c_b")  # rank 1 > rank 2

    def test_08_each_query_calls_hybrid_once(self):
        """Test 8: Mỗi query gọi retriever đúng 1 lần duy nhất."""
        calls = []

        def fake_gen(q, count):
            return [{"text": "Variant 1", "focus": "paraphrase"}]

        def fake_hybrid(q_text, candidate_k):
            calls.append(q_text)
            return [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]

        res = hierarchical_rag.search_multi_query_child(
            question="Câu hỏi gốc",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
        )
        self.assertEqual(len(calls), 2)  # Q0 và Q1
        self.assertEqual(res["query_counts"]["executed"], 2)

    def test_09_no_reranker_or_generation_loaded(self):
        """Test 9: Giai đoạn Multi-Child Search không nạp Reranker và không gọi Generation LLM."""
        def fake_gen(q, count):
            return [{"text": "Variant 1", "focus": "paraphrase"}]

        def fake_hybrid(q_text, candidate_k):
            return [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]

        res = hierarchical_rag.search_multi_query_child(
            question="Test",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
        )
        self.assertIn("results", res)
        # Đảm bảo chưa có rerank_score hay answer
        self.assertNotIn("rerank_score", res["results"][0])
        self.assertNotIn("answer", res)

    def test_10_q0_failure_vs_generated_query_partial_status(self):
        """Test 10: Q0 lỗi ném RuntimeError vs Generated Query lỗi trả status multi_query_partial."""
        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        # Scenario 1: Q0 fail -> ném RuntimeError
        def fake_hybrid_failing_q0(q_text, candidate_k):
            if q_text == "Câu hỏi gốc":
                raise RuntimeError("Chroma DB Connection Refused")
            return [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]

        with self.assertRaises(RuntimeError):
            hierarchical_rag.search_multi_query_child(
                question="Câu hỏi gốc",
                config=self.config,
                query_generator_fn=fake_gen,
                hybrid_retriever_fn=fake_hybrid_failing_q0,
            )

        # Scenario 2: Generated query V1 fail -> trả multi_query_partial
        def fake_hybrid_failing_v1(q_text, candidate_k):
            if q_text == "V1":
                raise RuntimeError("Timeout V1")
            return [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]

        res = hierarchical_rag.search_multi_query_child(
            question="Câu hỏi gốc",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid_failing_v1,
        )
        self.assertEqual(res["status"], "multi_query_partial")
        self.assertEqual(res["query_counts"]["failed"], 1)

    def test_11_trace_counts_and_latency_schema(self):
        """Test 11: Schema trace object trả về đúng cấu trúc đếm và phân bộc độ trễ latency."""
        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        def fake_hybrid(q_text, candidate_k):
            return [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]

        res = hierarchical_rag.search_multi_query_child(
            question="Câu hỏi",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
        )
        for key in ["query_counts", "latency_ms", "per_query_stats", "union_child_count", "overlap_distribution"]:
            self.assertIn(key, res)

        for lkey in ["multi_query_gen_ms", "per_query_retrieval_ms", "fusion_ms", "total_ms"]:
            self.assertIn(lkey, res["latency_ms"])

    def test_12_unit_tests_run_100percent_offline(self):
        """Test 12: 100% Unit tests thực thi offline hoàn toàn không gọi API hay storage thật."""
        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        def fake_hybrid(q_text, candidate_k):
            return [{"child_id": "c1", "text": "T1", "source": "s.pdf", "page_start": 1, "page_end": 1}]

        res = hierarchical_rag.search_multi_query_child(
            question="Offline Test",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
        )
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["results"]), 1)


if __name__ == "__main__":
    unittest.main()
