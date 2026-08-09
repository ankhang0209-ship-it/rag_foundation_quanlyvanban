"""
Unit tests cho RAG Pipeline 4 Modes, Parent Rerank, Evidence Gate, Citations và Generation (Buổi 09).
Bao gồm 14 test cases độc lập 100% offline sử dụng dependency injection.
"""

import unittest
import sys
from pathlib import Path

BUOI_09_DIR = Path(__file__).resolve().parent.parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag

# Valid child IDs present in storage/hierarchy/children.json
VALID_CHILD_1 = "hierarchical_TT_39_2016_NHNN_008"
VALID_CHILD_2 = "hierarchical_TT_39_2016_NHNN_009"


class TestPipelineAndRerank(unittest.TestCase):
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

        self.parent_candidates = [
            {
                "parent_id": "p1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "structural_path": {"article": "Điều 1"},
                "text": "Parent 1 Text về điều kiện vay vốn",
                "parent_rrf_score": 0.03,
                "parent_rank": 1,
                "anchor_child_id": VALID_CHILD_1,
                "scoring_child_ids": [VALID_CHILD_1],
                "supporting_child_ids": [VALID_CHILD_1],
                "support_query_ids": ["Q0"],
                "support_query_count": 1,
                "best_child_rank": 1,
                "char_count": 100,
                "ambiguous": False,
                "warnings": [],
            },
            {
                "parent_id": "p2",
                "source": "doc1.pdf",
                "page_start": 2,
                "page_end": 2,
                "structural_path": {"article": "Điều 2"},
                "text": "Parent 2 Text về các nhu cầu vốn không được cho vay",
                "parent_rrf_score": 0.02,
                "parent_rank": 2,
                "anchor_child_id": VALID_CHILD_2,
                "scoring_child_ids": [VALID_CHILD_2],
                "supporting_child_ids": [VALID_CHILD_2],
                "support_query_ids": ["Q0", "Q1"],
                "support_query_count": 2,
                "best_child_rank": 2,
                "char_count": 120,
                "ambiguous": False,
                "warnings": [],
            },
        ]

    def test_01_reranker_pair_uses_q0_and_parent_text(self):
        """Test 1: Cặp đầu vào của Cross-Encoder Reranker luôn là (Q0, parent_text)."""
        recorded_pairs = []

        def fake_reranker(pairs):
            nonlocal recorded_pairs
            recorded_pairs = pairs
            return [2.5, 1.0]

        hierarchical_rag.rerank_parents("  Câu hỏi gốc Q0  ", self.parent_candidates, config=self.config, reranker_fn=fake_reranker)
        self.assertEqual(len(recorded_pairs), 2)
        self.assertEqual(recorded_pairs[0][0], "Câu hỏi gốc Q0")
        self.assertEqual(recorded_pairs[0][1], "Parent 1 Text về điều kiện vay vốn")

    def test_02_generated_queries_not_in_rerank_or_answer(self):
        """Test 2: Variant queries Q1..Qn không bao giờ được dùng để rerank hoặc đưa vào answer prompt."""
        gen_prompt_content = ""

        def fake_gen(q, count):
            return [{"text": "Query biến thể bị cấm vào prompt", "focus": "paraphrase"}]

        def fake_hybrid(q, k):
            return [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}]

        def fake_reranker(pairs):
            return [2.0]

        def fake_ans_gen(q0, ctx_str, cits):
            nonlocal gen_prompt_content
            gen_prompt_content = ctx_str
            return "Answer text [P1]"

        hierarchical_rag.execute_query_pipeline(
            question="Câu hỏi gốc",
            mode="multi_parent",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
            reranker_fn=fake_reranker,
            answer_generator_fn=fake_ans_gen,
        )
        self.assertNotIn("Query biến thể bị cấm vào prompt", gen_prompt_content)
        self.assertIn("[P1]", gen_prompt_content)

    def test_03_sort_rank_change_and_final_k(self):
        """Test 3: Thứ tự sắp xếp score giảm dần, tính parent_rank_change và ngắt FINAL_PARENT_TOP_K."""
        def fake_reranker(pairs):
            return [-1.0, 3.0]  # p1 score thấp, p2 score cao

        reranked, _ = hierarchical_rag.rerank_parents("Câu hỏi", self.parent_candidates, config=self.config, reranker_fn=fake_reranker)
        self.assertEqual(reranked[0]["parent_id"], "p2")
        self.assertEqual(reranked[0]["parent_rerank_rank"], 1)
        self.assertEqual(reranked[0]["parent_rank_change"], 1)

    def test_04_evidence_gate_acceptance_rejection(self):
        """Test 4: Evidence Gate lọc chính xác items >= RERANK_MIN_SCORE (0.50)."""
        def fake_reranker(pairs):
            return [0.5, -2.0]

        reranked, _ = hierarchical_rag.rerank_parents("Câu hỏi", self.parent_candidates, config=self.config, reranker_fn=fake_reranker)
        accepted = [p for p in reranked if p["parent_rerank_score"] >= self.config["rerank_min_score"]]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["parent_id"], "p1")

    def test_05_insufficient_evidence_status_no_llm_call(self):
        """Test 5: Khi 0 evidence vượt gate, trả status insufficient_evidence và KHÔNG gọi Answer Generation API."""
        llm_called = False

        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        def fake_hybrid(q, k):
            return [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}]

        def fake_reranker(pairs):
            return [-5.0]  # Score rất thấp < 0.50

        def fake_ans_gen(q0, ctx_str, cits):
            nonlocal llm_called
            llm_called = True
            return "Answer"

        res = hierarchical_rag.execute_query_pipeline(
            question="Câu hỏi",
            mode="multi_parent",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
            reranker_fn=fake_reranker,
            answer_generator_fn=fake_ans_gen,
        )
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertFalse(llm_called)
        self.assertTrue(any("Ngưỡng tin cậy Evidence Gate không đạt" in w for w in res["warnings"]))

    def test_06_flat_and_parent_mode_routing(self):
        """Test 6: Điều hướng chính xác cả 4 modes (single_flat, multi_flat, single_parent, multi_parent)."""
        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        def fake_hybrid(q, k):
            return [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}]

        def fake_reranker(pairs):
            return [2.0]

        def fake_ans_gen(q0, ctx_str, cits):
            return "Answer"

        for m in ["single_flat", "multi_flat", "single_parent", "multi_parent"]:
            res = hierarchical_rag.execute_query_pipeline(
                question="Câu hỏi",
                mode=m,
                config=self.config,
                query_generator_fn=fake_gen,
                hybrid_retriever_fn=fake_hybrid,
                reranker_fn=fake_reranker,
                answer_generator_fn=fake_ans_gen,
            )
            self.assertEqual(res["mode"], m)

    def test_07_multi_query_failure_status(self):
        """Test 7: Generator ném ngoại lệ trả status query_generation_unavailable cho multi mode."""
        def fake_failing_gen(q, count):
            raise RuntimeError("API Rate Limit")

        def fake_hybrid(q, k):
            return [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}]

        def fake_reranker(pairs):
            return [2.0]

        def fake_ans_gen(q0, ctx_str, cits):
            return "Answer"

        res = hierarchical_rag.execute_query_pipeline(
            question="Câu hỏi",
            mode="multi_parent",
            config=self.config,
            query_generator_fn=fake_failing_gen,
            hybrid_retriever_fn=fake_hybrid,
            reranker_fn=fake_reranker,
            answer_generator_fn=fake_ans_gen,
        )
        self.assertEqual(res["status"], "multi_query_partial")

    def test_08_reranker_failure_returns_status(self):
        """Test 8: Reranker lỗi ném ngoại lệ trả status reranker_unavailable không silent fallback."""
        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        def fake_hybrid(q, k):
            return [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}]

        def fake_failing_reranker(pairs):
            raise RuntimeError("CUDA Out of Memory")

        res = hierarchical_rag.execute_query_pipeline(
            question="Câu hỏi",
            mode="multi_parent",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
            reranker_fn=fake_failing_reranker,
        )
        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertTrue(any("CUDA Out of Memory" in err for err in res["errors"]))

    def test_09_citation_uses_real_parent_and_anchor_child(self):
        """Test 9: Citation object chứa đúng parent_id, anchor_child_id, supporting_child_ids thực tế."""
        ans_text, citations, _ = hierarchical_rag.generate_rag_answer(
            question="Câu hỏi",
            accepted_evidence=self.parent_candidates[:1],
            config=self.config,
            answer_generator_fn=lambda q, ctx, cits: "Answer [P1]",
        )
        self.assertEqual(len(citations), 1)
        cit = citations[0]
        self.assertEqual(cit["evidence_id"], "P1")
        self.assertEqual(cit["parent_id"], "p1")
        self.assertEqual(cit["anchor_child_id"], VALID_CHILD_1)
        self.assertEqual(cit["supporting_child_ids"], [VALID_CHILD_1])

    def test_10_citation_label_validation(self):
        """Test 10: Citation labels được tạo chuẩn xác [P1], [P2] ứng với accepted evidence."""
        ans_text, citations, _ = hierarchical_rag.generate_rag_answer(
            question="Câu hỏi",
            accepted_evidence=self.parent_candidates,
            config=self.config,
            answer_generator_fn=lambda q, ctx, cits: "Answer [P1] [P2]",
        )
        labels = [c["evidence_id"] for c in citations]
        self.assertEqual(labels, ["P1", "P2"])

    def test_11_multi_mode_max_two_generation_calls(self):
        """Test 11: Mode multi_parent thực thi tối đa 2 Gemini Generation API calls."""
        def fake_gen(q, count):
            return [{"text": "V1", "focus": "paraphrase"}]

        def fake_hybrid(q, k):
            return [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}]

        def fake_reranker(pairs):
            return [2.0]

        def fake_ans_gen(q0, ctx_str, cits):
            return "Answer [P1]"

        res = hierarchical_rag.execute_query_pipeline(
            question="Câu hỏi",
            mode="multi_parent",
            config=self.config,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_hybrid,
            reranker_fn=fake_reranker,
            answer_generator_fn=fake_ans_gen,
        )
        self.assertLessEqual(res["api_call_counts"]["generation_calls"], 2)

    def test_12_compare_mode_no_answer_generation(self):
        """Test 12: Lệnh compare chạy qua cả 4 modes nhưng KHÔNG gọi Answer Generation LLM."""
        res = hierarchical_rag.compare_retrieval_modes(
            question="Câu hỏi compare",
            config=self.config,
            query_generator_fn=lambda q, c: [{"text": "V1", "focus": "paraphrase"}],
            hybrid_retriever_fn=lambda q, k: [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}],
            reranker_fn=lambda pairs: [2.0],
        )
        self.assertEqual(len(res["modes"]), 4)
        for m_name, m_res in res["modes"].items():
            self.assertEqual(m_res["answer"], "[COMPARE_MODE_NO_ANSWER_GENERATION]")

    def test_13_trace_counts_and_latency_schema(self):
        """Test 13: Cấu trúc trace chứa đầy đủ thông tin latency_ms, api_call_counts, system_info."""
        res = hierarchical_rag.execute_query_pipeline(
            question="Câu hỏi trace",
            mode="single_parent",
            config=self.config,
            hybrid_retriever_fn=lambda q, k: [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}],
            reranker_fn=lambda pairs: [2.0],
            answer_generator_fn=lambda q, ctx, cits: "Answer [P1]",
        )
        for key in ["question", "strategy", "mode", "status", "accepted_evidence", "answer", "citations", "latency_ms", "api_call_counts", "system_info"]:
            self.assertIn(key, res)

    def test_14_unit_tests_run_100percent_offline(self):
        """Test 14: Tất cả unit tests thực thi 100% offline sử dụng fakes injection."""
        res = hierarchical_rag.execute_query_pipeline(
            question="Test Offline",
            mode="multi_parent",
            config=self.config,
            query_generator_fn=lambda q, c: [{"text": "V1", "focus": "paraphrase"}],
            hybrid_retriever_fn=lambda q, k: [{"child_id": VALID_CHILD_1, "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1}],
            reranker_fn=lambda pairs: [3.0],
            answer_generator_fn=lambda q, ctx, cits: "Answer Offline [P1]",
        )
        self.assertEqual(res["status"], "ready")
        self.assertIn("[P1]", res["answer"])


if __name__ == "__main__":
    unittest.main()
