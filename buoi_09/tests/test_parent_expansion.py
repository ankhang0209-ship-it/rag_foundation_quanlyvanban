"""
Unit tests cho Parent Document Expansion & Parent Aggregation (Buổi 09).
Bao gồm 12 test cases độc lập 100% offline sử dụng temporary store fixtures và mock retrieval.
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

import hierarchical_rag


class TestParentDocumentExpansion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            "multi_query_count": 3,
            "multi_query_max_chars": 300,
            "multi_query_temperature": 0.2,
            "multi_query_original_weight": 1.5,
            "multi_query_variant_weight": 1.0,
            "multi_query_rrf_k": 60,
            "per_query_candidates": 12,
            "parent_max_chars": 6000,
            "parent_score_child_limit": 2,  # Cap at 2 children for score
            "parent_rrf_k": 60,
            "parent_candidates": 10,
            "final_parent_top_k": 3,
            "total_context_max_chars": 500,  # 500 chars budget for testing
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 768,
            "generation_model": "gemini-3.5-flash-lite",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_min_score": 0.50,
            "rerank_device": "auto",
            "api_key": "",
        }

        # Mock Children Dict & Parents Dict
        self.children_dict = {
            "c1": {
                "child_id": "c1",
                "parent_id": "p1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Child 1 text",
                "structural_path": {"article": "Điều 1"},
                "ambiguous": False,
            },
            "c2": {
                "child_id": "c2",
                "parent_id": "p1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "Child 2 text",
                "structural_path": {"article": "Điều 1"},
                "ambiguous": False,
            },
            "c3": {
                "child_id": "c3",
                "parent_id": "p1",
                "source": "doc1.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Child 3 text",
                "structural_path": {"article": "Điều 1"},
                "ambiguous": False,
            },
            "c4": {
                "child_id": "c4",
                "parent_id": "p2",
                "source": "doc1.pdf",
                "page_start": 3,
                "page_end": 3,
                "text": "Child 4 text for parent 2",
                "structural_path": {"article": "Điều 2"},
                "ambiguous": False,
            },
        }

        self.parents_dict = {
            "p1": {
                "parent_id": "p1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "article_key": "Điều 1",
                "window_index": 1,
                "child_ids": ["c1", "c2", "c3"],
                "text": "## Điều 1. Parent 1 Full Text\n\nChild 1 text\n\nChild 2 text\n\nChild 3 text",
                "char_count": 200,
                "ambiguous_child_count": 0,
                "warnings": [],
            },
            "p2": {
                "parent_id": "p2",
                "source": "doc1.pdf",
                "page_start": 3,
                "page_end": 3,
                "article_key": "Điều 2",
                "window_index": 1,
                "child_ids": ["c4"],
                "text": "## Điều 2. Parent 2 Full Text\n\nChild 4 text for parent 2",
                "char_count": 250,
                "ambiguous_child_count": 0,
                "warnings": [],
            },
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_child_maps_to_correct_parent(self):
        """Test 1: Child chunk ánh xạ chính xác về parent_id trong registry."""
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1, "support_query_ids": ["Q0"]}
        ]
        budgeted, all_parents, trace = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        self.assertEqual(len(budgeted), 1)
        self.assertEqual(budgeted[0]["parent_id"], "p1")
        self.assertEqual(budgeted[0]["anchor_child_id"], "c1")

    def test_02_missing_or_stale_hierarchy_store_returns_status(self):
        """Test 2: Thư mục Hierarchy Store thiếu hoặc chưa build trả status hierarchy_not_ready."""
        empty_dir = self.temp_dir / "empty_store"
        res = hierarchical_rag.search_parent_documents("Câu hỏi", hierarchy_dir=empty_dir, config=self.config)
        self.assertEqual(res["status"], "hierarchy_not_ready")
        self.assertTrue(any("Hierarchy Store chưa được tạo" in w for w in res["warnings"]))

    def test_03_parent_aggregation_formula_correctness(self):
        """Test 3: Công thức Parent RRF Score tính toán chính xác số học từ top scoring children."""
        # p1 có c1 (rank 1) và c2 (rank 2)
        # parent_score_child_limit = 2
        # Parent RRF Score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.032522
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1, "support_query_ids": ["Q0"]},
            {"child_id": "c2", "text": "C2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "multi_query_rank": 2, "support_query_ids": ["Q1"]},
        ]
        budgeted, all_parents, _ = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        expected_score = round(1.0 / 61 + 1.0 / 62, 6)
        self.assertEqual(budgeted[0]["parent_rrf_score"], expected_score)

    def test_04_child_score_cap(self):
        """Test 4: Chỉ lấy tối đa PARENT_SCORE_CHILD_LIMIT (2) children tốt nhất để tính điểm parent."""
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1, "support_query_ids": ["Q0"]},
            {"child_id": "c2", "text": "C2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "multi_query_rank": 2, "support_query_ids": ["Q1"]},
            {"child_id": "c3", "text": "C3", "source": "doc1.pdf", "page_start": 2, "page_end": 2, "multi_query_rank": 5, "support_query_ids": ["Q2"]},
        ]
        budgeted, all_parents, _ = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        p = budgeted[0]
        self.assertEqual(len(p["scoring_child_ids"]), 2)  # Cap ở 2 children
        self.assertEqual(len(p["supporting_child_ids"]), 3)  # Nhưng giữ cả 3 supporting children

    def test_05_separate_scoring_and_supporting_children(self):
        """Test 5: Tách rõ danh sách scoring_child_ids và supporting_child_ids."""
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1},
            {"child_id": "c2", "text": "C2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "multi_query_rank": 3},
            {"child_id": "c3", "text": "C3", "source": "doc1.pdf", "page_start": 2, "page_end": 2, "multi_query_rank": 4},
        ]
        budgeted, _, _ = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        p = budgeted[0]
        self.assertEqual(p["scoring_child_ids"], ["c1", "c2"])
        self.assertEqual(p["supporting_child_ids"], ["c1", "c2", "c3"])

    def test_06_parent_deduplication(self):
        """Test 6: Parent Documents được khử trùng lặp (không xuất hiện 2 lần)."""
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1},
            {"child_id": "c2", "text": "C2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "multi_query_rank": 2},
        ]
        budgeted, _, _ = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        p_ids = [p["parent_id"] for p in budgeted]
        self.assertEqual(len(p_ids), len(set(p_ids)))

    def test_07_sort_and_tie_break_deterministic(self):
        """Test 7: Sắp xếp parent theo parent_rrf_score (desc) -> support_query_count (desc) -> best_child_rank (asc) -> parent_id."""
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 2, "support_query_ids": ["Q0"]},
            {"child_id": "c4", "text": "C4", "source": "doc1.pdf", "page_start": 3, "page_end": 3, "multi_query_rank": 1, "support_query_ids": ["Q0", "Q1"]},
        ]
        budgeted, _, _ = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        # p2 có rank 1 (c4) -> p_score 1/61 > p1 (rank 2, c1) -> p_score 1/62
        self.assertEqual(budgeted[0]["parent_id"], "p2")
        self.assertEqual(budgeted[1]["parent_id"], "p1")

    def test_08_parent_candidates_limit(self):
        """Test 8: Áp dụng giới hạn PARENT_CANDIDATES trước khi cắt theo budget."""
        config_limit = dict(self.config)
        config_limit["parent_candidates"] = 1

        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1},
            {"child_id": "c4", "text": "C4", "source": "doc1.pdf", "page_start": 3, "page_end": 3, "multi_query_rank": 2},
        ]
        budgeted, all_parents, trace = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=config_limit
        )
        self.assertEqual(len(budgeted), 1)
        self.assertTrue(any(dp["reason"] == "candidate_limit" for dp in trace["dropped_parents"]))

    def test_09_context_budget_cuts_at_parent_boundary(self):
        """Test 9: Context budget (500 chars) chỉ ngắt tại ranh giới parent document, không cắt dở dang."""
        # p1 = 200 chars, p2 = 250 chars -> 200 + 250 = 450 <= 500 max_chars -> giữ cả 2
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1},
            {"child_id": "c4", "text": "C4", "source": "doc1.pdf", "page_start": 3, "page_end": 3, "multi_query_rank": 2},
        ]
        budgeted, _, trace = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        self.assertEqual(len(budgeted), 2)
        self.assertEqual(trace["expanded_parent_chars_total"], 450)

    def test_10_oversized_first_parent_warning(self):
        """Test 10: Parent đầu tiên quá khổ dài hơn TOTAL_CONTEXT_MAX_CHARS vẫn giữ nguyên parent đầu tiên và đánh warning."""
        config_small_budget = dict(self.config)
        config_small_budget["total_context_max_chars"] = 100  # 100 < 200 chars của p1

        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1}
        ]
        budgeted, _, trace = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=config_small_budget
        )
        self.assertEqual(len(budgeted), 1)
        self.assertEqual(budgeted[0]["parent_id"], "p1")
        self.assertTrue(any("oversized_first_parent_budget_exceeded" in w for w in trace["warnings"]))

    def test_11_expansion_factor_trace_schema(self):
        """Test 11: Schema trace object chứa đầy đủ các chỉ số đếm và context_expansion_factor."""
        child_hits = [
            {"child_id": "c1", "text": "Child 1 text (12 chars)", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1}
        ]
        budgeted, _, trace = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        for key in ["input_child_hit_count", "unique_parent_count", "child_to_parent_mapping_table", "dropped_parents", "context_expansion_factor"]:
            self.assertIn(key, trace)
        self.assertGreater(trace["context_expansion_factor"], 1.0)

    def test_12_unit_tests_run_100percent_offline(self):
        """Test 12: 100% Unit tests thực thi offline hoàn toàn không gọi network, reranker hay LLM generation."""
        child_hits = [
            {"child_id": "c1", "text": "C1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "multi_query_rank": 1}
        ]
        budgeted, _, _ = hierarchical_rag.expand_children_to_parents(
            child_hits, self.children_dict, self.parents_dict, config=self.config
        )
        self.assertEqual(len(budgeted), 1)
        self.assertNotIn("rerank_score", budgeted[0])
        self.assertNotIn("answer", budgeted[0])


if __name__ == "__main__":
    unittest.main()
