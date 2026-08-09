"""
Unit tests cho UI Helpers (Buổi 09).
Thuần Python 100% offline, 0 browser, 0 API/model calls.
"""

import unittest
import sys
from pathlib import Path

BUOI_09_DIR = Path(__file__).resolve().parent.parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag


class TestUIHelpers(unittest.TestCase):
    def test_01_build_query_child_matrix(self):
        """Test 1: Tạo ma trận Query-Child hiển thị rank của từng child chunk."""
        query_set = [
            {"query_id": "Q0", "text": "Q0 text"},
            {"query_id": "Q1", "text": "Q1 text"},
        ]
        child_hits = [
            {
                "child_id": "c1",
                "source": "doc1.pdf",
                "support_query_count": 2,
                "multi_query_rrf_score": 0.04,
                "per_query_ranks": {"Q0": 1, "Q1": 2},
            },
            {
                "child_id": "c2",
                "source": "doc1.pdf",
                "support_query_count": 1,
                "multi_query_rrf_score": 0.02,
                "per_query_ranks": {"Q0": 3},
            },
        ]
        matrix = hierarchical_rag.build_query_child_matrix(child_hits, query_set)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix[0]["c1" if "c1" in matrix[0] else "child_id"], "c1")
        self.assertEqual(matrix[0]["Q0"], 1)
        self.assertEqual(matrix[0]["Q1"], 2)
        self.assertEqual(matrix[1]["Q1"], "—")

    def test_02_format_parent_tree_node(self):
        """Test 2: Chuyển đổi dữ liệu Parent Candidate thành cấu trúc Cây hiển thị UI."""
        parent = {
            "parent_id": "p1",
            "parent_rank": 2,
            "parent_rerank_rank": 1,
            "parent_rank_change": 1,
            "parent_rrf_score": 0.03,
            "parent_rerank_score": 0.92,
            "source": "TT_39_2016_NHNN.pdf",
            "page_start": 1,
            "page_end": 2,
            "structural_path": {"article": "Điều 7"},
            "supporting_child_ids": ["c1", "c2"],
            "support_query_ids": ["Q0", "Q1"],
            "anchor_child_id": "c1",
            "ambiguous": False,
            "warnings": [],
        }
        node = hierarchical_rag.format_parent_tree_node(parent)
        self.assertEqual(node["parent_id"], "p1")
        self.assertEqual(node["rank_before"], 2)
        self.assertEqual(node["rank_after"], 1)
        self.assertEqual(node["rank_change"], 1)
        self.assertEqual(node["supporting_children_count"], 2)

    def test_03_map_status_to_ui_alert(self):
        """Test 3: Ánh xạ status code sang thông báo UI alert tương ứng."""
        al_ready = hierarchical_rag.map_status_to_ui_alert("ready")
        self.assertEqual(al_ready["type"], "success")

        al_not_ready = hierarchical_rag.map_status_to_ui_alert("hierarchy_not_ready")
        self.assertEqual(al_not_ready["type"], "error")

        al_insufficient = hierarchical_rag.map_status_to_ui_alert("insufficient_evidence")
        self.assertEqual(al_insufficient["type"], "warning")

    def test_04_citation_formatting_helper(self):
        """Test 4: Đánh số và tạo nhãn trích dẫn citation chuẩn [P1], [P2]."""
        evidence = [
            {"parent_id": "p1", "source": "s1.pdf", "page_start": 1, "page_end": 1, "parent_rerank_score": 0.95},
            {"parent_id": "p2", "source": "s2.pdf", "page_start": 2, "page_end": 2, "parent_rerank_score": 0.88},
        ]
        ans, cits, ms = hierarchical_rag.generate_rag_answer("Q0", evidence, answer_generator_fn=lambda q, c, cits: "Ans [P1] [P2]")
        self.assertEqual(len(cits), 2)
        self.assertEqual(cits[0]["evidence_id"], "P1")
        self.assertEqual(cits[1]["evidence_id"], "P2")

    def test_05_mode_comparison_row_formatting(self):
        """Test 5: Format bảng so sánh 4 modes trong compare command."""
        res = hierarchical_rag.compare_retrieval_modes(
            question="Test Compare",
            query_generator_fn=lambda q, c: [{"text": "V1", "focus": "paraphrase"}],
            hybrid_retriever_fn=lambda q, k: [{"child_id": "hierarchical_TT_39_2016_NHNN_008", "text": "T1", "source": "doc.pdf", "page_start": 1, "page_end": 1}],
            reranker_fn=lambda pairs: [2.0],
        )
        self.assertEqual(len(res["modes"]), 4)
        for m in ["single_flat", "multi_flat", "single_parent", "multi_parent"]:
            self.assertIn(m, res["modes"])
            self.assertEqual(res["modes"][m]["answer"], "[COMPARE_MODE_NO_ANSWER_GENERATION]")


if __name__ == "__main__":
    unittest.main()
