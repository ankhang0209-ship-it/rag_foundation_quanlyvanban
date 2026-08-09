"""
Unit tests cho Hierarchy Registry & Parent Store (Buổi 09).
Bao gồm 14 test cases độc lập 100% offline sử dụng temporary directory và fixtures.
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add buoi_09 directory to sys.path
import sys
BUOI_09_DIR = Path(__file__).resolve().parent.parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag


class TestHierarchyRegistry(unittest.TestCase):
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
            "parent_max_chars": 300,  # Nhỏ để dễ test window splitting
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
            "api_key": "test_key",
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_metadata_precedence(self):
        """Test 1: Metadata structure có độ ưu tiên cao nhất khi hợp lệ."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung không chứa heading.",
                "metadata": {
                    "structure": {
                        "chapter": "Chương I",
                        "article": "Điều 5. Quyền hạn",
                    }
                },
            }
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["resolution_method"], "metadata")
        self.assertEqual(resolved[0]["structural_path"]["article"], "Điều 5. Quyền hạn")
        self.assertFalse(resolved[0]["ambiguous"])

    def test_02_heading_inferred_at_start(self):
        """Test 2: Heading được trích xuất từ dòng đầu của chunk text."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 10. Trách nhiệm tổ chức\n\nNội dung chi tiết về trách nhiệm.",
            }
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["resolution_method"], "heading_inferred")
        self.assertIn("Điều 10", resolved[0]["structural_path"]["article"])

    def test_03_carry_forward_same_source(self):
        """Test 3: Carry forward Chương/Điều cho các chunk tiếp theo trong cùng source."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 4. Điều kiện cho vay\n\n1. Có năng lực hành vi dân sự.",
            },
            {
                "chunk_id": "hierarchical_src1_002",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "2. Có mục đích sử dụng vốn hợp pháp.",
            },
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[1]["resolution_method"], "carried_forward")
        self.assertIn("Điều 4", resolved[1]["structural_path"]["article"])

    def test_04_no_carry_forward_across_sources(self):
        """Test 4: Tuyệt đối không carry forward sang source văn bản khác."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 4. Điều kiện cho vay",
            },
            {
                "chunk_id": "hierarchical_src2_001",
                "strategy": "hierarchical",
                "source": "doc2.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung mở đầu văn bản 2 không ghi Điều.",
            },
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertEqual(len(resolved), 2)
        rec_src2 = [r for r in resolved if r["source"] == "doc2.pdf"][0]
        self.assertEqual(rec_src2["resolution_method"], "document_fallback")
        self.assertIsNone(rec_src2["structural_path"]["article"])

    def test_05_inline_dieu_not_falsely_identified(self):
        """Test 5: Cụm 'Điều N' xuất hiện giữa câu trích dẫn không bị nhầm là heading."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Khách hàng phải tuân thủ điều kiện quy định tại Điều 7 Thông tư này.",
            }
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertEqual(resolved[0]["resolution_method"], "document_fallback")
        self.assertIsNone(resolved[0]["structural_path"]["article"])

    def test_06_conflict_sets_ambiguous_and_warning(self):
        """Test 6: Xung đột giữa Metadata và Heading Inferred đặt ambiguous=True và ghi warning."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 8. Lãi suất cho vay\n\nNội dung lãi suất...",
                "metadata": {
                    "structure": {
                        "article": "Điều 7. Điều kiện cho vay",
                    }
                },
            }
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertTrue(resolved[0]["ambiguous"])
        self.assertTrue(any("metadata_heading_conflict" in w for w in resolved[0]["warnings"]))

    def test_07_numeric_chunk_ordering(self):
        """Test 7: Sắp xếp số học theo phần số cuối chunk_id (chống lỗi lexical '...010' trước '...002')."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_010",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Chunk 10 text",
            },
            {
                "chunk_id": "hierarchical_src1_002",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 1. Chunk 2 text",
            },
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        self.assertEqual(resolved[0]["child_id"], "hierarchical_src1_002")
        self.assertEqual(resolved[1]["child_id"], "hierarchical_src1_010")

    def test_08_stable_parent_id(self):
        """Test 8: Tạo parent_id ổn định theo cùng input/config."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 1. Phạm vi",
            }
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        parents1 = hierarchical_rag.build_parent_documents(resolved, self.config)
        parents2 = hierarchical_rag.build_parent_documents(resolved, self.config)
        self.assertEqual(parents1[0]["parent_id"], parents2[0]["parent_id"])
        self.assertIn("Điều_1", parents1[0]["parent_id"])

    def test_09_parent_split_at_child_boundary(self):
        """Test 9: Ngắt Parent thành nhiều window tại ranh giới child khi vượt PARENT_MAX_CHARS."""
        # config parent_max_chars = 300
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 1. Phạm vi\n\n" + "A" * 200,
            },
            {
                "chunk_id": "hierarchical_src1_002",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung tiếp theo...\n\n" + "B" * 200,
            },
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        parents = hierarchical_rag.build_parent_documents(resolved, self.config)
        # Tổng độ dài > 300 -> Phải ngắt thành 2 parent windows (w1 và w2)
        self.assertEqual(len(parents), 2)
        self.assertIn("_w1", parents[0]["parent_id"])
        self.assertIn("_w2", parents[1]["parent_id"])

    def test_10_oversized_child_warning(self):
        """Test 10: Single child dài hơn PARENT_MAX_CHARS được giữ nguyên và đánh warning oversized_single_child."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 1. Single Child Quá Khổ\n\n" + "C" * 500,  # 500 > 300 max_chars
            }
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        parents = hierarchical_rag.build_parent_documents(resolved, self.config)
        self.assertEqual(len(parents), 1)
        self.assertTrue(any("oversized_single_child" in w for w in parents[0]["warnings"]))
        self.assertEqual(parents[0]["char_count"], len(raw_chunks[0]["text"]))

    def test_11_each_child_one_parent(self):
        """Test 11: Mỗi child chunk thuộc về duy nhất 1 parent_id."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "## Điều 1. Text 1",
            },
            {
                "chunk_id": "hierarchical_src1_002",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "Text 2",
            },
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        parents = hierarchical_rag.build_parent_documents(resolved, self.config)
        child_parent_map = {c["child_id"]: c["parent_id"] for c in resolved}
        self.assertEqual(len(child_parent_map), 2)
        self.assertTrue(all(pid.startswith("parent_") for pid in child_parent_map.values()))

    def test_12_parent_pages_count_text_correctness(self):
        """Test 12: Parent page range, child count và ghép text chính xác."""
        raw_chunks = [
            {
                "chunk_id": "hierarchical_src1_001",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "## Điều 1. Đoạn 1",
            },
            {
                "chunk_id": "hierarchical_src1_002",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 2,
                "page_end": 4,
                "text": "Đoạn 2",
            },
        ]
        resolved = hierarchical_rag.resolve_hierarchy_for_chunks(raw_chunks)
        parents = hierarchical_rag.build_parent_documents(resolved, self.config)
        p = parents[0]
        self.assertEqual(p["page_start"], 1)
        self.assertEqual(p["page_end"], 4)
        self.assertEqual(len(p["child_ids"]), 2)
        self.assertIn("Đoạn 1\n\nĐoạn 2", p["text"])

    def test_13_atomic_build_and_manifest_fingerprint(self):
        """Test 13: Ghi store nguyên tử bằng tệp tạm và kiểm tra manifest input fingerprint."""
        input_json = self.temp_dir / "doc1_chunks.json"
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump({
                "source": "doc1.pdf",
                "hierarchical_chunks": [
                    {
                        "chunk_id": "hierarchical_doc1_001",
                        "strategy": "hierarchical",
                        "source": "doc1.pdf",
                        "page_start": 1,
                        "page_end": 1,
                        "text": "## Điều 1. Sample text",
                    }
                ]
            }, f, ensure_ascii=False)

        store_dir = self.temp_dir / "hierarchy_store"
        res = hierarchical_rag.build_hierarchy_store(input_path=input_json, output_dir=store_dir, config=self.config)
        self.assertEqual(res["status"], "success")
        self.assertTrue((store_dir / "manifest.json").exists())
        self.assertTrue((store_dir / "children.json").exists())
        self.assertTrue((store_dir / "parents.json").exists())

        with open(store_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["input_files_fingerprint"]), 1)
        self.assertEqual(manifest["input_files_fingerprint"][0]["filename"], "doc1_chunks.json")

    def test_14_status_does_not_create_or_modify_files(self):
        """Test 14: hierarchy-status là READ-ONLY, tuyệt đối không tạo/sửa file hoặc timestamp."""
        non_existent_dir = self.temp_dir / "non_existent_store"
        status_before = hierarchical_rag.get_hierarchy_status(output_dir=non_existent_dir)
        self.assertFalse(status_before["store_exists"])
        self.assertFalse(non_existent_dir.exists())  # Không được tự mkdir!


if __name__ == "__main__":
    unittest.main()
