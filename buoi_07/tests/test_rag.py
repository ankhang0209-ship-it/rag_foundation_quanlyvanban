"""
Bộ kiểm thử tự động (Unit Test Suite) cho Buổi 07 RAG Pipeline.
Yêu cầu: Runs 100% Offline, dùng unittest + unittest.mock + tempfile, không gọi Internet hay API thật.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm đường dẫn tới buoi_07 để import rag module
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag


class TestLoaderAndValidator(unittest.TestCase):
    """Group 1: Kiểm thử Data Loader và Validator Schema"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_loader_reads_json_list(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung 1",
            }
        ]
        fpath = self.test_dir / "test1.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        chunks, stats = rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c1")

    def test_02_loader_reads_json_object_with_chunks(self):
        data = {
            "source": "doc.pdf",
            "chunks": [
                {
                    "chunk_id": "c2",
                    "strategy": "hierarchical",
                    "source": "doc.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Nội dung 2",
                }
            ],
        }
        fpath = self.test_dir / "test2.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        chunks, stats = rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c2")

    def test_03_selects_only_target_strategy(self):
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "H"},
            {"chunk_id": "c2", "strategy": "semantic", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "S"},
        ]
        fpath = self.test_dir / "test3.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        chunks_h, stats_h = rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")
        self.assertEqual(len(chunks_h), 1)
        self.assertEqual(chunks_h[0]["chunk_id"], "c1")

        chunks_s, stats_s = rag.load_chunks(input_path=self.test_dir, target_strategy="semantic")
        self.assertEqual(len(chunks_s), 1)
        self.assertEqual(chunks_s[0]["chunk_id"], "c2")

    def test_04_missing_required_field_fails(self):
        data = [{"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1}]  # Thiếu text
        fpath = self.test_dir / "test4.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")

    def test_05_field_wrong_type_fails(self):
        data = [{"chunk_id": 123, "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "T"}]  # chunk_id là int
        fpath = self.test_dir / "test5.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")

    def test_06_boolean_page_number_fails(self):
        data = [{"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": True, "page_end": 1, "text": "T"}]
        fpath = self.test_dir / "test6.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")

    def test_07_page_start_greater_than_page_end_fails(self):
        data = [{"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 5, "page_end": 2, "text": "T"}]
        fpath = self.test_dir / "test7.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")

    def test_08_empty_text_skipped_and_counted(self):
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "   "},
            {"chunk_id": "c2", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Hợp lệ"},
        ]
        fpath = self.test_dir / "test8.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        chunks, stats = rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stats["empty_text_skipped"], 1)

    def test_09_duplicate_chunk_id_fails(self):
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "T1"},
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "T2"},
        ]
        fpath = self.test_dir / "test9.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")

    def test_38_non_json_object_record_fails(self):
        data = ["string_record", 12345]
        fpath = self.test_dir / "test38.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            rag.load_chunks(input_path=self.test_dir, target_strategy="hierarchical")


class TestConfigAndEnvironment(unittest.TestCase):
    """Group 2: Cấu hình và Môi trường Safety"""

    def test_20_missing_api_key_fails_clearly_no_fake_vectors(self):
        fake_config = rag.load_config(config_override={"api_key": ""})
        chunks = [{"chunk_id": "c1", "source": "s", "text": "t"}]
        with self.assertRaises(ValueError) as ctx:
            rag.generate_embeddings(chunks, fake_config)
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_24_empty_question_fails(self):
        with self.assertRaises(ValueError):
            rag.query_rag(question="   ", strategy="hierarchical")

    def test_25_top_k_out_of_bounds_fails(self):
        with self.assertRaises(ValueError):
            rag.query_rag(question="Hỏi?", top_k=99, strategy="hierarchical")
        with self.assertRaises(ValueError):
            rag.query_rag(question="Hỏi?", top_k=0, strategy="hierarchical")

    def test_47_cwd_independent_path_resolution(self):
        current_cwd = os.getcwd()
        try:
            # Change CWD to parent directory
            os.chdir(Path(current_cwd).parent)
            cfg = rag.load_config()
            self.assertIn("api_key", cfg)
        finally:
            os.chdir(current_cwd)


class TestVectorValidation(unittest.TestCase):
    """Group 3: Vector Validation Safety Gate"""

    def test_15_vector_count_mismatch_fails(self):
        with self.assertRaises(ValueError):
            rag.validate_vectors([[0.1] * 768], expected_count=2, expected_dim=768)

    def test_16_empty_vector_fails(self):
        with self.assertRaises(ValueError):
            rag.validate_vectors([[]], expected_count=1, expected_dim=768)

    def test_17_dimension_mismatch_fails(self):
        with self.assertRaises(ValueError):
            rag.validate_vectors([[0.1] * 512], expected_count=1, expected_dim=768)

    def test_18_vector_nan_or_inf_fails(self):
        vec_nan = [0.1] * 767 + [float("nan")]
        vec_inf = [0.1] * 767 + [float("inf")]
        with self.assertRaises(ValueError):
            rag.validate_vectors([vec_nan], expected_count=1, expected_dim=768)
        with self.assertRaises(ValueError):
            rag.validate_vectors([vec_inf], expected_count=1, expected_dim=768)

    def test_39_vector_boolean_and_zero_vector_fails(self):
        vec_bool = [0.1] * 767 + [True]
        vec_zero = [0.0] * 768
        with self.assertRaises(ValueError):
            rag.validate_vectors([vec_bool], expected_count=1, expected_dim=768)
        with self.assertRaises(ValueError):
            rag.validate_vectors([vec_zero], expected_count=1, expected_dim=768)


class TestIndexingAndStore(unittest.TestCase):
    """Group 4: Persistent Store & Collection Identity Indexing"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.test_dir = Path(self.temp_dir.name)
        self.storage_dir = self.test_dir / "storage" / "chroma"
        self.chunks_dir = self.test_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.sample_chunk = {
            "chunk_id": "c1",
            "strategy": "hierarchical",
            "source": "doc.pdf",
            "page_start": 1,
            "page_end": 2,
            "text": "Nội dung thử nghiệm",
        }
        with open(self.chunks_dir / "data.json", "w", encoding="utf-8") as f:
            json.dump([self.sample_chunk], f)

        self.mock_config = {
            "api_key": "mock_key",
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "top_k": 5,
            "max_distance": 0.45,
        }

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    @patch("rag.generate_embeddings")
    def test_10_indexing_twice_does_not_duplicate_records(self, mock_gen):
        mock_gen.return_value = [[0.1] * 128]

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            res1 = rag.index_chunks(strategy="hierarchical", input_dir=self.chunks_dir)
            self.assertEqual(res1["total_records"], 1)

            res2 = rag.index_chunks(strategy="hierarchical", input_dir=self.chunks_dir)
            self.assertEqual(res2["total_records"], 1)  # Count giữ nguyên = 1

    @patch("rag.generate_embeddings")
    def test_11_metadata_citation_saved_completely(self, mock_gen):
        mock_gen.return_value = [[0.1] * 128]

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            res = rag.index_chunks(strategy="hierarchical", input_dir=self.chunks_dir)
            client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
            col = client.get_collection(res["collection_name"], embedding_function=None)
            item = col.get(ids=["c1"], include=["metadatas"])
            meta = item["metadatas"][0]

            self.assertEqual(meta["source"], "doc.pdf")
            self.assertEqual(meta["page_start"], 1)
            self.assertEqual(meta["page_end"], 2)
            self.assertEqual(meta["chunk_id"], "c1")

    def test_12_13_collection_identity_changes_on_strategy_dim_model(self):
        col1 = rag.get_collection_name("hierarchical", 128, "model-A")
        col2 = rag.get_collection_name("semantic", 128, "model-A")
        col3 = rag.get_collection_name("hierarchical", 256, "model-A")
        col4 = rag.get_collection_name("hierarchical", 128, "model-B")

        self.assertNotEqual(col1, col2)
        self.assertNotEqual(col1, col3)
        self.assertNotEqual(col1, col4)

    @patch("rag.generate_embeddings")
    def test_19_embedding_error_before_upsert_does_not_add_records(self, mock_gen):
        mock_gen.side_effect = ValueError("EMBEDDING_FAILED")

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            with self.assertRaises(ValueError):
                rag.index_chunks(strategy="hierarchical", input_dir=self.chunks_dir)

            client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
            cols = client.list_collections()
            self.assertEqual(len(cols), 0)

    def test_40_status_on_empty_storage_does_not_create_collection(self):
        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            rag.status_command(strategy="hierarchical")

            if self.storage_dir.exists():
                client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
                cols = client.list_collections()
                self.assertEqual(len(cols), 0)

    @patch("rag.generate_embeddings")
    def test_41_reset_failing_embedding_preserves_old_collection(self, mock_gen):
        mock_gen.return_value = [[0.1] * 128]

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            # Step 1: Index thành công lần 1
            res1 = rag.index_chunks(strategy="hierarchical", input_dir=self.chunks_dir)
            col_name = res1["collection_name"]

            # Step 2: Thử index --reset nhưng embedding bị lỗi
            mock_gen.side_effect = ValueError("GEMINI_ERROR_ON_RESET")

            with self.assertRaises(ValueError):
                rag.index_chunks(strategy="hierarchical", reset=True, input_dir=self.chunks_dir)

            # Khôi phục kiểm tra: collection cũ vẫn còn nguyên 1 record
            client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
            col = client.get_collection(col_name, embedding_function=None)
            self.assertEqual(col.count(), 1)


class TestRetrievalAndConfidenceGate(unittest.TestCase):
    """Group 5: Retrieval & Confidence Gate Enforcement"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.test_dir = Path(self.temp_dir.name)
        self.storage_dir = self.test_dir / "storage" / "chroma"

        self.mock_config = {
            "api_key": "mock_key",
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "top_k": 5,
            "max_distance": 0.45,
        }

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_26_empty_collection_query_fails(self):
        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            with self.assertRaises(ValueError):
                rag.query_rag("Câu hỏi?", strategy="hierarchical")

    @patch("google.genai.Client")
    @patch("rag.generate_query_embedding")
    def test_27_best_evidence_exceeds_threshold_insufficient_evidence(self, mock_q_embed, mock_genai):
        mock_q_embed.return_value = [0.1] * 128

        # Khởi tạo DB tạm với 1 evidence distance 0.8 (> 0.45)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", 128, "gemini-embedding-2")
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 128},
            embedding_function=None,
        )
        col.add(
            ids=["c1"],
            documents=["Văn bản không liên quan"],
            embeddings=[[0.9] * 128],
            metadatas=[{"source": "s.pdf", "page_start": 1, "page_end": 1, "chunk_id": "c1", "strategy": "hierarchical"}],
        )

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            res = rag.query_rag("Câu hỏi bất kỳ?", strategy="hierarchical")

            self.assertEqual(res["status"], "insufficient_evidence")
            self.assertEqual(res["citations"], [])
            # Đảm bảo Generation Client KHÔNG được gọi
            mock_genai.assert_not_called()

    @patch("google.genai.Client")
    @patch("rag.generate_query_embedding")
    def test_28_29_30_31_43_44_evidence_accepted_triggers_generation(self, mock_q_embed, mock_genai_cls):
        mock_q_embed.return_value = [0.1] * 128

        # Mock Gemini Generation API
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Theo quy định [E1] cho vay.")
        mock_genai_cls.return_value = mock_client

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", 128, "gemini-embedding-2")
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 128},
            embedding_function=None,
        )
        # Add 1 evidence gần (accepted) và 1 evidence xa (rejected)
        col.add(
            ids=["c1", "c2"],
            documents=["Đoạn văn bản hợp lệ 1", "Đoạn văn bản xa 2"],
            embeddings=[[0.1] * 128, [0.9] * 128],
            metadatas=[
                {"source": "s1.pdf", "page_start": 1, "page_end": 1, "chunk_id": "c1", "strategy": "hierarchical"},
                {"source": "s2.pdf", "page_start": 2, "page_end": 3, "chunk_id": "c2", "strategy": "hierarchical"},
            ],
        )

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            res = rag.query_rag("Hỏi về hợp lệ 1?", strategy="hierarchical")

            self.assertEqual(res["status"], "answered")
            mock_client.models.generate_content.assert_called_once()

            # Kiểm tra prompt được tạo
            called_args = mock_client.models.generate_content.call_args[1]
            prompt = called_args["contents"]

            self.assertIn("Hỏi về hợp lệ 1?", prompt)
            self.assertIn("[E1]", prompt)
            self.assertIn("Đoạn văn bản hợp lệ 1", prompt)
            self.assertNotIn("Đoạn văn bản xa 2", prompt)  # Chunk vượt threshold không có trong prompt
            self.assertIn("--- CONTEXT BẮT ĐẦU ---", prompt)  # Delimiter bọc context


class TestCitationMappingAndFailures(unittest.TestCase):
    """Group 6: Citation Engine & Error Recovery"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.test_dir = Path(self.temp_dir.name)
        self.storage_dir = self.test_dir / "storage" / "chroma"

        self.mock_config = {
            "api_key": "mock_key",
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "top_k": 5,
            "max_distance": 0.45,
        }

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    @patch("google.genai.Client")
    @patch("rag.generate_query_embedding")
    def test_32_33_34_35_45_citation_mapping_and_fake_label_warning(self, mock_q_embed, mock_genai_cls):
        mock_q_embed.return_value = [0.1] * 128

        # Mock LLM trả câu trả lời có nhãn đúng [E1], [E2] và nhãn ảo [E99]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Quy định 1 [E1] và quy định 2 [E2] kèm nhãn sai [E99].")
        mock_genai_cls.return_value = mock_client

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", 128, "gemini-embedding-2")
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 128},
            embedding_function=None,
        )
        col.add(
            ids=["c1", "c2"],
            documents=["Nội dung 1", "Nội dung 2"],
            embeddings=[[0.1] * 128, [0.11] * 128],
            metadatas=[
                {"source": "s1.pdf", "page_start": 5, "page_end": 5, "chunk_id": "c1", "strategy": "hierarchical"},  # Single page
                {"source": "s2.pdf", "page_start": 10, "page_end": 12, "chunk_id": "c2", "strategy": "hierarchical"},  # Page range
            ],
        )

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            res = rag.query_rag("Cho vay?", strategy="hierarchical")

            self.assertEqual(res["status"], "answered")
            # 37: Result Schema đủ 8 fields
            for field in ["status", "answer", "evidence", "citations", "warnings", "collection", "strategy", "top_k"]:
                self.assertIn(field, res)

            # 32 & 34: E1 single page mapped
            self.assertIn("[Nguồn: s1.pdf, tr. 5, chunk: c1]", res["answer"])
            # 33 & 34: E2 range page mapped
            self.assertIn("[Nguồn: s2.pdf, tr. 10-12, chunk: c2]", res["answer"])

            # 35 & 45: [E99] bị xóa khỏi answer và ghi nhận warning
            self.assertNotIn("[E99]", res["answer"])
            self.assertTrue(any("[E99]" in w for w in res["warnings"]))

    @patch("google.genai.Client")
    @patch("rag.generate_query_embedding")
    def test_36_46_generation_error_or_empty_text_returns_retrieval_only(self, mock_q_embed, mock_genai_cls):
        mock_q_embed.return_value = [0.1] * 128

        # Mock Generation ném Exception
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("GENERATION_API_OUTAGE")
        mock_genai_cls.return_value = mock_client

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        client = rag.chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", 128, "gemini-embedding-2")
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 128},
            embedding_function=None,
        )
        col.add(
            ids=["c1"],
            documents=["Nội dung 1"],
            embeddings=[[0.1] * 128],
            metadatas=[{"source": "s1.pdf", "page_start": 1, "page_end": 1, "chunk_id": "c1", "strategy": "hierarchical"}],
        )

        with patch("rag.load_config", return_value=self.mock_config), patch("rag.CHROMA_DIR", self.storage_dir):
            res = rag.query_rag("Cho vay?", strategy="hierarchical")

            self.assertEqual(res["status"], "retrieval_only")
            self.assertEqual(len(res["evidence"]), 1)
            self.assertEqual(res["citations"], [])
            self.assertTrue(len(res["warnings"]) > 0)


if __name__ == "__main__":
    unittest.main()
