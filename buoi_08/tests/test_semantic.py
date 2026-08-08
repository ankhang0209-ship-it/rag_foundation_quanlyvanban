"""
Unit tests cho Semantic Candidate Retrieval stage và Status command - Buổi 08.
Sử dụng Mock embedding và temporary ChromaDB client.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import chromadb

# Import module từ rag_foundation/buoi_08/advanced_rag.py
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import advanced_rag
import rag


class TestSemanticRetrieval(unittest.TestCase):
    """
    Tập hợp unit test cho Semantic Candidates Retrieval và Status command.
    """

    def setUp(self):
        """
        Tạo thư mục tạm thời cho ChromaDB testing và sample data.
        """
        self.test_dir = tempfile.mkdtemp()
        self.chroma_dir = Path(self.test_dir) / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.sample_chunks = [
            {
                "chunk_id": "sem_001",
                "text": "Tổ chức tín dụng cơ cấu lại thời hạn trả nợ gốc và lãi vay.",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "strategy": "hierarchical",
            },
            {
                "chunk_id": "sem_002",
                "text": "Ngân hàng thương mại không được cho vay mua cổ phần chưa niêm yết.",
                "source": "TT_06_2023_NHNN.pdf",
                "page_start": 2,
                "page_end": 2,
                "strategy": "hierarchical",
            },
        ]

        self.mock_config = {
            "api_key": "test_mock_gemini_api_key",
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 768,
            "generation_model": "gemini-3.5-flash-lite",
            "max_distance": 0.45,
            "bm25_candidates": 20,
            "semantic_candidates": 20,
            "rrf_k": 60,
            "rrf_bm25_weight": 1.0,
            "rrf_semantic_weight": 1.0,
            "rerank_candidates": 20,
            "final_top_k": 5,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_max_length": 512,
            "rerank_batch_size": 4,
            "rerank_min_score": 0.50,
            "rerank_device": "auto",
        }

    def tearDown(self):
        """
        Dọn dẹp thư mục tạm thời sau khi test.
        """
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("advanced_rag.CHROMA_DIR")
    @patch("advanced_rag.load_config")
    @patch("rag.generate_query_embedding")
    def test_01_semantic_topk_count_order(self, mock_gen_query, mock_cfg, mock_chroma):
        """
        Test 1: Semantic Retrieval trả đúng top-k, count và sắp xếp theo Cosine Distance (nhỏ hơn xếp trước).
        """
        mock_chroma.__str__.return_value = str(self.chroma_dir)
        mock_chroma.exists.return_value = True
        mock_cfg.return_value = self.mock_config

        # Query vector hướng theo trục thứ 1
        query_vec = [1.0] + [0.0] * 767
        mock_gen_query.return_value = query_vec

        # Khởi tạo mock collection trong ChromaDB tạm
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        col_name = rag.get_collection_name("hierarchical", 768, "gemini-embedding-2")
        col = client.create_collection(
            name=col_name,
            metadata={
                "strategy": "hierarchical",
                "embedding_model": "gemini-embedding-2",
                "embedding_dim": 768,
            },
            configuration={"hnsw": {"space": "cosine"}},
        )

        # sem_001 có vector gần trùng hướng query_vec, sem_002 có vector vuông góc
        vec_close = [1.0] + [0.0] * 767
        vec_far = [0.0] * 767 + [1.0]

        col.add(
            ids=["sem_001", "sem_002"],
            documents=[c["text"] for c in self.sample_chunks],
            embeddings=[vec_close, vec_far],
            metadatas=[{"source": c["source"], "page_start": c["page_start"], "page_end": c["page_end"]} for c in self.sample_chunks],
        )

        results = advanced_rag.search_semantic(question="cơ cấu lại nợ", candidate_k=2, strategy="hierarchical")

        self.assertEqual(len(results), 2)
        # Vector sem_001 có khoảng cách cosine nhỏ hơn (0.0) nên được xếp #1
        self.assertEqual(results[0]["chunk_id"], "sem_001")
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertLessEqual(results[0]["semantic_distance"], results[1]["semantic_distance"])

    @patch("advanced_rag.CHROMA_DIR")
    @patch("advanced_rag.load_config")
    @patch("rag.generate_query_embedding")
    def test_02_metadata_completeness(self, mock_gen_query, mock_cfg, mock_chroma):
        """
        Test 2: Kết quả Semantic Candidate trả về đầy đủ metadata chuẩn Buổi 07.
        """
        mock_chroma.__str__.return_value = str(self.chroma_dir)
        mock_chroma.exists.return_value = True
        mock_cfg.return_value = self.mock_config
        mock_gen_query.return_value = [1.0] + [0.0] * 767

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        col_name = rag.get_collection_name("hierarchical", 768, "gemini-embedding-2")
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 768},
        )
        col.add(
            ids=["sem_001"],
            documents=[self.sample_chunks[0]["text"]],
            embeddings=[[1.0] + [0.0] * 767],
            metadatas=[{"source": "TT_02_2023_NHNN.pdf", "page_start": 1, "page_end": 1}],
        )

        results = advanced_rag.search_semantic(question="cơ cấu nợ", candidate_k=1, strategy="hierarchical")
        res = results[0]

        required_keys = {"chunk_id", "text", "source", "page_start", "page_end", "semantic_rank", "semantic_distance"}
        self.assertTrue(required_keys.issubset(set(res.keys())))
        self.assertEqual(res["source"], "TT_02_2023_NHNN.pdf")

    @patch("advanced_rag.CHROMA_DIR")
    @patch("advanced_rag.load_config")
    def test_03_collection_mismatch_blocked(self, mock_cfg, mock_chroma):
        """
        Test 3: Collection mismatch metadata (sai strategy/model/dim) bị chặn với ValueError.
        """
        mock_chroma.__str__.return_value = str(self.chroma_dir)
        mock_chroma.exists.return_value = True
        mock_cfg.return_value = self.mock_config

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        col_name = rag.get_collection_name("hierarchical", 768, "gemini-embedding-2")
        # Tạo collection sai metadata (strategy mismatch)
        client.create_collection(
            name=col_name,
            metadata={"strategy": "fixed-size", "embedding_model": "gemini-embedding-2", "embedding_dim": 768},
        )

        with self.assertRaises(ValueError) as ctx:
            advanced_rag.search_semantic(question="thử nghiệm mismatch", candidate_k=1, strategy="hierarchical")
        self.assertIn("không tương thích", str(ctx.exception))

    @patch("advanced_rag.CHROMA_DIR")
    @patch("advanced_rag.load_config")
    def test_04_status_does_not_create_collection(self, mock_cfg, mock_chroma):
        """
        Test 4: Status command là read-only, tuyệt đối không tự tạo collection mới nếu chưa có.
        """
        mock_chroma.__str__.return_value = str(self.chroma_dir)
        mock_chroma.exists.return_value = True
        mock_cfg.return_value = self.mock_config

        st_info = advanced_rag.get_status(strategy="hierarchical")
        self.assertFalse(st_info["collection_exists"])
        self.assertEqual(st_info["collection_count"], 0)

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        # Xác nhận số lượng collection trong Chroma vẫn bằng 0
        self.assertEqual(len(client.list_collections()), 0)

    @patch("advanced_rag.load_config")
    def test_05_missing_key_fails_without_mock_vector(self, mock_cfg):
        """
        Test 5: Thiếu GEMINI_API_KEY phải raise ValueError rõ ràng, không tự sinh vector giả.
        """
        invalid_config = dict(self.mock_config)
        invalid_config["api_key"] = ""
        mock_cfg.return_value = invalid_config

        with self.assertRaises(ValueError) as ctx:
            advanced_rag.search_semantic(question="truy vấn thiếu key", candidate_k=1, strategy="hierarchical")
        self.assertIn("GEMINI_API_KEY chưa được cấu hình", str(ctx.exception))

    @patch("google.genai.Client")
    def test_06_no_generation_called(self, mock_genai_client):
        """
        Test 6: Giai đoạn Semantic Candidate không được phép gọi LLM Generation API.
        """
        mock_instance = MagicMock()
        mock_genai_client.return_value = mock_instance

        # Thực thi status
        advanced_rag.get_status(strategy="hierarchical")
        mock_instance.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
