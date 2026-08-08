"""
Unit tests cho BM25 Lexical Retrieval và Vietnamese Legal Tokenizer - Buổi 08.
Kiểm tra hoàn toàn offline, không gọi Gemini API, ChromaDB hay Cross-Encoder.
"""

import unittest
from pathlib import Path
from typing import Dict, List, Any

# Import module từ rag_foundation/buoi_08/advanced_rag.py
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from advanced_rag import tokenize_vi_legal, BM25Retriever, search_bm25


class TestBM25LexicalRetrieval(unittest.TestCase):
    """
    Tập hợp unit test cho Tokenizer và BM25 Retrieval Engine.
    """

    def setUp(self):
        """
        Khởi tạo tập sample chunks giả lập cho unit test.
        """
        self.sample_chunks = [
            {
                "chunk_id": "chunk_001",
                "text": "Điều 7 Khoản 2. Tổ chức tín dụng xem xét cơ cấu lại thời hạn trả nợ gốc và lãi.",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "strategy": "fixed-size",
            },
            {
                "chunk_id": "chunk_002",
                "text": "Điều 8. Quy định về lãi suất cho vay và phí dịch vụ ngân hàng thương mại.",
                "source": "TT_39_2016_NHNN.pdf",
                "page_start": 2,
                "page_end": 2,
                "strategy": "fixed-size",
            },
            {
                "chunk_id": "chunk_003",
                "text": "Doanh nghiệp đăng ký thành lập công ty và kê khai thuế giá trị gia tăng tại cơ quan thuế.",
                "source": "LUAT_THUE_2020.pdf",
                "page_start": 1,
                "page_end": 1,
                "strategy": "fixed-size",
            },
        ]

    def test_01_tokenizer_preserves_vietnamese_diacritics(self):
        """
        Test 1: Tokenizer giữ nguyên dấu tiếng Việt chuẩn Unicode NFC.
        """
        text = "cơ cấu lại thời hạn trả nợ"
        tokens = tokenize_vi_legal(text)
        expected = ["cơ", "cấu", "lại", "thời", "hạn", "trả", "nợ"]
        self.assertEqual(tokens, expected)

    def test_02_tokenizer_preserves_article_and_clause_numbers(self):
        """
        Test 2: Tokenizer bảo toàn chữ 'điều', 'khoản' và các con số.
        """
        text = "Điều 7, Khoản 2"
        tokens = tokenize_vi_legal(text)
        self.assertIn("điều", tokens)
        self.assertIn("7", tokens)
        self.assertIn("khoản", tokens)
        self.assertIn("2", tokens)
        self.assertEqual(tokens, ["điều", "7", "khoản", "2"])

    def test_03_corpus_and_query_same_preprocessing(self):
        """
        Test 3: Corpus và Query đều sử dụng chung hàm tokenize_vi_legal.
        """
        raw_corpus = "ĐIỀU 7. CƠ CẤU LẠI THỜI HẠN TRẢ NỢ"
        raw_query = "Điều 7, cơ cấu lại thời hạn..."

        tokens_corpus = tokenize_vi_legal(raw_corpus)
        tokens_query = tokenize_vi_legal(raw_query)

        # Cả hai đều chứa 'điều', '7', 'cơ', 'cấu', 'lại', 'thời', 'hạn'
        common_tokens = set(tokens_corpus).intersection(set(tokens_query))
        self.assertGreater(len(common_tokens), 0)
        self.assertIn("điều", common_tokens)
        self.assertIn("7", common_tokens)

    def test_04_exact_legal_term_ranked_higher(self):
        """
        Test 4: Đoạn văn chứa exact legal term được BM25 xếp hạng cao hơn đoạn không chứa.
        """
        query = "cơ cấu lại thời hạn trả nợ"
        results = search_bm25(question=query, chunks=self.sample_chunks, candidate_k=3)

        self.assertGreater(len(results), 0)
        # chunk_001 chứa exact term nên phải xếp vị trí đầu tiên (#1)
        top_result = results[0]
        self.assertEqual(top_result["chunk_id"], "chunk_001")
        self.assertGreater(top_result["bm25_score"], 0)

    def test_05_candidate_k_larger_than_corpus(self):
        """
        Test 5: candidate_k lớn hơn kích thước corpus vẫn hoạt động bình thường mà không bị crash.
        """
        results = search_bm25(question="Điều 7", chunks=self.sample_chunks, candidate_k=100)
        # Số lượng ứng viên tối đa trả về chỉ bằng kích thước corpus (3)
        self.assertEqual(len(results), len(self.sample_chunks))

    def test_06_empty_question_fails(self):
        """
        Test 6: Truy vấn rỗng hoặc chỉ chứa khoảng trắng/dấu câu rỗng phải quăng ValueError.
        """
        with self.assertRaises(ValueError):
            search_bm25(question="", chunks=self.sample_chunks)

        with self.assertRaises(ValueError):
            search_bm25(question="   !!! ???   ", chunks=self.sample_chunks)

    def test_07_tie_break_deterministic(self):
        """
        Test 7: Tie-break ổn định bằng chunk_id khi các chunks có cùng điểm BM25.
        """
        duplicate_score_chunks = [
            {
                "chunk_id": "chunk_B",
                "text": "Ngân hàng thương mại",
                "source": "A.pdf",
                "page_start": 1,
                "page_end": 1,
                "strategy": "fixed-size",
            },
            {
                "chunk_id": "chunk_A",
                "text": "Ngân hàng thương mại",
                "source": "B.pdf",
                "page_start": 1,
                "page_end": 1,
                "strategy": "fixed-size",
            },
        ]
        results = search_bm25(question="Ngân hàng", chunks=duplicate_score_chunks, candidate_k=2)
        # Cùng điểm score, chunk_A xếp trước chunk_B theo thứ tự bảng chữ cái
        self.assertEqual(results[0]["chunk_id"], "chunk_A")
        self.assertEqual(results[1]["chunk_id"], "chunk_B")

    def test_08_no_external_api_calls(self):
        """
        Test 8: Kiểm thử hoàn toàn offline, không phụ thuộc môi trường mạng hoặc Gemini API.
        """
        results = search_bm25(question="công ty kê khai thuế", chunks=self.sample_chunks, candidate_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_003")


if __name__ == "__main__":
    unittest.main()
