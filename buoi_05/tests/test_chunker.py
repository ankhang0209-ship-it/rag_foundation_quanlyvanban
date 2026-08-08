"""
BÀI TEST TỰ ĐỘNG CHO BÀI THỰC HÀNH BUỔI 05 (CHUNKING & OCR)
File: RAG/rag_foundation/buoi_05/tests/test_chunker.py
"""

import sys
import unittest
from pathlib import Path

# Add src and buoi_05 directory to sys.path
BUOI_05_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BUOI_05_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BUOI_05_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_05_DIR))

try:
    from ocr_processor import normalize_vietnamese_nfc, check_text_quality  # type: ignore
    from chunker import chunk_fixed_size, chunk_semantic, chunk_hierarchical  # type: ignore
except ImportError:
    from src.ocr_processor import normalize_vietnamese_nfc, check_text_quality  # type: ignore
    from src.chunker import chunk_fixed_size, chunk_semantic, chunk_hierarchical  # type: ignore

class TestBuoi05Pipeline(unittest.TestCase):

    def setUp(self):
        self.sample_legal_text = (
            "CHƯƠNG I\n"
            "QUY ĐỊNH CHUNG\n\n"
            "Điều 1. Phạm vi điều chỉnh\n"
            "Thông tư này quy định về hoạt động cho vay của tổ chức tín dụng, chi nhánh ngân hàng nước ngoài đối với khách hàng.\n\n"
            "Điều 2. Đối tượng áp dụng\n"
            "1. Tổ chức tín dụng bao gồm ngân hàng thương mại, ngân hàng hợp tác xã.\n"
            "2. Khách hàng vay vốn tại tổ chức tín dụng."
        )

    def test_unicode_nfc_normalization(self):
        # Test combined accents vs decomposed accent
        raw_decomposed = "Thông tư này quy định"
        normalized = normalize_vietnamese_nfc(raw_decomposed)
        self.assertEqual(len(normalized), len("Thông tư này quy định"))

    def test_check_text_quality_valid(self):
        is_valid, msg = check_text_quality(self.sample_legal_text)
        self.assertTrue(is_valid, f"Text quality check failed: {msg}")

    def test_check_text_quality_mojibake(self):
        mojibake_text = "Didu 1. Ph4m vi dliu6 kh6khin thdng"
        is_valid, msg = check_text_quality(mojibake_text)
        self.assertFalse(is_valid, "Mojibake text should be detected as invalid")

    def test_fixed_size_chunking(self):
        chunks = chunk_fixed_size(self.sample_legal_text, "test.pdf", chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].strategy, "fixed-size")
        self.assertEqual(chunks[0].source, "test.pdf")

    def test_semantic_chunking(self):
        chunks = chunk_semantic(self.sample_legal_text, "test.pdf", max_chunk_size=150)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].strategy, "semantic")

    def test_hierarchical_chunking(self):
        chunks = chunk_hierarchical(self.sample_legal_text, "test.pdf")
        self.assertTrue(any("Điều 1" in c.text for c in chunks), "Cần có ít nhất 1 chunk chứa Điều 1")

if __name__ == "__main__":
    unittest.main()
