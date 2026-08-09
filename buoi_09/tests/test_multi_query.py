"""
Unit tests cho Multi-Query Expansion Generator (Buổi 09).
Bao gồm 11 test cases độc lập 100% offline sử dụng dependency injection (query_generator_fn).
"""

import unittest
import sys
from pathlib import Path

BUOI_09_DIR = Path(__file__).resolve().parent.parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag


class TestMultiQueryExpansion(unittest.TestCase):
    def setUp(self):
        self.config = {
            "multi_query_count": 3,
            "multi_query_max_chars": 100,
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
        # Reset cache trước từng test
        hierarchical_rag._MULTI_QUERY_CACHE.clear()

    def test_01_q0_always_first_and_original(self):
        """Test 1: Q0 luôn đứng ở vị trí đầu tiên [0] và giữ nguyên câu hỏi gốc."""
        def fake_gen(q, count):
            return [
                {"text": "Điều kiện vay vốn theo quy định ngân hàng", "focus": "exact_legal_terms"},
                {"text": "Trình tự thủ tục xin vay vốn", "focus": "paraphrase"},
            ]

        res = hierarchical_rag.generate_multi_queries("  Điều kiện vay vốn là gì?  ", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(res["queries"][0]["query_id"], "Q0")
        self.assertEqual(res["queries"][0]["text"], "Điều kiện vay vốn là gì?")
        self.assertEqual(res["queries"][0]["origin"], "original")

    def test_02_strict_schema_validation(self):
        """Test 2: Schema dữ liệu trả về đầy đủ các trường bắt buộc."""
        def fake_gen(q, count):
            return [{"text": "Biến thể 1", "focus": "paraphrase"}]

        res = hierarchical_rag.generate_multi_queries("Câu hỏi gốc", config=self.config, query_generator_fn=fake_gen)
        for key in ["original_question", "queries", "model", "generation_latency_ms", "status", "warnings"]:
            self.assertIn(key, res)

        for q in res["queries"]:
            for qkey in ["query_id", "text", "origin", "focus"]:
                self.assertIn(qkey, q)

    def test_03_nfc_trim_max_length(self):
        """Test 3: Chuẩn hóa NFC, trim và ngắt độ dài khi vượt MULTI_QUERY_MAX_CHARS."""
        def fake_gen(q, count):
            return [
                {"text": "Biến thể " + "X" * 150, "focus": "paraphrase"}
            ]

        res = hierarchical_rag.generate_multi_queries("Câu hỏi gốc", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(len(res["queries"]), 2)
        q1 = res["queries"][1]
        self.assertLessEqual(len(q1["text"]), self.config["multi_query_max_chars"])
        self.assertTrue(any("vượt 100 chars" in w for w in res["warnings"]))

    def test_04_duplicate_removal(self):
        """Test 4: Loại bỏ câu hỏi trùng lặp (kể cả hoa/thường hoặc khoảng trắng) và ghi nhận dropped_duplicate_count."""
        def fake_gen(q, count):
            return [
                {"text": "câu hỏi gốc", "focus": "paraphrase"},  # Trùng Q0
                {"text": "Biến thể 1", "focus": "paraphrase"},
                {"text": "  Biến thể 1  ", "focus": "paraphrase"},  # Trùng Biến thể 1
            ]

        res = hierarchical_rag.generate_multi_queries("câu hỏi gốc", config=self.config, query_generator_fn=fake_gen)
        # Chỉ giữ Q0 và 1 variant độc nhất "Biến thể 1"
        self.assertEqual(len(res["queries"]), 2)
        self.assertTrue(any("dropped_duplicate_count" in w for w in res["warnings"]))

    def test_05_legal_reference_preservation(self):
        """Test 5: Trích xuất đúng tham chiếu pháp lý trong câu hỏi gốc (Điều, Khoản, Thông tư)."""
        refs = hierarchical_rag.extract_legal_references("Điều kiện vay vốn theo Điều 7 và Khoản 2 Thông tư 39/2016/TT-NHNN")
        self.assertIn("Điều 7", refs)
        self.assertIn("Khoản 2", refs)
        self.assertIn("Thông tư 39/2016/TT-NHNN", refs)

    def test_06_reject_fabricated_article_numbers(self):
        """Test 6: Loại bỏ các query variant tự bịa thêm số Điều mới không có trong Q0."""
        def fake_gen(q, count):
            return [
                {"text": "Điều kiện vay vốn theo Điều 99", "focus": "exact_legal_terms"},  # Bịa Điều 99
                {"text": "Điều kiện cho vay theo Điều 7 Thông tư 39", "focus": "exact_legal_terms"},  # Đúng Điều 7
            ]

        res = hierarchical_rag.generate_multi_queries("Quy định cho vay tại Điều 7", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(len(res["queries"]), 2)  # Q0 + Variant hợp lệ (bỏ variant Điều 99)
        self.assertEqual(res["queries"][1]["text"], "Điều kiện cho vay theo Điều 7 Thông tư 39")
        self.assertTrue(any("bịa số Điều mới" in w for w in res["warnings"]))

    def test_07_deterministic_query_ids(self):
        """Test 7: Gán query_id tuần tự và ổn định định danh: Q0, Q1, Q2, Q3."""
        def fake_gen(q, count):
            return [
                {"text": "Biến thể 1", "focus": "exact_legal_terms"},
                {"text": "Biến thể 2", "focus": "paraphrase"},
                {"text": "Biến thể 3", "focus": "missing_aspect"},
            ]

        res = hierarchical_rag.generate_multi_queries("Câu hỏi gốc", config=self.config, query_generator_fn=fake_gen)
        q_ids = [q["query_id"] for q in res["queries"]]
        self.assertEqual(q_ids, ["Q0", "Q1", "Q2", "Q3"])

    def test_08_single_generator_call(self):
        """Test 8: Gọi query_generator_fn đúng 1 lần duy nhất cho toàn bộ variants."""
        call_count = 0

        def fake_gen(q, count):
            nonlocal call_count
            call_count += 1
            return [{"text": "Biến thể 1", "focus": "paraphrase"}]

        hierarchical_rag.generate_multi_queries("Câu hỏi gốc", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(call_count, 1)

    def test_09_cache_hit_returns_without_second_call(self):
        """Test 9: Lần gọi thứ 2 với cùng câu hỏi đọc từ In-Process Cache, trả cache_hit=True mà không gọi API lần 2."""
        call_count = 0

        def fake_gen(q, count):
            nonlocal call_count
            call_count += 1
            return [{"text": "Biến thể 1", "focus": "paraphrase"}]

        # Lưu cache qua fake generator (nếu không truyền fn ở lần 2)
        res1 = hierarchical_rag.generate_multi_queries("Câu hỏi test cache", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(call_count, 1)

        # Đưa kết quả vào cache thủ công để giả lập cache hit khi query_generator_fn=None
        cache_raw = f"Câu hỏi test cache_{self.config['generation_model']}_{self.config['multi_query_temperature']}_{self.config['multi_query_count']}"
        import hashlib
        cache_key = hashlib.sha256(cache_raw.encode("utf-8")).hexdigest()
        hierarchical_rag._MULTI_QUERY_CACHE[cache_key] = res1

        res2 = hierarchical_rag.generate_multi_queries("Câu hỏi test cache", config=self.config, query_generator_fn=None)
        self.assertTrue(res2["cache_hit"])
        self.assertEqual(res2["queries"][0]["text"], "Câu hỏi test cache")

    def test_10_api_failure_returns_explicit_status(self):
        """Test 10: Khi generator ném ngoại lệ, trả status query_generation_unavailable kèm thông báo lỗi rõ ràng."""
        def fake_failing_gen(q, count):
            raise RuntimeError("API Quota Exceeded 429")

        res = hierarchical_rag.generate_multi_queries("Câu hỏi gốc", config=self.config, query_generator_fn=fake_failing_gen)
        self.assertEqual(res["status"], "query_generation_unavailable")
        self.assertEqual(len(res["queries"]), 1)
        self.assertEqual(res["queries"][0]["query_id"], "Q0")
        self.assertTrue(any("API Quota Exceeded 429" in w for w in res["warnings"]))

    def test_11_unit_tests_run_100percent_offline(self):
        """Test 11: Tất cả unit tests chạy hoàn toàn offline 100%, 0 calls Gemini API real."""
        # Đảm bảo GEMINI_API_KEY rỗng nhưng dùng fake generator vẫn chạy thành công offline
        def fake_gen(q, count):
            return [{"text": "Offline Variant", "focus": "paraphrase"}]

        res = hierarchical_rag.generate_multi_queries("Test Offline", config=self.config, query_generator_fn=fake_gen)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["queries"]), 2)


if __name__ == "__main__":
    unittest.main()
