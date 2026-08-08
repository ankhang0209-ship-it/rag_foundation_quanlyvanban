# AGENT SPECIFICATION — BUỔI 07: RAG PIPELINE NÂNG CAO

Tài liệu này định nghĩa chi tiết các quy tắc, ràng buộc kỹ thuật và hợp đồng dữ liệu (Data & Pipeline Contracts) cho việc xây dựng RAG Pipeline Buổi 07.

---

## 1. Workspace
- **Vùng được đọc:**
  - `rag_foundation/buoi_05/output/chunks/`
  - `rag_foundation/buoi_05/.venv/`
  - `rag_foundation/buoi_06/`
  - `rag_foundation/buoi_07/`
- **Vùng được ghi:**
  - `rag_foundation/buoi_07/`
- **Ràng buộc:** Không sửa đổi bất kỳ file code, dữ liệu hoặc cấu hình nào thuộc Buổi 05 và Buổi 06.

---

## 2. Python Environment
- Sử dụng trực tiếp môi trường ảo của Buổi 05 tại `rag_foundation/buoi_05/.venv/`.
- Không tạo venv mới trong `buoi_07`.
- Đường dẫn trong code sử dụng `Path(__file__).resolve()` tương đối, không hard-code đường dẫn tuyệt đối theo máy cá nhân.

---

## 3. Input & Data Sources
- Nguồn dữ liệu đầu vào là các file JSON chunk đã được xử lý và chuẩn hóa tại `rag_foundation/buoi_05/output/chunks/`.
- **Tuyệt đối không:** Thực hiện OCR, parse lại PDF hay chia lại chunk dưới bất kỳ hình thức nào.

---

## 4. Packages
- **Chỉ sử dụng trực tiếp các package:**
  - `streamlit`
  - `google-genai`
  - `chromadb`
  - `python-dotenv`
- **Thư viện chuẩn Python:** `argparse`, `hashlib`, `json`, `math`, `os`, `pathlib`, `re`, `tempfile`, `unittest`, `unittest.mock`.
- **Cấm:** Không dùng LangChain, LlamaIndex, Pytest, Hybrid Search, Reranker, Agent Framework phức tạp.

---

## 5. Pipeline Stages
1. **Validate:** Kiểm tra schema và tính hợp lệ của các file chunks JSON đầu vào.
2. **Embedding:** Nhúng vector bằng Gemini Embedding API (`gemini-embedding-2`, dim: 768).
3. **Chroma Persistent:** Lưu trữ vector vào ChromaDB theo chế độ persistent.
4. **Retrieval:** Truy vấn vector ngữ nghĩa (semantic search top-k).
5. **Confidence Gate:** Đánh giá ngưỡng khoảng cách (distance threshold), chặn câu hỏi nếu không đủ căn cứ.
6. **Generation:** Gọi Gemini Flash Lite (`gemini-3.5-flash-lite`) để tổng hợp câu trả lời dựa trên context.
7. **Citation:** Trích dẫn chính xác nguồn, trang và `chunk_id` từ metadata thật.
8. **Streamlit:** Hiển thị giao diện người dùng trực quan.
9. **Unittest Offline:** Kiểm thử tự động không cần kết nối Internet hoặc API key thật.

---

## 6. Data Contract
Mỗi chunk JSON hợp lệ bắt buộc phải chứa đủ 6 trường sau:
- `chunk_id`: String (định danh duy nhất của chunk)
- `strategy`: String (`fixed-size`, `semantic`, hoặc `hierarchical`)
- `source`: String (tên file PDF gốc)
- `page_start`: Integer (trang bắt đầu)
- `page_end`: Integer (trang kết thúc)
- `text`: String (nội dung văn bản của chunk)

---

## 7. Index Contract
- **Cấu trúc Collection:** Mỗi chiến lược chunking (`strategy`) lưu trong một collection riêng hoặc phân tách rõ ràng.
- **Tính nhất quán:** Model và dimension của index và query phải luôn khớp nhau (768 dimensions).
- **Embedding thật:** Chỉ sử dụng vector được tính toán thực tế từ Embedding API; không tạo vector giả hoặc ngẫu nhiên khi gặp lỗi.
- **Kiểm duyệt vector:** Chặn triệt để các vector chứa `NaN`, `Infinity`, `boolean` hoặc toàn số `0` (zero vector).
- **ChromaDB Config:** Sử dụng metric `cosine`, cấu hình `embedding_function=None` (do code tự truyền vector).
- **Tính Idempotent:** Đảm bảo có thể chạy lại việc indexing nhiều lần mà không tạo duplicate dữ liệu.
- **Trạng thái Chờ:** Quá trình validate embedding phải hoàn tất thành công trước khi tiến hành reset/upsert vào database.

---

## 8. Retrieval Contract
- Kết quả truy xuất phải trả về evidence thật đi kèm điểm số khoảng cách (`distance`).
- Chỉ những evidence đạt ngưỡng khoảng cách an toàn (`distance <= RAG_MAX_DISTANCE`) mới được đưa vào ngữ cảnh sinh câu trả lời.
- **Confidence Gate:** Nếu tất cả evidence đều vượt ngưỡng khoảng cách (evidence yếu / thông tin không liên quan), hệ thống dừng ngay và thông báo không đủ căn cứ, **không gọi API sinh câu trả lời (generation)**.

---

## 9. Citation Contract
- Nguồn trích dẫn (citation) bắt buộc được trích rút trực tiếp từ metadata thật trong database (`source`, `page_start`, `page_end`, `chunk_id`).
- Không tin tưởng hoặc sử dụng các trích dẫn do LLM tự bịa ra trong nội dung văn bản.
- Kết quả trả về gồm danh sách `citations` và `warnings`; code tự động thay thế các label hợp lệ bằng trích dẫn chuẩn hóa thực tế.

---

## 10. Security & Credentials
- API Key và các tham số cấu hình chỉ được nạp từ biến môi trường (`.env`).
- Không in API Key, không hard-code secret trong mã nguồn, không log thông tin nhạy cảm.

---

## 11. Testing & Verification
- Tất cả unit test phải chạy offline 100%, sử dụng `unittest.mock` để giả lập API Gemini và bộ lưu trữ tạm thời (`tempfile`).
- Không phụ thuộc vào kết nối mạng hay API key thực tế khi chạy test suite.

---

## 12. Coding Style
- Mã nguồn đơn giản, gọn nhẹ: Tối thiểu số lượng file, class và function.
- Không áp dụng các kiến trúc nhiều tầng phức tạp không cần thiết (Over-engineering).
