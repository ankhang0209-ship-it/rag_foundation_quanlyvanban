# SPECIFICATION — BUỔI 06: END-TO-END RAG FOUNDATION & STREAMLIT UI

## 1. Mục tiêu & Phạm vi (Scope)
- **Mục tiêu:** Xây dựng một ứng dụng RAG hoàn chỉnh (End-to-End RAG) tinh gọn phục vụ Workshop Buổi 06, kết nối luồng truy vấn vector và sinh câu trả lời có dẫn nguồn (Citations).
- **Thành phần chính:**
  1. **Embedding & Vector Database (`rag.py`):**
     - Tạo Vector Embedding cho các văn bản/chunk bằng mô hình Gemini Embedding (`text-embedding-004`).
     - Lưu trữ và truy vấn Vector tương đồng bằng **ChromaDB**.
  2. **Retrieval & Answer Generation (`rag.py`):**
     - Truy vấn `top_k` chunk liên quan nhất theo Cosine Similarity dựa trên câu hỏi của người dùng.
     - Đóng gói Prompt kèm Context và gọi Gemini LLM (`gemini-2.5-flash` / `gemini-1.5-flash`) để sinh câu trả lời.
     - Ép buộc LLM chỉ trả lời dựa trên Context và đưa ra dẫn nguồn (Citation) cụ thể.
  3. **Streamlit Interface (`app.py`):**
     - Giao diện Chatbot trực quan, thân thiện cho người dùng thử nghiệm.
     - Hiển thị câu trả lời của LLM cùng danh sách các Chunk được trích xuất (với chỉ số Similarity Score & Citation).

## 2. Quy định Thiết kế & Ràng buộc (Constraints)
- **Kiến trúc tối giản:** Không chia nhỏ thành quá nhiều module phức tạp. Toàn bộ logic RAG nằm trong `rag.py` và giao diện nằm trong `app.py`.
- **Không tạo:** Không tạo các thư mục `tests`, `docs`, `CLI`, `logging` hay các tiện ích nâng cao chưa cần thiết.
- **Bảo mật:** Không lưu cứng API Key trong code. Đọc key từ `.env` (mẫu tại `.env.example`).
- **Ưu tiên:** Code ngắn gọn, dễ hiểu, có comment tiếng Việt rõ ràng để học viên dễ theo dõi trong Workshop.

## 3. Cấu trúc Dự án Buổi 06
```text
RAG/rag_foundation/buoi_06/
├── SPEC_buoi_06.md      # Tài liệu đặc tả kỹ thuật dự án
├── README.md            # Hướng dẫn cài đặt và khởi chạy
├── requirements.txt     # Các thư viện phụ thuộc (streamlit, chromadb, google-genai,...)
├── .env.example         # File mẫu cấu hình biến môi trường
├── rag.py               # Module xử lý RAG Core (Indexing, Vector Search, QA Gemini)
└── app.py               # Giao diện Streamlit ứng dụng Chatbot RAG
```

## 4. Dữ liệu Đầu vào & Metadata
- **Dữ liệu:** Tiếp nhận dữ liệu văn bản thô hoặc kết quả Chunking JSON từ Buổi 05.
- **Metadata chuẩn:**
  ```json
  {
    "source": "TT_39_2016_NHNN.pdf",
    "page": 1,
    "chunk_id": "chunk_001",
    "text": "Nội dung đoạn trích..."
  }
  ```

## 5. Kịch bản Trực quan hóa trên Streamlit (`app.py`)
1. **Sidebar:** Nhập `GEMINI_API_KEY`, chọn `top_k` (mặc định 3-5), chọn nút "Khởi tạo / Index Dữ liệu".
2. **Main Area:**
   - Khung nhập câu hỏi người dùng.
   - Khung hiển thị câu trả lời RAG kèm Citation.
   - Expander xem chi tiết các Chunk được trích xuất từ ChromaDB (Score, Source, Page).
