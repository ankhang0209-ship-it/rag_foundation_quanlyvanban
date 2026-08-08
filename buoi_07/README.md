# BUỔI 07 — HOÀN THIỆN RAG PIPELINE VỚI AI AGENT

Dự án RAG (Retrieval-Augmented Generation) hoàn chỉnh xử lý văn bản quy định ngân hàng bằng tiếng Việt, tích hợp Gemini Embedding API, ChromaDB Persistent Vector Store, Confidence Gate, LLM Grounding & Citation Mapping tự động.

---

## 1. Mục tiêu bài học
- Xây dựng pipeline RAG hoàn chỉnh end-to-end có khả năng trích dẫn căn cứ thực tế (`citations`).
- Đảm bảo kiểm soát chất lượng dữ liệu đầu vào, ép kiểu nghiêm ngặt và validate vector embedding trước khi lưu trữ vào database.
- Tích hợp **Confidence Gate** với ngưỡng khoảng cách `RAG_MAX_DISTANCE` nhằm loại bỏ hiện tượng bịa đặt thông tin (hallucination) của LLM.
- Xây dựng giao diện Streamlit thân thiện hỗ trợ kiểm tra trạng thái, index dữ liệu và hỏi đáp trực quan.

---

## 2. Mối quan hệ với Buổi 05 và Buổi 06
- **Buổi 05:** Đã thực hiện đọc văn bản PDF, OCR, chuẩn hóa và chia thành các chunk JSON lưu tại `rag_foundation/buoi_05/output/chunks/`. Buổi 07 đọc trực tiếp các file JSON này làm đầu vào dữ liệu chuẩn bị sẵn.
- **Buổi 06:** Bản demo RAG ban đầu. Buổi 07 hoàn thiện toàn bộ mã nguồn theo chuẩn thiết kế kỹ thuật tại [`SPEC_buoi_07.md`](SPEC_buoi_07.md).

---

## 3. Sơ đồ RAG Pipeline

```text
Chunks JSON Buổi 05
   │
   ▼
[ 1. Validate & Filter Chunk Data ] (rag.py validate)
   │
   ▼
[ 2. Generate Gemini Embeddings ] (gemini-embedding-2, dim: 768)
   │
   ▼
[ 3. Validate Vector Gate ] (Chặn NaN, Infinity, Zero vector)
   │
   ▼
[ 4. Upsert ChromaDB Persistent ] (storage/chroma, space: cosine)
   │
   ▼
[ 5. Semantic Retrieval ] (Query Embedding & Cosine Distance)
   │
   ▼
[ 6. Confidence Gate ] ── (Distance > RAG_MAX_DISTANCE?) ──► [ Insufficient Evidence ]
   │ (Distance <= 0.45)
   ▼
[ 7. LLM Grounding Prompt ] (gemini-3.5-flash-lite)
   │
   ▼
[ 8. Citation Mapping Engine ] (Ánh xạ [E1] -> [Nguồn: ..., tr. ..., chunk: ...])
   │
   ▼
[ 9. Render Streamlit UI / CLI Output ]
```

---

## 4. Cấu trúc thư mục dự án

```text
rag_foundation/buoi_07/
├── SPEC_buoi_07.md             # Bản tả thiết kế kỹ thuật chi tiết
├── buoi_07.md                  # Hướng dẫn thực hành từng bước
├── rag.py                      # Mã nguồn Core RAG Pipeline (Loader, Indexer, Query Engine)
├── app.py                      # Mã nguồn Giao diện Streamlit Web App
├── requirements.txt            # Danh sách gói thư viện giới hạn
├── .env.example                # File mẫu cấu hình biến môi trường
├── .env                        # File cấu hình biến môi trường thực tế (Git ignored)
├── .gitignore                  # File cấu hình bỏ qua Git
├── README.md                   # Tài liệu hướng dẫn & nghiệm thu bài thực hành
├── tests/                      # Thư mục kiểm thử tự động
│   ├── __init__.py
│   ├── fixtures/
│   │   └── chunks_sample.json  # Dữ liệu test fixture mẫu
│   └── test_rag.py             # Bộ kiểm thử Unit Test Suite (30 test methods)
└── storage/                    # Thư mục lưu trữ bộ cơ sở dữ liệu vector persistent
    ├── .gitkeep
    └── chroma/                 # ChromaDB Persistent Storage
```

---

## 5. Điều kiện đầu vào
- Đã chạy xong Buổi 05 và có các file chunk JSON trong `rag_foundation/buoi_05/output/chunks/`.
- Môi trường ảo Python `.venv` của Buổi 05 đã được cài đặt và đang hoạt động tốt.

---

## 6. Hướng dẫn sử dụng môi trường ảo Python (`.venv`)
Sử dụng chính xác interpreter trong môi trường ảo `.venv` của Buổi 05:

- **Windows (PowerShell):** `rag_foundation/buoi_05/.venv/Scripts/python.exe`
- **Linux/macOS:** `rag_foundation/buoi_05/.venv/bin/python`

*(Tuyệt đối không tạo virtual environment mới trong `buoi_07`)*

---

## 7. Hướng dẫn cài đặt Requirements
Mở terminal tại thư mục gốc `RAG` và chạy lệnh:

**Windows PowerShell:**
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m pip install -r .\rag_foundation\buoi_07\requirements.txt
```

**Linux/macOS:**
```bash
./rag_foundation/buoi_05/.venv/bin/python -m pip install -r ./rag_foundation/buoi_07/requirements.txt
```

---

## 8. Hướng dẫn tạo `.env` từ `.env.example`
Sao chép file `.env.example` thành `.env` trong thư mục `rag_foundation/buoi_07/`:

**Windows PowerShell:**
```powershell
Copy-Item .\rag_foundation\buoi_07\.env.example .\rag_foundation\buoi_07\.env
```

**Linux/macOS:**
```bash
cp ./rag_foundation/buoi_07/.env.example ./rag_foundation/buoi_07/.env
```

---

## 9. Giải thích các biến môi trường trong `.env`

| Tên biến môi trường | Giá trị mặc định | Giải thích chi tiết |
|---|---|---|
| `GEMINI_API_KEY` | *(Điền key)* | Khóa API cá nhân để truy cập dịch vụ Gemini API |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Mô hình AI sử dụng để tính toán vector embedding cho văn bản |
| `GEMINI_EMBEDDING_DIM` | `768` | Số chiều của vector embedding (128 đến 3072) |
| `GEMINI_GENERATION_MODEL` | `gemini-3.5-flash-lite` | Mô hình LLM tổng hợp câu trả lời dựa trên context |
| `DEFAULT_TOP_K` | `5` | Số lượng đoạn văn bản liên quan nhất cần truy xuất từ DB (1 đến 20) |
| `RAG_MAX_DISTANCE` | `0.45` | Ngưỡng khoảng cách Cosine tối đa để chấp nhận căn cứ (Confidence Gate) |

---

## 10. Hướng dẫn các lệnh thực thi CLI

> **Lưu ý:** Tất cả các lệnh dưới đây phải chạy từ thư mục gốc `RAG` (thư mục chứa trực tiếp `rag_foundation/`).

### A. Lệnh Kiểm tra dữ liệu đầu vào (`validate`)
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py validate --strategy hierarchical
```

### B. Lệnh Kiểm tra trạng thái Index (`status` - Read-only)
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py status --strategy hierarchical
```

### C. Lệnh Tạo Vector Index (`index`)
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py index --strategy hierarchical
```

### D. Lệnh làm sạch và Re-index (`index --reset`)
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py index --strategy hierarchical --reset
```

### E. Lệnh Truy vấn hỏi đáp CLI (`query`)
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```

### F. Lệnh Chạy Bộ kiểm thử tự động (Unit Test Suite)
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m unittest discover -s .\rag_foundation\buoi_07\tests -v
```

### G. Lệnh Khởi chạy Giao diện Streamlit Web App
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m streamlit run .\rag_foundation\buoi_07\app.py
```
*(Để dừng server Streamlit, nhấn phím `Ctrl + C` tại cửa sổ terminal)*

---

## 11. Giải thích Thuật ngữ Kỹ thuật
- **Strategy:** Phương pháp chia nhỏ văn bản (`hierarchical`, `semantic`, `fixed-size`).
- **Embedding Model & Dim:** Mô hình vector hóa và kích thước không gian vector (768 chiều).
- **Collection Identity:** Tên duy nhất định danh collection ChromaDB (`nhnn-<strategy>-<dimension>-<model_hash>`).
- **Top-K:** Số lượng k đoạn văn bản có khoảng cách gần nhất với câu hỏi được truy xuất.
- **Cosine Distance:** Thước đo độ bất tương đồng giữa 2 vector (giá trị càng nhỏ biểu thị hai văn bản càng tương đồng về ngữ nghĩa).
- **RAG_MAX_DISTANCE & Confidence Gate:** Mức khoảng cách an toàn. Nếu khoảng cách nhỏ nhất lớn hơn ngưỡng này, hệ thống chặn ngay không gọi LLM để tránh bịa đặt.
- **Retrieval-Only:** Trạng thái khi truy xuất được nguồn tham khảo nhưng quá trình gọi LLM tổng hợp bị lỗi.
- **Citation:** Chuỗi trích dẫn chứa nguồn file, trang và chunk_id chính xác được mã hóa tự động bằng mã Python.

---

## 12. Hướng dẫn Khắc phục Lỗi (Troubleshooting)
1. **Lỗi `ModuleNotFoundError` (Thiếu package):** Chạy lệnh cài đặt `pip install -r requirements.txt` bằng đúng interpreter `.venv`.
2. **Lỗi `PermissionError` khi xóa file:** Tắt các tiến trình Python hoặc Streamlit đang chạy ngầm trước khi xóa thư mục `storage/`.
3. **Lỗi `429 RESOURCE_EXHAUSTED` (Rate Limit):** Mã nguồn đã tích hợp cơ chế tự động tạm dừng và thử lại (retry backoff). Nếu vẫn gặp lỗi, hãy chờ 1 phút rồi thực thi lại.
4. **Lỗi `Collection rỗng` hoặc `Mismatch Metadata`:** Thực thi lệnh `index --reset` để tạo mới lại collection tương ứng.

---

## 13. Kịch bản Kiểm thử Thủ công (Manual Test Plan)

Thực hiện kiểm thử trực tiếp trên CLI với 3 câu hỏi mẫu:

### Câu hỏi A (Thuộc phạm vi tài liệu):
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```
- **Kỳ vọng:** Trạng thái `answered`, trích dẫn nguồn từ `TT_02_2023_NHNN.pdf`.

### Câu hỏi B (Thuộc phạm vi tài liệu):
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Việc phân loại nợ và trích lập dự phòng được thực hiện như thế nào?"
```
- **Kỳ vọng:** Trạng thái `answered`, trích dẫn từ `TT_39_2016_NHNN.pdf`.

### Câu hỏi C (Ngoài phạm vi tài liệu):
```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?"
```
- **Kỳ vọng:** Confidence Gate phát hiện khoảng cách > `0.45`, trả về trạng thái `insufficient_evidence` với câu trả lời *"Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp."*, **không gọi LLM sinh câu trả lời giả**.

---

## 14. Giới hạn Demo & Cảnh báo Quan trọng
- **Không phải tư vấn pháp lý:** Câu trả lời tổng hợp từ AI chỉ mang tính chất tham khảo học tập, không thay thế cho văn bản pháp luật chính thức.
- **Hiệu chỉnh Threshold:** Ngưỡng `RAG_MAX_DISTANCE = 0.45` cần được tinh chỉnh thực nghiệm tùy theo từng bài toán và tập dữ liệu cụ thể.
- **Bảo mật Dữ liệu:** Quá trình tạo embedding và tổng hợp câu trả lời sẽ gửi nội dung chunk tới Gemini API. Chỉ áp dụng hệ thống cho các dữ liệu công khai hoặc được phép gửi ra dịch vụ ngoài.
