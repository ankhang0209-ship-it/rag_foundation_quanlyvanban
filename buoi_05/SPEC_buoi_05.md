# SPECIFICATION — BUỔI 05: OCR & CHUNKING STRATEGIES FOR VIETNAMESE DOCUMENTS

## 1. Đầu vào & Mục tiêu (Input & Objectives)
- **Đầu vào:** Tất cả các tệp PDF tiếng Việt trong thư mục `RAG/rag_foundation/buoi_05/datademo/` (đặc biệt là PDF scan hoặc PDF có lỗi font/mojibake).
- **Mục tiêu:**
  1. Trích xuất văn bản từ PDF (Ưu tiên PyMuPDF text layer; nếu rỗng, lỗi font, lỗi encoding hoặc ký tự lạ thì tự động chuyển đổi sang OCR với LlamaParse).
  2. Chuẩn hóa văn bản Tiếng Việt về chuẩn **Unicode NFC**.
  3. Xuất văn bản thô (raw text) và kết quả chunking ra thư mục `RAG/rag_foundation/buoi_05/output/`.
  4. Thực hiện và báo cáo so sánh 3 chiến lược chia chunking:
     - **Fixed-size:** Cắt cố định theo số ký tự/token với overlap (ví dụ: `chunk_size=500`, `overlap=100`).
     - **Semantic:** Ưu tiên ngắt theo ranh giới đoạn văn thường ngắt như ngắt đoạn, kết đoạn, cách dòng (`\n\n`), bảo đảm không cắt giữa câu/từ.
     - **Hierarchical:** Chia theo cấu trúc văn bản pháp luật Việt Nam, trong đó mỗi mốc Chương → Mục → Điều/Khoản → Điểm sẽ thành mốc bắt đầu của một chunk độc lập.

## 2. Đầu ra & Cấu trúc Metadata (Outputs & Schema)
Đầu ra bao gồm tệp text thô chuẩn Unicode NFC và tệp JSON chứa danh sách các chunk. Mỗi chunk phải chứa thông tin metadata theo chuẩn Pydantic Schema:

```json
{
  "chunk_id": "hierarchical_TT_02_2023_NHNN_001",
  "strategy": "hierarchical",
  "source": "TT_02_2023_NHNN.pdf",
  "page_start": 1,
  "page_end": 1,
  "text": "Điều 1. Phạm vi điều chỉnh...",
  "metadata": {
    "ocr_used": true,
    "language": "vi",
    "chapter": "Chương I",
    "section": "Mục 1",
    "article": "Điều 1",
    "clause": "Khoản 1",
    "point": "Điểm a"
  }
}
```

Báo cáo thống kê đầu ra phải hiển thị các chỉ số của 3 chiến lược:
- Tổng số lượng chunk tạo ra.
- Độ dài ký tự nhỏ nhất (Min), lớn nhất (Max), và trung bình (Avg).

## 3. Quản lý Secret & API Key (.env)
- Mã nguồn phải nạp API Key `LLAMA_CLOUD_API_KEY` từ tệp `.env` nằm trong thư mục `src/` (`RAG/rag_foundation/buoi_05/src/.env`).
- **Quy định bảo mật:** Tuyệt đối **không đọc/in/log giá trị** của Key ra màn hình Terminal, file log hoặc trên giao diện UI Streamlit. Chỉ kiểm tra sự tồn tại (PASS/FAIL) của Key.

## 4. Các Ràng buộc Bắt buộc (Constraints)
1. **Không tạo Embedding Vector.**
2. **Không lưu vào Vector Database.**
3. **Không gọi LLM** để tóm tắt hay sinh câu trả lời.
4. **Không ghi đè hoặc thay đổi các tệp PDF gốc** trong `datademo/`.
5. **Không tự bịa đặt tiêu đề/cấu trúc:** Nếu văn bản không chứa mốc Chương/Điều, hệ thống không tự bịa đặt mà xuất cảnh báo (warning) và ngắt theo đoạn tự nhiên.
6. **Mức độ phức tạp:** Code ở mức demo giảng dạy đơn giản, tường minh, dễ đọc, không phức tạp hóa mã nguồn nhưng không được bỏ sót bất kỳ yêu cầu nào.

## 5. Cấu trúc Thư mục Buổi 05
```text
RAG/rag_foundation/buoi_05/
├── SPEC_buoi_05.md
├── app.py                      # UI Streamlit Trực quan hóa
├── requirements.txt
├── datademo/                   # PDF Tiếng Việt công khai
├── output/                     # Kết quả raw.txt và chunks.json
├── storage/                    # Lưu trữ trung gian
├── src/
│   ├── .env                    # Đứa API Key LlamaParse
│   ├── check_ocr_env.py        # Kiểm tra môi trường
│   ├── ocr_processor.py        # Trích xuất PyMuPDF & OCR LlamaParse
│   ├── chunker.py              # 3 chiến lược Chunking
│   └── main_pipeline.py        # CLI Pipeline chính
└── tests/
    └── test_chunker.py         # Unit tests
```