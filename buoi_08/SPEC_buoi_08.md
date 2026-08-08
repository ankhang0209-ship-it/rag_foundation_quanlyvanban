# SPECIFICATION BUỔI 08 — ADVANCED HYBRID RAG ENGINE

## 1. Workspace và Security
- **Workspace Scope**: Tất cả tài nguyên code, dữ liệu cấu hình, lưu trữ index và báo cáo đánh giá của Buổi 08 phải nằm hoàn toàn trong thư mục `rag_foundation/buoi_08/`.
- **Security & Secret Handling**:
  - Tuyệt đối không lưu vết hoặc commit API key thực tế vào Git. Tất cả các biến môi trường nhạy cảm phải được quản lý thông qua tệp `.env` (đã được cấu hình trong `.gitignore`).
  - `.env.example` chỉ cung cấp cấu trúc khai báo biến môi trường mẫu với giá trị rỗng hoặc giá trị mặc định an toàn.
  - Không đọc hoặc sử dụng giá trị secret từ `.env` của các buổi học khác.

## 2. Quan hệ với Buổi 05 và Buổi 07
- **Tương tác với Buổi 05**:
  - Nguồn dữ liệu chunks duy nhất được kế thừa là các tệp JSON output được sinh ra từ pipeline của Buổi 05 tại đường dẫn tĩnh `rag_foundation/buoi_05/output/chunks/`.
  - Sử dụng chung Python Virtual Environment tại `rag_foundation/buoi_05/.venv/` để đảm bảo tính nhất quán của hệ sinh thái thư viện.
- **Tương tác với Buổi 07**:
  - Buổi 08 sao chép `rag_foundation/buoi_07/rag.py` sang `rag_foundation/buoi_08/rag.py` để làm **Semantic Baseline**.
  - Tệp `buoi_08/rag.py` tự quản lý môi trường `.env` và cơ sở dữ liệu `storage/chroma/` riêng độc lập với Buổi 07. Không import runtime trực tiếp từ Buổi 07.
  - Tuyệt đối không thay đổi hay chỉnh sửa bất kỳ tệp nguồn nào thuộc Buổi 05, Buổi 06 hay Buổi 07.

## 3. Data Contract
mỗi record chunk được nạp vào hệ thống phải thỏa mãn schema chuẩn từ Buổi 07:
```json
{
  "chunk_id": "string (không rỗng)",
  "strategy": "fixed-size | semantic | hierarchical",
  "source": "string (tên tệp nguồn PDF)",
  "page_start": "integer (>= 1)",
  "page_end": "integer (>= page_start)",
  "text": "string (không rỗng)",
  "metadata": "object"
}
```
Mọi record vi phạm kiểu dữ liệu hoặc thiếu trường bắt buộc phải bị từ chối bởi Validator trước khi đưa vào indexing hoặc retrieval.

## 4. BM25 Tokenizer/Retrieval Contract
- **Tokenizer Standard**:
  - Chuẩn hóa văn bản: Chuyển toàn bộ ký tự sang chữ thường (lowercase), loại bỏ dấu câu và các ký tự đặc biệt không thuộc chữ cái/chữ số tiếng Việt.
  - Tách từ: Tách theo khoảng trắng (whitespace tokenization) kết hợp bảo toàn các thuật ngữ pháp lý và con số quan trọng (ví dụ: `điều 4`, `khoản 1`, `thông tư 02`).
- **Retrieval Output**:
  - `BM25Retriever.retrieve(query, top_k)` nhận vào câu hỏi và trả về danh sách ứng viên top $K$ (mặc định $K = 20$).
  - Mỗi phần tử trả về chứa thông tin chunk kèm điểm số BM25 score lũy tiến thực tế.

## 5. Semantic Candidate Contract
- **Vector Search Standard**:
  - Sử dụng Gemini Embedding API (`gemini-embedding-2`, dimension `768`) kết hợp ChromaDB `PersistentClient`.
  - Truy xuất top $K$ (mặc định $K = 20$) ứng viên có Cosine Distance nhỏ nhất từ Vector Collection tương ứng với chiến lược chunking được chọn.
- **Output Standard**:
  - Danh sách trả về bao gồm thông tin chunk, vị trí thứ hạng semantic rank ($1 \dots K$), và khoảng cách Cosine Distance.

## 6. RRF Fusion Contract
- **Reciprocal Rank Fusion (RRF) Formula**:
  $$\text{RRF Score}(d) = w_{\text{BM25}} \cdot \frac{1}{k + r_{\text{BM25}}(d)} + w_{\text{semantic}} \cdot \frac{1}{k + r_{\text{semantic}}(d)}$$
  Trong đó:
  - $k = 60$ (hằng số làm mịn RRF).
  - $r_{\text{BM25}}(d)$ và $r_{\text{semantic}}(d)$ lần lượt là thứ hạng (1-indexed) của tài liệu $d$ trong danh sách BM25 và Semantic. Nếu tài liệu không xuất hiện trong một danh sách, thứ hạng đó được coi là $\infty$ (điểm đóng góp bằng 0).
  - $w_{\text{BM25}} = 1.0$, $w_{\text{semantic}} = 1.0$ (trọng số kết hợp có thể điều chỉnh qua cấu hình).
- **Fusion Output**:
  - Sắp xếp tất cả tài liệu hợp nhất theo thứ tự điểm $\text{RRF Score}$ giảm dần, chọn ra danh sách candidate sau dung hợp.

## 7. Cross-Encoder Reranker Contract
- **Model Standard**:
  - Sử dụng pretrained model `BAAI/bge-reranker-v2-m3` từ thư viện `transformers` / `torch`.
- **Reranking Process**:
  - Nhận vào cặp `(query, candidate_text)` cho từng ứng viên từ kết quả RRF Fusion.
  - Tính toán logit/score trực tiếp thông qua Cross-Encoder.
  - Sắp xếp lại danh sách theo score Cross-Encoder giảm dần và cắt lấy Top $N$ final evidences (mặc định $N = 5$).

## 8. Final Evidence và Citation Contract
- **Confidence Gate**:
  - Lọc các ứng viên có điểm số sau Rerank hoặc Cosine Distance vượt quá ngưỡng an toàn (`RAG_MAX_DISTANCE`).
- **Citation Format**:
  - Mọi câu khẳng định trong câu trả lời sinh ra từ Gemini LLM bắt buộc phải đính kèm nhãn trích dẫn `[E1]`, `[E2]`,...
  - Mỗi nhãn `[E1]` phải được ánh xạ chính xác với danh sách nguồn trích dẫn: `Source`, `Trang X-Y`, `Chunk ID`.

## 9. Pipeline Trace Contract
Đối với mỗi câu hỏi, hệ thống Advanced RAG phải ghi lại vết thực thi chi tiết (Pipeline Trace) gồm:
1. `mode`: "semantic_only" | "hybrid"
2. `bm25_candidates`: Danh sách top 20 từ BM25 kèm thứ hạng và BM25 score.
3. `semantic_candidates`: Danh sách top 20 từ Semantic Retrieval kèm thứ hạng và distance.
4. `rrf_fused_results`: Danh sách sau dung hợp RRF kèm thứ hạng và RRF score.
5. `reranked_results`: Danh sách top 5 sau Cross-Encoder Rerank kèm Reranker score.
6. `final_evidence`: Danh sách evidence được đưa vào Prompt CONTEXT.
7. `total_latency_ms`: Thời gian xử lý từng giai đoạn và tổng thời gian (ms).

## 10. Evaluation Metrics Contract
Khung đánh giá Offline (`evaluate.py`) thực hiện đo lường trên tập câu hỏi `eval/questions.json`:
- **Hit Rate@K**: Tỷ lệ % câu hỏi mà trong Top $K$ kết quả truy xuất chứa ít nhất 1 `relevant_chunk_id` trong Gold Label.
- **Mean Reciprocal Rank (MRR@K)**:
  $$\text{MRR}@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  Trong đó $\text{rank}_i$ là thứ hạng đầu tiên của chunk đúng trong kết quả truy xuất của câu hỏi $i$.
- **Latency (ms)**: Thời gian truy xuất trung bình (P50, P95) giữa 2 chế độ `semantic_only` và `hybrid`.

## 11. Offline Testing Contract
- Tất cả unit test và integration test phải chạy hoàn toàn offline (sử dụng mock cho Gemini API và test fixture `chunks_advanced_sample.json`).
- Kiểm thử đầy đủ các trường hợp biên:
  - Từ khóa không khớp trong BM25.
  - Câu hỏi ngoài phạm vi (out-of-scope).
  - Tệp JSON chunks lỗi cấu trúc hoặc rỗng.

## 12. UI Comparison Contract
Giao diện Streamlit (`app.py`) cung cấp bảng điều khiển tương tác song song:
- Cột bên trái: Kết quả truy xuất & câu trả lời của **Semantic Baseline (Buổi 07)**.
- Cột bên phải: Kết quả truy xuất & câu trả lời của **Advanced Hybrid RAG (Buổi 08)**.
- Bảng so sánh trực quan thứ hạng, điểm số RRF, điểm Reranker, chỉ số Latency (ms) và danh sách Citation.
