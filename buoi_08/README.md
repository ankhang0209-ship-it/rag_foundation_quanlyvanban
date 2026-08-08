# HƯỚNG DẪN VÀ THUYẾT MINH KIẾN TRÚC ADVANCED HYBRID RAG ENGINE — BUỔI 08

Hệ thống **Advanced Hybrid RAG Engine (Buổi 08)** kết hợp đa nhánh truy xuất từ vựng (BM25 Lexical Keyword Search), tìm kiếm ngữ nghĩa vector (Gemini Embeddings + ChromaDB), thuật toán hợp nhất thứ hạng Reciprocal Rank Fusion (RRF) và mô hình xếp hạng lại Cross-Encoder (`BAAI/bge-reranker-v2-m3`).

---

## 1. Mục tiêu và Khác biệt giữa Buổi 07 và Buổi 08

| Tiêu chí | Buổi 07 (Semantic Baseline) | Buổi 08 (Advanced Hybrid RAG) |
|---|---|---|
| **Kiến trúc Truy xuất** | Nhánh đơn (Single-branch Vector Search) | Đa nhánh (Multi-branch: Lexical BM25 + Vector Semantic) |
| **Bảo toàn Từ khóa Pháp lý** | Thấp (Dễ bỏ sót con số Điều/Khoản, số hiệu Thông tư) | Rất cao (BM25Okapi với Tokenizer Tiếng Việt chuyên biệt) |
| **Hợp nhất Kết quả (Fusion)** | Không có | Reciprocal Rank Fusion (RRF, $k=60$) |
| **Tinh chỉnh Thứ hạng** | Không có | Cross-Encoder Reranking (`BAAI/bge-reranker-v2-m3`) |
| **Chuyển dịch Thứ hạng** | Cố định theo Cosine Distance | Có `rank_change` thể hiện sự thay đổi vị trí trước/sau Rerank |
| **Trích dẫn Nguồn (Citation)** | Không bóc tách nhãn | Tự động bóc tách `[E1]`, `[E2]`, loại bỏ nhãn giả và map metadata |
| **Đánh giá Offline** | Đơn giản | Đo lường chi tiết Hit Rate@K, Recall@K, MRR@K, nDCG@K, Latency P50 |

---

## 2. Sơ đồ Kiến trúc Pipeline

```text
                               ┌────────────────────────────────┐
                               │     User Query (Câu hỏi)       │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │    BM25 Lexical Search      │                 │   Gemini Vector Retrieval   │
        │   (Tokenize vi_legal)       │                 │  (gemini-embedding-2 768d)  │
        └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                       │ Top 20 candidates                             │ Top 20 candidates
                       └───────────────────────┬───────────────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │  Reciprocal Rank Fusion (RRF)  │
                               │   RRF(d) = 1/(k+r1) + 1/(k+r2) │
                               └───────────────┬────────────────┘
                                               │ Top 20 fused candidates
                                               ▼
                               ┌────────────────────────────────┐
                               │     Cross-Encoder Reranker     │
                               │   (BAAI/bge-reranker-v2-m3)    │
                               └───────────────┬────────────────┘
                                               │ Sigmoid Score & Re-order
                                               ▼
                               ┌────────────────────────────────┐
                               │    Gating & Confidence Filter   │
                               │   (rerank_score >= 0.50 / Max) │
                               └───────────────┬────────────────┘
                                               │ Top N Accepted Evidences
                                               ▼
                               ┌────────────────────────────────┐
                               │     Gemini LLM Generation      │
                               │ (Grounding Context + Citations)│
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Final Answer + Citations [E]  │
                               └────────────────────────────────┘
```

---

## 3. Cấu trúc Dự án Buổi 08

```text
rag_foundation/buoi_08/
├── .env.example                # Khai báo cấu hình mẫu (không chứa secret)
├── .gitignore                  # Giới hạn không commit venv, storage, reports, secrets
├── README.md                   # Tài liệu hướng dẫn toàn diện
├── SPEC_buoi_08.md             # Hợp đồng thông số kỹ thuật chi tiết
├── advanced_rag.py             # Engine core: BM25, RRF, Cross-Encoder Reranker, Query, Compare
├── app.py                      # Streamlit Comparison Dashboard (4 Tabs)
├── evaluate.py                 # Framework Đánh giá Offline (Recall@K, MRR@K, nDCG@K, Latency)
├── rag.py                      # Semantic Baseline module kế thừa Buổi 07
├── requirements.txt            # Thư viện phụ thuộc
├── eval/
│   └── questions.json          # Tập câu hỏi đánh giá và Gold Labels
├── reports/
│   └── evaluation_report.json  # Báo cáo JSON kết quả đánh giá offline
├── storage/
│   ├── chroma/                 # Lưu trữ Vector Index ChromaDB
│   └── huggingface/            # Cache weights Reranker Model
└── tests/                      # Tập unit tests offline (100% mock boundary)
    ├── test_answer_pipeline.py # Tests cho Answer pipeline, gating & citations
    ├── test_bm25.py            # Tests cho BM25 tokenizer & search
    ├── test_evaluator.py       # Tests cho công thức metrics (Recall, MRR, nDCG)
    ├── test_reranker.py        # Tests cho Cross-Encoder reranker & lazy loading
    ├── test_rrf.py             # Tests cho RRF fusion formula & weights
    └── test_semantic.py        # Tests cho Semantic vector candidate search
```

---

## 4. Thiết lập Môi trường và Cấu hình

Sử dụng chung Python Virtual Environment tại `rag_foundation/buoi_05/.venv/`:

```bash
# 1. Kích hoạt venv (Windows PowerShell)
..\buoi_05\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Tạo tệp cấu hình .env từ mẫu
cp .env.example .env
```

Nội dung tệp `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=768
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
RAG_MAX_DISTANCE=0.45
BM25_CANDIDATES=20
SEMANTIC_CANDIDATES=20
RRF_K=60
RRF_BM25_WEIGHT=1.0
RRF_SEMANTIC_WEIGHT=1.0
RERANK_CANDIDATES=20
FINAL_TOP_K=5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_MAX_LENGTH=512
RERANK_BATCH_SIZE=4
RERANK_MIN_SCORE=0.50
RERANK_DEVICE=auto
```

---

## 5. Cảnh báo Kích thước Mô hình và Tài nguyên Reranker

> [!WARNING]
> **Yêu cầu Tài nguyên phần cứng cho Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`):**
> - **Dung lượng Weights**: Khoảng **~2.2 GB**. Khi chạy lệnh Rerank lần đầu tiên, weights sẽ tự động tải về thư mục cache `storage/huggingface/`.
> - **Bộ nhớ RAM**: Khuyên dùng tối thiểu **4 GB RAM trống** để khởi tạo model mượt mà.
> - **Thiết bị tính toán (Device)**:
>   - `RERANK_DEVICE=auto`: Tự động phát hiện GPU CUDA nếu có, ngược lại dùng CPU.
>   - `RERANK_DEVICE=cpu`: Ép buộc sử dụng CPU (an toàn cho máy không có GPU).

---

## 6. Danh sách Câu lệnh CLI

```bash
# 1. Kiểm tra trạng thái hệ thống (Read-only)
python advanced_rag.py status --strategy hierarchical

# 2. Khởi tạo/Index dữ liệu Semantic Retrieval vào ChromaDB
python advanced_rag.py prepare-semantic --strategy hierarchical

# 3. Truy xuất từ khóa BM25 (Lexical)
python advanced_rag.py bm25 --strategy hierarchical --question "Điều 7 quy định gì?"

# 4. Truy xuất vector Semantic
python advanced_rag.py semantic --strategy hierarchical --question "Điều 7 quy định gì?"

# 5. Truy xuất dung hợp Hybrid RRF
python advanced_rag.py hybrid --strategy hierarchical --question "Điều 7 quy định gì?"

# 6. Truy xuất Cross-Encoder Reranking
python advanced_rag.py rerank --strategy hierarchical --question "Điều 7 quy định gì?"

# 7. Truy vấn Advanced RAG Answer đầy đủ (Gọi generation tối đa 1 lần)
python advanced_rag.py query --mode hybrid_rerank --strategy hierarchical --question "Điều 7 quy định gì?"

# 8. So sánh 4 chế độ retrieval (0 calls generation)
python advanced_rag.py compare --strategy hierarchical --question "Điều 7 quy định gì?"

# 9. Chạy đánh giá Offline Engine
python evaluate.py --strategy hierarchical --k 5
```

---

## 7. Lệnh Kiểm thử và Khởi chạy Dashboard

```bash
# Chạy toàn bộ 44 Unit Tests Offline (100% Mock, 0 Network Calls)
python -m unittest discover tests

# Khởi chạy giao diện Streamlit Dashboard (4 Tabs)
python -m streamlit run app.py
```

---

## 8. Giải thích Chi tiết các Chỉ số Score và Distance

1. **BM25 Score**: Điểm số tần suất xuất hiện từ khóa khớp chính xác trong câu hỏi. *Càng cao càng tốt*.
2. **Cosine Distance**: Khoảng cách vector giữa embedding câu hỏi và chunk. *Càng thấp càng tốt* ($0.0$ là trùng khớp ngữ nghĩa tuyệt đối).
3. **RRF Score**: Điểm dung hợp Reciprocal Rank Fusion: $\text{RRF}(d) = \sum \frac{w}{k + r(d)}$. *Càng cao càng tốt*.
4. **Rerank Score (Sigmoid)**: Raw logit từ Cross-Encoder sau khi đưa qua hàm Sigmoid $\frac{1}{1 + e^{-logit}}$ về dải $[0, 1]$. *Càng cao càng tốt* (ngưỡng chấp nhận $\ge 0.50$).
   - ⚠️ **Lưu ý**: Rerank score thể hiện mức độ tương quan ngữ nghĩa trực tiếp, **KHÔNG PHẢI xác suất toán học**.

---

## 9. Phân biệt Candidate K và Final Top-K

- **BM25 / Semantic Candidates ($K = 20$)**: Số lượng ứng viên thô được lấy ra ở từng nhánh riêng lẻ.
- **Rerank Candidates ($K = 20$)**: Số lượng ứng viên từ kết quả RRF Fusion đưa vào mô hình Cross-Encoder để tính toán logit.
- **Final Top-K ($N = 5$)**: Số lượng bằng chứng tốt nhất cuối cùng (đã qua Gating Filter) được đưa vào Prompt Context cho LLM sinh câu trả lời.

---

## 10. Chỉ số Đánh giá Offline và Giới hạn Gold Labels

- **Recall@K**: Tỷ lệ phần trăm các relevant chunks tìm thấy trong Top K.
- **MRR@K (Mean Reciprocal Rank)**: Điểm vị trí xuất hiện đầu tiên của chunk đúng ($1/\text{rank}$).
- **nDCG@K**: Điểm tích lũy giảm dần có chuẩn hóa đối với độ liên quan nhị phân.
- **Latency (mean & p50)**: Thời gian truy xuất trung bình và vị trí trung vị P50 (ms).

> [!CAUTION]
> **Giới hạn Gold Labels (`needs_human_review = True`):**
> Trong tập dữ liệu câu hỏi đánh giá `eval/questions.json`, một số câu hỏi vẫn đang gắn nhãn `needs_human_review = True`. Báo cáo đánh giá JSON sẽ tự động hiển thị cảnh báo và **không tuyên bố mode nào thắng cuộc chính thức** cho đến khi dữ liệu được kiểm duyệt thủ công hoàn toàn.

---

## 11. Xử lý Sự cố (Troubleshooting)

- **Lỗi không kết nối được Hugging Face khi nạp Reranker**: Kiểm tra đường truyền internet hoặc cài đặt proxy. Nếu máy offline, hãy copy thư mục cache model đã tải vào `storage/huggingface/`.
- **CPU xử lý Rerank chậm**: Giảm `RERANK_CANDIDATES=10` hoặc `RERANK_BATCH_SIZE=2` trong tệp `.env`.
- **Lỗi thiếu bộ nhớ RAM**: Giảm số lượng candidate $K$ và đảm bảo đóng các ứng dụng nặng khác.
- **Lỗi `collection_not_found`**: Chạy lệnh `python advanced_rag.py prepare-semantic --strategy hierarchical` để khởi tạo vector collection ChromaDB.

---

## 12. Bảng So sánh 4 Câu hỏi Thử nghiệm Thực tế (Manual Comparison)

Thực hiện chạy so sánh trực tiếp trên 4 dạng câu hỏi đặc trưng:

| Nhóm Câu hỏi | Nội dung Câu hỏi | Kết quả Thứ hạng & Nhận xét Trực quan |
|---|---|---|
| **A. Exact Legal Reference** | *Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?* | BM25 và Hybrid đưa chunk chứa "Điều 7" lên đầu (#1). Reranker khẳng định điểm số $0.92$ cho thấy độ tương quan tuyệt đối với thuật ngữ pháp lý chính xác. |
| **B. Paraphrase Semantic** | *Khách hàng gặp khó khăn có thể được điều chỉnh kỳ hạn trả nợ ra sao?* | BM25 điểm thấp do không có từ khóa "Điều 7". Semantic và Hybrid Rerank kéo chunk quy định hoãn/gia hạn nợ lên vị trí Top 1 với rank movement $+3$. |
| **C. Multi-concept** | *Phân loại nợ và trích lập dự phòng được thực hiện như thế nào?* | Kết hợp thông tin từ Thông tư 02 và Thông tư 11. Branch Hybrid RRF phủ rộng cả 2 khái niệm (Union count cao), Reranker xếp hạng chính xác các chunk tổng hợp. |
| **D. Out-of-scope** | *Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?* | Cả 4 mode đều cho điểm thấp. Rerank score $< 0.50$ và Cosine distance $> 0.45$. Gating filter loại bỏ 100% evidences $\rightarrow$ Trả về status `insufficient_evidence` an toàn. |

---

---

## 14. Lệnh Chạy Tham Khảo (Từ Thư mục Gốc RAG)

### Windows PowerShell

```powershell
# Status
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py status --strategy hierarchical

# Prepare semantic index
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py prepare-semantic --strategy hierarchical

# BM25
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py bm25 --strategy hierarchical --question "Điều 7 quy định gì?"

# Hybrid RRF
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py hybrid --strategy hierarchical --question "Điều 7 quy định gì?"

# Hybrid + rerank
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py rerank --strategy hierarchical --question "Điều 7 quy định gì?"

# So sánh retrieval
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py compare --strategy hierarchical --question "Điều 7 quy định gì?"

# Query Advanced RAG
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\advanced_rag.py query --mode hybrid_rerank --strategy hierarchical --question "Điều 7 quy định gì?"

# Test
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m unittest discover -s .\rag_foundation\buoi_08\tests -v

# Evaluation
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_08\evaluate.py --strategy hierarchical --k 5

# Streamlit
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m streamlit run .\rag_foundation\buoi_08\app.py
```

### Linux / macOS

Thay interpreter Windows bằng `./rag_foundation/buoi_05/.venv/bin/python` và giữ nguyên các tham số.

---

## 15. Kịch Bản So Sánh Trực Tiếp với Buổi 07

### Câu Mở Đầu Demo
> *"Buổi 07 dùng semantic retrieval để tìm các đoạn gần câu hỏi trong không gian vector. Buổi 08 bổ sung BM25 để bắt từ khóa pháp lý chính xác, dùng RRF hợp nhất hai danh sách và dùng cross-encoder đọc đồng thời câu hỏi với từng đoạn để sắp xếp lại candidate."*

### Demo Cases:
1. **Demo 1 — Exact reference:** `Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?` (Quan sát BM25 bắt đúng Điều 7, RRF đưa chunk giao thoa lên đầu, Reranker khẳng định thứ hạng #1).
2. **Demo 2 — Paraphrase:** `Khách hàng gặp khó khăn có thể được điều chỉnh kỳ hạn trả nợ ra sao?` (Quan sát Semantic bổ sung candidate mà lexical có thể bỏ sót).
3. **Demo 3 — Out-of-scope:** `Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?` (Kiểm tra Rerank gate chặn generation, trả ra `insufficient_evidence` an toàn).

### Câu Kết Demo
> *"Khác biệt của Buổi 08 không chỉ nằm ở việc có thêm một model. Giao diện cho phép nhìn toàn bộ hành trình của mỗi chunk: được BM25 tìm thấy ở hạng nào, được semantic tìm thấy ở hạng nào, RRF hợp nhất ra sao và reranker đưa lên hay đẩy xuống. Chất lượng được so sánh bằng metric và latency thực tế, không dựa trên cảm giác câu trả lời nghe hay hơn."*

