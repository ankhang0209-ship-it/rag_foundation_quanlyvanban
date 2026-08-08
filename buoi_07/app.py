"""
Streamlit Web App - Buổi 07 RAG Demo.
Giao diện hỏi đáp RAG Pipeline nâng cao với AI Agent.
"""

from pathlib import Path
import sys
import chromadb
import streamlit as st

# Thêm đường dẫn để import rag.py
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag

# -----------------------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Pipeline - Buổi 07",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RAG Pipeline Demo - Buổi 07")
st.caption("Hệ thống RAG hoàn thiện với Gemini Embedding, ChromaDB Persistent Index, Confidence Gate & Citation Mapping.")

# -----------------------------------------------------------------------------
# KHỞI TẠO CẤU HÌNH BẰNG RAG.PY
# -----------------------------------------------------------------------------
try:
    config = rag.load_config()
except Exception as e:
    st.error(f"❌ Lỗi tải cấu hình .env: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR: CẤU HÌNH VÀ TRẠNG THÁI HỆ THỐNG
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình & Trạng thái")

# 1. Hiển thị thông tin API & Models
api_key_status = "Có" if config["api_key"] else "Thiếu"
st.sidebar.markdown(f"**GEMINI_API_KEY:** `{'✅ ' + api_key_status if config['api_key'] else '❌ ' + api_key_status}`")
st.sidebar.markdown(f"**Embedding Model:** `{config['embedding_model']}` ({config['embedding_dim']}d)")
st.sidebar.markdown(f"**Generation Model:** `{config['generation_model']}`")
st.sidebar.markdown(f"**Ngưỡng Max Distance:** `{config['max_distance']}`")

st.sidebar.divider()

# 2. Selectbox chọn Strategy & Top-K
strategy = st.sidebar.selectbox(
    "Chiến lược Chunking (Strategy):",
    options=["hierarchical", "semantic", "fixed-size"],
    index=0,
)

top_k = st.sidebar.slider(
    "Số lượng nguồn truy xuất (Top-K):",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

st.sidebar.divider()

# 3. Đọc trạng thái Collection tương ứng với Strategy (Read-only)
col_name = rag.get_collection_name(strategy, config["embedding_dim"], config["embedding_model"])
col_exists = False
col_count = 0

if rag.CHROMA_DIR.exists():
    try:
        client = chromadb.PersistentClient(path=str(rag.CHROMA_DIR))
        collections = client.list_collections()
        for c in collections:
            c_name = getattr(c, "name", str(c))
            if c_name == col_name:
                col_exists = True
                try:
                    col = client.get_collection(name=col_name, embedding_function=None)
                    col_count = col.count()
                except Exception:
                    pass
                break
    except Exception:
        pass

st.sidebar.subheader("📦 Collection Index")
st.sidebar.markdown(f"**Tên Collection:** `{col_name}`")
st.sidebar.markdown(f"**Trạng thái:** `{'✅ Đã tạo' if col_exists else '❌ Chưa tạo'}`")
st.sidebar.markdown(f"**Số lượng Record:** `{col_count}` chunks")

# -----------------------------------------------------------------------------
# MAIN CONTENT: CHIẾN LƯỢC BAN ĐẦU & INDEX DATA AREA
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Index dữ liệu vào Vector Store")
    reset_index = st.checkbox("Reset/Xóa collection cũ trước khi index", value=False)
    
    if st.button("🚀 Index dữ liệu", type="primary", use_container_width=True):
        if not config["api_key"]:
            st.error("❌ Thiếu GEMINI_API_KEY trong file .env. Vui lòng cấu hình API key trước khi index dữ liệu!")
        else:
            with st.spinner(f"Đang đọc dữ liệu, tạo embeddings và lưu vào collection '{col_name}'..."):
                try:
                    res_index = rag.index_chunks(strategy=strategy, reset=reset_index)
                    st.session_state["index_result"] = res_index
                    st.success(f"✅ Index thành công {res_index['indexed_chunks']} chunks vào collection '{res_index['collection_name']}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Lỗi index: {e}")

    if "index_result" in st.session_state:
        res = st.session_state["index_result"]
        st.info(f"Kết quả index gần nhất: **{res['indexed_chunks']}** chunks | Tổng records: **{res['total_records']}**")

with col_right:
    st.subheader("2. Hướng dẫn nhanh")
    st.markdown("""
    1. Select chiến lược chunking (`hierarchical`, `semantic`, `fixed-size`) ở menu bên trái.
    2. Nhấn nút **Index dữ liệu** nếu collection chưa được tạo hoặc cần index lại.
    3. Nhập câu hỏi nghiệp vụ và nhấn **Gửi câu hỏi** để nhận câu trả lời có kèm nguồn trích dẫn kiểm chứng.
    """)

st.divider()

# -----------------------------------------------------------------------------
# HỎI ĐÁP RAG PIPELINE
# -----------------------------------------------------------------------------
st.subheader("3. Hỏi đáp RAG với AI Agent")

question_input = st.text_area(
    "Nhập câu hỏi của bạn:",
    height=100,
    placeholder="Ví dụ: Quy định về phạm vi điều chỉnh và nguyên tắc cho vay thế nào?",
)

btn_ask = st.button("💬 Gửi câu hỏi", type="primary")

if btn_ask:
    clean_q = question_input.strip()
    if not clean_q:
        st.warning("⚠️ Vui lòng nhập nội dung câu hỏi trước khi gửi.")
    elif not config["api_key"]:
        st.error("❌ Thiếu GEMINI_API_KEY trong file .env. Không thể gửi câu hỏi.")
    elif not col_exists or col_count == 0:
        st.error("❌ Collection dữ liệu chưa tồn tại hoặc đang rỗng. Vui lòng thực hiện 'Index dữ liệu' ở bước 1 trước!")
    else:
        with st.spinner("Đang truy xuất nguồn tài liệu và tạo câu trả lời..."):
            try:
                res_query = rag.query_rag(question=clean_q, top_k=top_k, strategy=strategy)
                st.session_state["query_result"] = res_query
            except Exception as e:
                st.error(f"❌ Lỗi xử lý truy vấn: {e}")

# -----------------------------------------------------------------------------
# HIỂN THỊ KẾT QUẢ TRUY VẤN
# -----------------------------------------------------------------------------
if "query_result" in st.session_state:
    res = st.session_state["query_result"]
    status = res.get("status")

    st.markdown("### 💬 Kết quả trả lời")

    # Status Banner
    if status == "answered":
        st.success("✅ **Đã tổng hợp câu trả lời có đầy đủ căn cứ trích dẫn.**")
        st.markdown(res["answer"])
    elif status == "insufficient_evidence":
        st.warning("⚠️ **Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp.**")
        st.info(res["answer"])
    elif status == "retrieval_only":
        st.error("🚨 **Đã truy xuất được nguồn thông tin nhưng quá trình tổng hợp câu trả lời gặp lỗi.**")
        st.caption(res["answer"])
    else:
        st.write(res.get("answer", ""))

    # Hiển thị Warnings nếu có
    if res.get("warnings"):
        for warn in res["warnings"]:
            st.warning(f"⚠️ {warn}")

    # Hiển thị Citations
    if res.get("citations"):
        st.markdown("#### 📜 Danh sách trích dẫn (Citations):")
        for cite in res["citations"]:
            st.markdown(f"- **{cite['evidence_id']}**: `{cite['display']}`")

    st.divider()

    # -------------------------------------------------------------------------
    # HIỂN THỊ DANH SÁCH EVIDENCE (NGUỒN THAM KHẢO)
    # -------------------------------------------------------------------------
    st.markdown("### 📌 Nguồn tham khảo (Retrieved Evidences)")
    st.caption("Khoảng cách Cosine (Distance) biểu thị độ bất tương đồng: khoảng cách càng nhỏ thì đoạn văn bản càng liên quan. Ngưỡng chấp nhận: `RAG_MAX_DISTANCE <= 0.45`.")

    evidences = res.get("evidence", [])
    if not evidences:
        st.info("Chưa có evidence nào được truy xuất.")
    else:
        for ev in evidences:
            is_acc = ev["accepted"]
            acc_badge = "✅ (Đạt Threshold)" if is_acc else "❌ (Bị loại bởi Gate)"
            page_str = f"tr. {ev['page_start']}" if ev['page_start'] == ev['page_end'] else f"tr. {ev['page_start']}-{ev['page_end']}"

            title_summary = f"[{ev['evidence_id']}] {acc_badge} | {ev['source']} – {page_str} – Distance: {ev['distance']:.4f}"

            with st.expander(title_summary, expanded=is_acc):
                st.markdown(f"**ID:** `{ev['chunk_id']}`")
                st.markdown(f"**Nguồn:** `{ev['source']}` ({page_str})")
                st.markdown(f"**Distance:** `{ev['distance']:.4f}` ({'Đạt ngưỡng safe threshold' if is_acc else 'Vượt ngưỡng safe threshold - Không đưa vào Prompt'})")
                st.markdown("**Nội dung đoạn văn:**")
                st.text(ev["text"])
