import os
import sys
import streamlit as st

# Import module RAG Buổi 06
import rag

# ----------------------------------------------------
# 1. Cấu hình Trang Streamlit
# ----------------------------------------------------
st.set_page_config(
    page_title="RAG Foundation - Buổi 06 Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại & đẹp mắt
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #546E7A;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-badge-ok {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }
    .status-badge-warn {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. SIDEBAR: Trạng thái Hệ thống & Môi trường
# ----------------------------------------------------
st.sidebar.title("⚙️ Môi trường & Hệ thống")

# 1. Trạng thái Gemini API Key (Có / Thiếu)
api_key = rag.GEMINI_API_KEY
st.sidebar.markdown("**Gemini API Key:**")
if api_key:
    st.sidebar.markdown('<div class="status-badge-ok">🟢 Có (Đã cấu hình)</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge-warn">🔴 Thiếu (Chỉ Retrieval)</div>', unsafe_allow_html=True)

st.sidebar.divider()

# 2. Trạng thái PostgreSQL
st.sidebar.markdown("**PostgreSQL:**")
try:
    _, db_type = rag.get_db_connection()
    if db_type == "postgres":
        st.sidebar.markdown('<div class="status-badge-ok">🟢 Đã kết nối (rag_db)</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-badge-warn">🟠 Dùng SQLite Local (.db)</div>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown(f'<div class="status-badge-warn">⚠️ Lỗi DB: {str(e)}</div>', unsafe_allow_html=True)

st.sidebar.divider()

# 3. Trạng thái ChromaDB
st.sidebar.markdown("**ChromaDB:**")
st.sidebar.markdown('<div class="status-badge-ok">🟢 Embedded Local (storage/chroma)</div>', unsafe_allow_html=True)

st.sidebar.divider()

# Thống kê số lượng hiện có
stats = rag.status()
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Số Document", stats.get("documents", 0))
col_s2.metric("Số Chunks", stats.get("chunks", 0))

# ----------------------------------------------------
# 3. MAIN AREA: Indexing ➔ Question ➔ Top-k ➔ Gemini ➔ Answer
# ----------------------------------------------------
st.markdown('<div class="main-title">🤖 RAG Foundation - Buổi 06 Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pipeline: <code>Question</code> ➔ <code>Top-k Vector Search</code> ➔ <code>Gemini LLM</code> ➔ <code>Answer</code></div>', unsafe_allow_html=True)

# 1. Nút Index
with st.expander("📥 1. Khởi tạo & Đánh chỉ mục Dữ liệu (Index)", expanded=(stats.get("chunks", 0) == 0)):
    st.write("Đọc các tệp JSON chunk trong `buoi_05/output/chunks/`, tạo Vector Embeddings (384-dim) và lưu trữ vào DB & ChromaDB.")
    if st.button("🚀 Chạy Index", type="secondary"):
        with st.spinner("Đang nạp dữ liệu và đánh chỉ mục vector..."):
            res = rag.index()
            if res.get("status") == "success":
                st.success(f"✅ Index thành công **{res.get('indexed_chunks')} chunks** từ {len(res.get('processed_files', []))} tệp JSON!")
                st.rerun()
            else:
                st.error(f"❌ Lỗi: {res.get('message')}")

st.divider()

# 2. Ô nhập câu hỏi & Chọn Top-k
st.subheader("🔍 2. Hỏi đáp Dữ liệu & Trích xuất Top-k")

with st.form("qa_form"):
    col_input, col_k = st.columns([4, 1])
    with col_input:
        question = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Quy định về cơ cấu lại thời hạn trả nợ?")
    with col_k:
        top_k = st.slider("Top-k:", min_value=1, max_value=10, value=5)

    btn_ask = st.form_submit_button("💬 Gửi câu hỏi", type="primary")

# 3. Pipeline xử lý & Hiển thị Kết quả
if btn_ask and question.strip():
    with st.spinner("Đang tìm kiếm Vector & Gửi Gemini sinh câu trả lời..."):
        result = rag.ask(question.strip(), top_k=top_k)
        
        answer = result.get("answer", "")
        retrieved_chunks = result.get("retrieved_chunks", [])

        # Hiển thị Câu trả lời (Answer)
        st.markdown("### 📝 Answer (Câu trả lời):")
        st.info(answer)

        st.divider()

        # Hiển thị Kết quả Top-k Chunks (Context)
        st.markdown(f"### 📚 Kết quả Top-{len(retrieved_chunks)} Chunks Trích xuất:")
        if not retrieved_chunks:
            st.warning("Không tìm thấy chunk nào phù hợp.")
        else:
            for idx, item in enumerate(retrieved_chunks):
                cid = item.get("chunk_id", "N/A")
                source = item.get("source", "N/A")
                text = item.get("text", "")
                
                with st.expander(f"📌 Chunk [{idx+1}] - File: {source} | ID: {cid}"):
                    st.markdown(f"```text\n{text}\n```")
