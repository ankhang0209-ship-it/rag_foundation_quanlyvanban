"""
ỨNG DỤNG STREAMLIT TRỰC QUAN HOÁ RAG OCR & CHUNKING (BUỔI 5)
File: RAG/rag_foundation/buoi_05/app.py
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Foundation - Buổi 5: OCR & Chunking Visualizer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Đường dẫn thư mục output
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output"

# CSS Tùy chỉnh giao diện Premium Dark/Light UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #78909C;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .chunk-box {
        background-color: #ffffff;
        border-left: 5px solid #1E88E5;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-fixed { background-color: #E3F2FD; color: #1565C0; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-semantic { background-color: #E8F5E9; color: #2E7D32; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-hierarchical { background-color: #FFF3E0; color: #E65100; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

def load_json_chunks(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp {json_path.name}: {str(e)}")
        return None

def load_raw_text(raw_path: Path):
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Không tìm thấy nội dung văn bản thô."

def main():
    st.markdown('<div class="main-header">🧩 RAG Foundation - Visualizer Buổi 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trực quan hóa luồng chuyển đổi từ PDF → OCR/Unicode NFC → Chunking (Fixed-Size, Semantic, Hierarchical)</div>', unsafe_allow_html=True)

    if not OUTPUT_DIR.exists():
        st.warning(f"⚠️ Thư mục output không tồn tại: `{OUTPUT_DIR}`. Hãy chạy `main_pipeline.py --write` trước!")
        st.stop()

    chunk_files = list(OUTPUT_DIR.glob("*_chunks.json"))
    if not chunk_files:
        st.info("ℹ️ Chưa có tệp kết quả JSON nào trong thư mục `output/`. Vui lòng thực thi pipeline `--write` để tạo dữ liệu.")
        st.stop()

    # Sidebar Controls
    st.sidebar.header("⚙️ Cấu hình hiển thị")
    file_map = {f.name.replace("_chunks.json", ".pdf"): f for f in chunk_files}
    selected_doc_name = st.sidebar.selectbox("📁 Chọn tài liệu PDF:", list(file_map.keys()))
    selected_file = file_map[selected_doc_name]

    # Load data
    doc_data = load_json_chunks(selected_file)
    if not doc_data:
        st.stop()

    raw_file_name = selected_file.name.replace("_chunks.json", "_raw.txt")
    raw_text = load_raw_text(OUTPUT_DIR / raw_file_name)

    # Filtering options
    strategy_option = st.sidebar.radio(
        "🎯 Chiến lược Chunking:",
        ["Tất cả", "Fixed-size", "Semantic", "Hierarchical"]
    )
    
    search_term = st.sidebar.text_input("🔍 Tìm kiếm từ khóa trong Chunks:", "")

    # Aggregate Chunks
    fixed_list = doc_data.get("fixed_size_chunks", [])
    semantic_list = doc_data.get("semantic_chunks", [])
    hierarchical_list = doc_data.get("hierarchical_chunks", [])
    all_chunks = fixed_list + semantic_list + hierarchical_list

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tài liệu đang xem", selected_doc_name)
    with col2:
        ocr_status = "Bật (LlamaParse)" if doc_data.get("ocr_used") else "Tắt (PyMuPDF Text Layer)"
        st.metric("Trạng thái OCR", ocr_status)
    with col3:
        st.metric("Tổng số Chunks", len(all_chunks))
    with col4:
        st.metric("Độ dài Văn bản thô", f"{len(raw_text):,} ký tự")

    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🧩 Visual Chunk Explorer", "📄 Văn bản thô (Unicode NFC)", "📊 So sánh 3 Chiến lược"])

    with tab1:
        st.subheader("Danh sách Chunks")
        
        # Filter chunks
        display_chunks = []
        if strategy_option == "Tất cả":
            display_chunks = all_chunks
        elif strategy_option == "Fixed-size":
            display_chunks = fixed_list
        elif strategy_option == "Semantic":
            display_chunks = semantic_list
        elif strategy_option == "Hierarchical":
            display_chunks = hierarchical_list

        if search_term:
            display_chunks = [c for c in display_chunks if search_term.lower() in c.get("text", "").lower()]

        st.caption(f"Hiển thị **{len(display_chunks)}** chunks tương ứng với bộ lọc.")

        for item in display_chunks:
            strat = item.get("strategy", "")
            c_id = item.get("chunk_id", "")
            text_content = item.get("text", "")
            meta = item.get("metadata", {})

            badge_class = "badge-fixed"
            if strat == "semantic":
                badge_class = "badge-semantic"
            elif strat == "hierarchical":
                badge_class = "badge-hierarchical"

            with st.container():
                st.markdown(f"""
                <div class="chunk-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #37474F;">🆔 {c_id}</span>
                        <span class="{badge_class}">{strat.upper()}</span>
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; color: #263238; background: #fafafa; padding: 10px; border-radius: 4px;">{text_content}</div>
                    <div style="font-size: 0.8rem; color: #78909C; margin-top: 6px;">
                        📏 Độ dài: {len(text_content)} ký tự | Trang: {item.get('page_start', 1)} - {item.get('page_end', 1)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Xem Metadata chi tiết"):
                    st.json(meta)

    with tab2:
        st.subheader("Nội dung văn bản thô sau trích xuất & chuẩn hóa Unicode NFC")
        st.text_area("Raw Text Extracted:", value=raw_text, height=500)

    with tab3:
        st.subheader("Bảng so sánh chỉ số giữa 3 chiến lược Chunking")
        
        def calc_stats_df(c_list, name):
            if not c_list:
                return {"Chiến lược": name, "Số chunk": 0, "Độ dài Min": 0, "Độ dài Max": 0, "Độ dài Trung bình": 0}
            lens = [len(c["text"]) for c in c_list]
            return {
                "Chiến lược": name,
                "Số chunk": len(c_list),
                "Độ dài Min": min(lens),
                "Độ dài Max": max(lens),
                "Độ dài Trung bình": round(sum(lens)/len(lens), 1)
            }

        stats_data = [
            calc_stats_df(fixed_list, "Fixed-size"),
            calc_stats_df(semantic_list, "Semantic"),
            calc_stats_df(hierarchical_list, "Hierarchical")
        ]
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True)

        st.bar_chart(df_stats.set_index("Chiến lược")[["Số chunk", "Độ dài Trung bình"]])

if __name__ == "__main__":
    main()
