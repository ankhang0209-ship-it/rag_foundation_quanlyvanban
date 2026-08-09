"""
Ứng dụng Streamlit Multi-Query & Parent–Child Retrieval Explorer (Buổi 09).
Sử dụng 5 Tabs giao diện chuyên sâu hiển thị ma trận Query-Child và cây Parent-Child.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Thêm đường dẫn module Buổi 09
BUOI_09_DIR = Path(__file__).resolve().parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag
import advanced_rag

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .subtitle-pipeline {
        font-size: 1.05rem;
        font-weight: 500;
        color: #555555;
        background-color: #F0F4F8;
        padding: 8px 16px;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1.5rem;
    }
    .q-card-original {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .q-card-generated {
        background-color: #F5F5F5;
        border-left: 5px solid #78909C;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .parent-node-card {
        background-color: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .metric-badge {
        font-weight: 600;
        color: #2E7D32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & Subtitle Header
st.markdown('<div class="main-title">⚖️ RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-pipeline">🔗 Pipeline: Query fan-out ➔ Hybrid per query ➔ Cross-query RRF ➔ Parent expansion ➔ Parent rerank</div>', unsafe_allow_html=True)

# Nạp Cấu Hình Ban Đầu
config = hierarchical_rag.load_config()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Cấu Hình Pipeline")

# 1. Mode Selector
selected_mode = st.sidebar.selectbox(
    "Lựa chọn Pipeline Mode",
    options=["multi_parent", "single_parent", "multi_flat", "single_flat"],
    index=0,
    help="Chế độ thực thi tìm kiếm nâng cao Buổi 09",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Tham Số Tuning")

mq_count = st.sidebar.slider("MULTI_QUERY_COUNT", min_value=1, max_value=5, value=config["multi_query_count"], help="Số lượng query biến thể sinh thêm")
per_cand = st.sidebar.slider("PER_QUERY_CANDIDATES", min_value=1, max_value=50, value=config["per_query_candidates"], help="Số child candidates lấy mỗi query")
parent_cand = st.sidebar.slider("PARENT_CANDIDATES", min_value=1, max_value=20, value=config["parent_candidates"], help="Số parent candidates trước rerank")
final_top_k = st.sidebar.slider("FINAL_PARENT_TOP_K", min_value=1, max_value=10, value=config["final_parent_top_k"], help="Số context xuất ra câu trả lời")
rerank_min_score = st.sidebar.slider("RERANK_MIN_SCORE", min_value=0.0, max_value=1.0, value=float(config["rerank_min_score"]), step=0.05, help="Ngưỡng Evidence Gate")

# Update runtime config
config["multi_query_count"] = mq_count
config["per_query_candidates"] = per_cand
config["parent_candidates"] = parent_cand
config["final_parent_top_k"] = final_top_k
config["rerank_min_score"] = rerank_min_score

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Trạng Thái Hệ Thống")

# Strategy Badge
st.sidebar.info("🎯 **Strategy**: `hierarchical` (Cố định Buổi 09)")

# Check API Key
api_key = config.get("api_key")
if api_key:
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    st.sidebar.success(f"🔑 **Gemini API Key**: Đã cấu hình (`{masked_key}`)")
else:
    st.sidebar.warning("⚠️ **Gemini API Key**: Chưa cấu hình trong `.env`!")

# System Model Identities
st.sidebar.caption(f"🤖 **Gen Model**: `{config['generation_model']}`")
st.sidebar.caption(f"🎯 **Reranker**: `{config['reranker_model']}`")

# Hierarchy Store Status
st_info = hierarchical_rag.get_hierarchy_status()
if st_info["store_exists"]:
    st.sidebar.success(f"📦 **Hierarchy Store**: Sẵn sàng ({st_info['children_count']} children, {st_info['parents_count']} parents)")
else:
    st.sidebar.error("❌ **Hierarchy Store**: Chưa khởi tạo!")

# Chroma Collection Status
col_status = advanced_rag.get_status()
if col_status.get("collection_exists"):
    st.sidebar.success(f"💾 **Chroma Vector DB**: Sẵn sàng ({col_status.get('document_count')} chunks)")
else:
    st.sidebar.warning("⚠️ **Chroma Vector DB**: Chưa nạp vector collection!")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Thao Tác Quản Lý Store")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🔨 Build Hierarchy", help="Xây dựng lại Hierarchy Registry & Parent Store"):
        with st.spinner("Đang xây dựng Hierarchy Store..."):
            build_res = hierarchical_rag.build_hierarchy_store(config=config)
            st.sidebar.success(f"Đã build store thành công! ({build_res['parents_count']} parents)")
            st.rerun()

with col_btn2:
    if st.button("🔄 Reset Status", help="Làm mới trạng thái hệ thống"):
        st.rerun()

# Session State Initialization
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "compare_result" not in st.session_state:
    st.session_state["compare_result"] = None

# -----------------------------------------------------------------------------
# MAIN TABS INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Tab 1: Hỏi Đáp Advanced RAG",
    "🔍 Tab 2: Query Fan-out",
    "🌳 Tab 3: Parent–Child Explorer",
    "📊 Tab 4: Mode Comparison",
    "📈 Tab 5: Evaluation",
])

# -----------------------------------------------------------------------------
# TAB 1: ASK ADVANCED RAG
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 💬 Hỏi Đáp Pháp Luật Ngân Hàng với Multi-Query & Parent RAG")
    default_q = "Điều kiện vay vốn và các trường hợp nhu cầu vốn không được cho vay được quy định thế nào?"
    user_question = st.text_area("Nhập câu hỏi của bạn:", value=default_q, height=100)

    if st.button("🚀 Chạy RAG Pipeline", type="primary", use_container_width=True):
        with st.spinner(f"Đang xử lý câu hỏi với mode `{selected_mode}`..."):
            res = hierarchical_rag.execute_query_pipeline(
                question=user_question,
                mode=selected_mode,
                config=config,
            )
            st.session_state["last_result"] = res

    res = st.session_state["last_result"]

    if res is not None:
        st.markdown("---")
        
        # Performance Header Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Status", res["status"])
        col_m2.metric("Tổng Thời Gian", f"{res['latency_ms'].get('total_ms', 0)} ms")
        col_m3.metric("Generation Calls", res["api_call_counts"]["generation_calls"], help="Tối đa 2 calls trong multi mode")
        col_m4.metric("Embedding Calls", res["api_call_counts"]["embedding_calls"])

        # UI Alert Banner based on status
        alert_info = hierarchical_rag.map_status_to_ui_alert(res["status"])
        if alert_info["type"] == "error":
            st.error(f"**{alert_info['title']}**: {alert_info['action']}")
        elif alert_info["type"] == "warning":
            st.warning(f"**{alert_info['title']}**: {alert_info['action']}")

        # Answer Section
        st.markdown("### 📝 Câu Trả Lời Sinh Ra từ Bằng Chứng Chấp Nhận")
        st.markdown(res["answer"])

        # Warnings container
        if res.get("warnings"):
            with st.expander("⚠️ Danh Sách Cảnh Báo System Warnings", expanded=False):
                for w in res["warnings"]:
                    st.caption(f"• {w}")

        # Expandable Accepted Evidence & Citations
        if res.get("accepted_evidence"):
            st.markdown("### 📜 Chi Tiết Các Trích Dẫn Pháp Lý Accepted Evidence")
            for idx, ev in enumerate(res["accepted_evidence"], start=1):
                lbl = f"P{idx}"
                with st.expander(f" Trích dẫn [{lbl}] - Score: {ev.get('parent_rerank_score', ev.get('rerank_score', 0.0)):.4f} | ID: {ev.get('parent_id', ev.get('child_id'))}", expanded=(idx == 1)):
                    st.write(f"**Nguồn văn bản**: {ev.get('source')} (Trang {ev.get('page_start')}-{ev.get('page_end')})")
                    st.write(f"**Cơ cấu điều khoản**: {ev.get('structural_path')}")
                    st.write(f"**Supporting Children**: {ev.get('supporting_child_ids')}")
                    st.write(f"**Supporting Queries**: {ev.get('support_query_ids')}")
                    st.markdown("**Nội dung văn bản mở rộng**:")
                    st.info(ev.get("text"))

# -----------------------------------------------------------------------------
# TAB 2: QUERY FAN-OUT
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔍 Phân Tích Query Fan-out & Ma Trận Query–Child Retrieval")
    res = st.session_state["last_result"]

    if res is None:
        st.info("Vui lòng thực thi câu hỏi tại Tab 1 trước để xem chi tiết Query Fan-out.")
    else:
        q_set = res.get("query_set", [])
        st.markdown(f"#### 1. Danh Sách Câu Hỏi Mở Rộng ({len(q_set)} queries)")

        col_q_list = st.columns(min(len(q_set), 4) if q_set else 1)
        for idx, q_item in enumerate(q_set):
            col_target = col_q_list[idx % len(col_q_list)]
            with col_target:
                qid = q_item.get("query_id", f"Q{idx}")
                origin = q_item.get("origin", "original")
                focus = q_item.get("focus", "original_intent")
                q_text = q_item.get("text", "")

                if origin == "original" or qid == "Q0":
                    st.markdown(
                        f'<div class="q-card-original"><b>[{qid}] (Câu hỏi gốc)</b><br>{q_text}<br><small>Focus: {focus}</small></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="q-card-generated"><b>[{qid}] (Biến thể sinh thêm)</b><br>{q_text}<br><small>Focus: {focus}</small></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("#### 2. Ma Trận Query–Child Retrieval (Query-Child Rank Matrix)")
        st.caption("Bảng dưới đây hiển thị Rank truy xuất của từng Child Chunk nhỏ trong từng query biến thể:")

        child_hits = res.get("child_hits", [])
        if child_hits:
            matrix_data = hierarchical_rag.build_query_child_matrix(child_hits, q_set)
            df_matrix = pd.DataFrame(matrix_data)
            st.dataframe(df_matrix, use_container_width=True)
        else:
            st.warning("Chưa có dữ liệu Child Hits trong kết quả thực thi hiện tại.")

# -----------------------------------------------------------------------------
# TAB 3: PARENT–CHILD EXPLORER
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🌳 Parent–Child Document Hierarchy Explorer")
    st.caption("Cây biểu diễn sự mở rộng từ các Child Chunks tìm kiếm được sang Parent Document có ngữ cảnh pháp lý đầy đủ:")

    res = st.session_state["last_result"]

    if res is None:
        st.info("Vui lòng thực thi câu hỏi tại Tab 1 trước để xem Cây Parent–Child.")
    else:
        parent_cands = res.get("parent_candidates", [])
        if not parent_cands:
            st.warning("Chưa có Parent Candidates được mở rộng.")
        else:
            for idx, p in enumerate(parent_cands, start=1):
                node = hierarchical_rag.format_parent_tree_node(p)
                rank_change_str = f"▲ +{node['rank_change']}" if node['rank_change'] > 0 else (f"▼ {node['rank_change']}" if node['rank_change'] < 0 else "=")
                header_text = (
                    f"🏢 #{idx} Parent [{node['parent_id']}] | Rerank Score: {node['rerank_score']:.4f} | "
                    f"Rank: #{node['rank_before']} ➔ #{node['rank_after']} ({rank_change_str}) | Support Qs: {node['support_queries']}"
                )

                with st.expander(header_text, expanded=(idx <= 2)):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.write(f"**Nguồn**: {node['source']} ({node['pages']})")
                        st.write(f"**Parent RRF Score**: {node['rrf_score']:.6f}")
                    with col_p2:
                        st.write(f"**Anchor Child**: `{node['anchor_child']}`")
                        st.write(f"**Số Child Hỗ Trợ**: {node['supporting_children_count']}")

                    if node["ambiguous"]:
                        st.warning("⚠️ Parent này chứa Child Chunk có sự chưa rõ ràng trong việc phân giải Hierarchy.")

                    st.markdown("**Cấu trúc cây hỗ trợ (Supporting Children)**:")
                    for cid in p.get("supporting_child_ids", []):
                        st.markdown(f"  └── 📄 **Child**: `{cid}`")

                    st.markdown("**Nội dung Parent Document đầy đủ (Thu gọn)**:")
                    st.text_area(f"Text parent {node['parent_id']}", value=p.get("text", ""), height=150, key=f"txt_{node['parent_id']}")

# -----------------------------------------------------------------------------
# TAB 4: MODE COMPARISON
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 📊 So Sánh 4 Retrieval Modes (Retrieval-Only)")
    st.caption("Chạy cùng một câu hỏi qua cả 4 modes để so sánh số lượng bằng chứng, độ trễ và hệ số mở rộng ngữ cảnh:")

    comp_q = st.text_input("Câu hỏi so sánh:", value=default_q, key="cmp_q_input")

    if st.button("🚀 Chạy So Sánh 4 Modes", use_container_width=True):
        with st.spinner("Đang chạy so sánh qua 4 modes (single_flat, multi_flat, single_parent, multi_parent)..."):
            comp_res = hierarchical_rag.compare_retrieval_modes(question=comp_q, config=config)
            st.session_state["compare_result"] = comp_res

    comp_res = st.session_state["compare_result"]

    if comp_res is not None:
        st.markdown(f"**Tổng thời gian so sánh**: `{comp_res['total_compare_ms']} ms`")

        compare_table_rows = []
        for m_name, m_data in comp_res["modes"].items():
            accepted = m_data.get("accepted_evidence", [])
            top_score = accepted[0].get("parent_rerank_score", accepted[0].get("rerank_score", 0.0)) if accepted else 0.0
            exp_factor = m_data.get("expansion_trace", {}).get("context_expansion_factor", 1.0) if "expansion_trace" in m_data else 1.0

            compare_table_rows.append({
                "Mode": m_name,
                "Unit Type": "parent" if "parent" in m_name else "child",
                "Status": m_data.get("status"),
                "Accepted Evidence": len(accepted),
                "Top Rerank Score": round(top_score, 4),
                "Expansion Factor": f"{exp_factor:.1f}x",
                "Retrieval Child Count": len(m_data.get("child_hits", [])),
                "Gen Calls": m_data.get("api_call_counts", {}).get("generation_calls", 0),
                "Embed Calls": m_data.get("api_call_counts", {}).get("embedding_calls", 0),
                "Latency (ms)": m_data.get("latency_ms", {}).get("total_ms", 0),
            })

        df_compare = pd.DataFrame(compare_table_rows)
        st.dataframe(df_compare, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: EVALUATION
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 📈 Báo Cáo Đánh Giá Chất Lượng Retrieval & RAG (Buổi 09)")

    # Read latest evaluation report if exists
    eval_report_file = BUOI_09_DIR / "reports" / "evaluation_report.json"

    if not eval_report_file.exists():
        st.info("Chưa có báo cáo đánh giá tự động tại `reports/evaluation_report.json`. Vui lòng chạy script `evaluate.py` từ CLI.")
    else:
        try:
            with open(eval_report_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            st.success(f"Báo cáo đánh giá mới nhất: `{eval_data.get('timestamp', 'N/A')}`")

            summary_dict = eval_data.get("summary_per_mode", {})
            mp_summary = summary_dict.get("multi_parent", eval_data.get("metrics", {}))

            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            col_e1.metric("Child Recall@K", f"{mp_summary.get('mean_child_recall_at_k', mp_summary.get('child_recall_at_k', 0.0)):.4f}")
            col_e2.metric("Parent Recall@K", f"{mp_summary.get('mean_parent_recall_at_k', mp_summary.get('parent_recall_at_k', 0.0)):.4f}")
            col_e3.metric("MRR@K", f"{mp_summary.get('mean_mrr_at_k', mp_summary.get('mrr_at_k', 0.0)):.4f}")
            col_e4.metric("nDCG@K", f"{mp_summary.get('mean_ndcg_at_k', mp_summary.get('ndcg_at_k', 0.0)):.4f}")

            if eval_data.get("needs_human_review"):
                st.warning("⚠️ Cảnh báo: Tập dữ liệu kiểm thử chứa các câu hỏi đánh dấu `needs_human_review = true` cần được chuyên gia xem xét.")

            st.markdown("#### Bảng Chi Tiết Kết Quả Đánh Giá Theo Mode:")
            table_rows = []
            for m_name, m_stats in summary_dict.items():
                table_rows.append({
                    "Mode": m_name,
                    "Child Recall@K": round(m_stats.get("mean_child_recall_at_k", 0.0), 4),
                    "Parent Recall@K": round(m_stats.get("mean_parent_recall_at_k", 0.0), 4),
                    "MRR@K": round(m_stats.get("mean_mrr_at_k", 0.0), 4),
                    "nDCG@K": round(m_stats.get("mean_ndcg_at_k", 0.0), 4),
                    "Expansion Factor": f"{m_stats.get('mean_context_expansion_factor', 1.0):.1f}x",
                    "Avg Latency (ms)": m_stats.get("mean_latency_ms", 0.0),
                    "Gen Calls": m_stats.get("total_generation_calls", 0),
                    "Embed Calls": m_stats.get("total_embedding_calls", 0),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi đọc file evaluation_report.json: {e}")
