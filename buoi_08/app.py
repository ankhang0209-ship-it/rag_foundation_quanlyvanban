"""
Streamlit Web Application - Buổi 08: Advanced Hybrid RAG Engine Dashboard.
Giao diện hiển thị trực quan Pipeline truy xuất nhiều tầng, Bảng so sánh 4 chế độ Retrieval,
Pipeline Trace chi tiết và Báo cáo đánh giá Offline.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import streamlit as st

# Cấu hình đường dẫn tĩnh đảm bảo import chạy chính xác từ mọi CWD
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import advanced_rag
import rag

# -----------------------------------------------------------------------------
# PAGE CONFIG & CSS STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Hybrid RAG Engine - Buổi 08",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .badge-accepted { background-color: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .card-box { border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #FAFAFA; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SIDEBAR - SYSTEM STATUS & CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Cấu hình & Trạng thái")

strategy = st.sidebar.selectbox(
    "Chiến lược Chunking (Strategy)",
    ["hierarchical", "semantic", "fixed-size"],
    index=0,
    help="Chọn chiến lược chia nhỏ văn bản cần truy vấn",
)

mode = st.sidebar.selectbox(
    "Retrieval Mode Mặc định",
    ["hybrid_rerank", "hybrid", "semantic", "bm25"],
    index=0,
    help="hybrid_rerank là chế độ khuyến nghị cho Advanced RAG",
)

final_top_k = st.sidebar.number_input(
    "Final Top-K Evidences",
    min_value=1,
    max_value=20,
    value=5,
    help="Số lượng bằng chứng cuối cùng chọn đưa vào context",
)

# Read status & config (read-only, zero side-effects)
try:
    st_info = advanced_rag.get_status(strategy=strategy)
    cfg = advanced_rag.load_config()
except Exception as e:
    st.sidebar.error(f"Lỗi đọc trạng thái: {e}")
    st_info = {}
    cfg = {}

with st.sidebar.expander("📌 Trạng thái Hệ thống (Read-only)", expanded=True):
    st.markdown(f"**Corpus Size:** {st_info.get('corpus_size', 0)} chunks")
    st.markdown(f"**BM25 Ready:** {'✅ Sẵn sàng' if st_info.get('bm25_ready') else '❌ Chưa ready'}")

    col_exists = st_info.get('collection_exists', False)
    col_count = st_info.get('collection_count', 0)
    st.markdown(f"**Semantic Index:** {'✅ ' + str(col_count) + ' recs' if col_exists else '❌ Chưa khởi tạo'}")

    reranker_model = st_info.get('reranker_model_name', 'BAAI/bge-reranker-v2-m3')
    cache_exists = st_info.get('reranker_cache_exists', False)
    st.markdown(f"**Reranker Model:** `{reranker_model}`")
    st.markdown(f"**Reranker Cache:** {'✅ Đã cache' if cache_exists else '⚠️ Chưa cache (Tải khi dùng)'}")

    api_key_status = "✅ Đã cấu hình" if cfg.get("api_key") else "❌ Chưa cấu hình"
    st.markdown(f"**Gemini API Key:** {api_key_status}")

with st.sidebar.expander("⚙️ Tham số RAG & Gating"):
    st.markdown(f"- **BM25 Candidates:** `{cfg.get('bm25_candidates', 20)}`")
    st.markdown(f"- **Semantic Candidates:** `{cfg.get('semantic_candidates', 20)}`")
    st.markdown(f"- **RRF k:** `{cfg.get('rrf_k', 60)}` (Weights: BM25 `{cfg.get('rrf_bm25_weight')}` / Sem `{cfg.get('rrf_semantic_weight')}`)")
    st.markdown(f"- **Rerank Candidates:** `{cfg.get('rerank_candidates', 20)}`")
    st.markdown(f"- **Rerank Min Score:** `{cfg.get('rerank_min_score', 0.50)}`")
    st.markdown(f"- **Max Distance (Cosine):** `{cfg.get('max_distance', 0.45)}`")


# -----------------------------------------------------------------------------
# MAIN APP HEADER & TABS
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>🔍 Advanced Hybrid RAG Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Hệ thống RAG nâng cao kết hợp BM25 Keyword Search, Gemini Vector Retrieval, RRF Fusion & Cross-Encoder Reranking</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Hỏi đáp Advanced RAG",
    "📊 So sánh Retrieval (4 Modes)",
    "🔬 Pipeline Trace",
    "📈 Báo cáo Đánh giá",
])


# -----------------------------------------------------------------------------
# TAB 1 — HỎI ĐÁP ADVANCED RAG
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("💬 Hỏi đáp & Trích dẫn Nguồn dữ liệu (Grounding)")

    default_question = "Điều kiện để tổ chức tín dụng cơ cấu lại thời hạn trả nợ gốc và lãi vay theo Thông tư 02 là gì?"
    question_input = st.text_area("Nhập câu hỏi cần tra cứu:", value=default_question, height=85)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_query_btn = st.button("🚀 Truy vấn RAG", type="primary", use_container_width=True)

    if run_query_btn and question_input.strip():
        with st.spinner("Đang thực hiện quy trình Advanced RAG (Retrieval + Rerank + Grounding)..."):
            res = advanced_rag.query_advanced_rag(
                question=question_input.strip(),
                top_k=final_top_k,
                strategy=strategy,
                mode=mode,
            )
            st.session_state["latest_query_result"] = res
            st.session_state["latest_question"] = question_input.strip()

    # Hiển thị kết quả truy vấn gần nhất
    if "latest_query_result" in st.session_state:
        res = st.session_state["latest_query_result"]
        status = res.get("status")

        st.markdown("---")
        # Status indicators
        if status == "answered":
            st.success("✅ **Trạng thái:** Đã trả lời thành công (answered)")
        elif status == "insufficient_evidence":
            st.warning("⚠️ **Trạng thái:** Không có đủ bằng chứng phù hợp trong tài liệu để trả lời câu hỏi (insufficient_evidence)")
        elif status == "retrieval_only":
            st.info("ℹ️ **Trạng thái:** Retrieval Only (Chỉ thực hiện truy xuất bằng chứng)")
        elif status == "reranker_unavailable":
            st.error("❌ **Trạng thái:** Reranker Model Chưa Sẵn Sàng (reranker_unavailable)")
            st.info("💡 **Hướng dẫn xử lý:** Vui lòng nạp/tải weights Cross-Encoder model bằng cách chạy lệnh CLI:\n`python rag_foundation/buoi_08/advanced_rag.py rerank --question \"...\"`")

        # Answer text
        if res.get("answer"):
            st.markdown("### 💡 Câu trả lời")
            st.markdown(f"> {res['answer']}")

        # Citations
        citations = res.get("citations", [])
        if citations:
            st.markdown("### 📌 Trích dẫn Nguồn dữ liệu (Citations)")
            for cite in citations:
                p_str = f"Trang {cite['page_start']}" if cite['page_start'] == cite['page_end'] else f"Trang {cite['page_start']}-{cite['page_end']}"
                st.markdown(f"- **{cite['label']}** → Source: `{cite['source']}` ({p_str}) | Chunk ID: `{cite['chunk_id']}`")

        # Warnings
        warnings_list = res.get("warnings", [])
        if warnings_list:
            with st.expander("⚠️ Cảnh báo Pipeline (Warnings)", expanded=False):
                for w in warnings_list:
                    st.write(f"- {w}")

        # Evidences Cards
        evidences = res.get("evidence", [])
        if evidences:
            st.markdown("### 📄 Danh sách Bằng chứng (Evidences & Score Details)")
            for idx, ev in enumerate(evidences, start=1):
                p_str = f"Trang {ev['page_start']}" if ev['page_start'] == ev['page_end'] else f"Trang {ev['page_start']}-{ev['page_end']}"
                acc_badge = "<span class='badge-accepted'>✅ ACCEPTED</span>" if ev['accepted'] else "<span class='badge-rejected'>❌ REJECTED</span>"

                expander_title = f"[{idx}] {ev['chunk_id']} | Source: {ev['source']} ({p_str})"
                with st.expander(expander_title, expanded=(idx <= 3)):
                    st.markdown(f"**Trạng thái Gating:** {acc_badge}", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        b_rank = f"#{ev['bm25_rank']}" if ev['bm25_rank'] else "N/A"
                        b_score = f"{ev['bm25_score']:.4f}" if ev['bm25_score'] is not None else "N/A"
                        st.metric("BM25 Rank / Score", b_rank, b_score)
                    with c2:
                        s_rank = f"#{ev['semantic_rank']}" if ev['semantic_rank'] else "N/A"
                        s_dist = f"{ev['semantic_distance']:.4f}" if ev['semantic_distance'] is not None else "N/A"
                        st.metric("Semantic Rank / Dist", s_rank, s_dist)
                    with c3:
                        f_rank = f"#{ev['fused_rank']}" if ev['fused_rank'] else "N/A"
                        f_score = f"{ev['rrf_score']:.6f}" if ev['rrf_score'] is not None else "N/A"
                        st.metric("RRF Fused Rank / Score", f_rank, f_score)
                    with c4:
                        r_rank = f"#{ev['rerank_rank']}" if ev['rerank_rank'] else "N/A"
                        r_score = f"{ev['rerank_score']:.4f}" if ev['rerank_score'] is not None else "N/A"
                        chg = ev.get('rank_change')
                        chg_str = f"+{chg}" if isinstance(chg, int) and chg > 0 else f"{chg}" if chg is not None else "N/A"
                        st.metric("Rerank Rank / Score", r_rank, f"{r_score} (Move: {chg_str})")

                    st.markdown("**Nội dung Chunk:**")
                    st.text_area(f"text_{idx}", value=ev['text'], height=100, disabled=True, label_visibility="collapsed")


# -----------------------------------------------------------------------------
# TAB 2 — SO SÁNH RETRIEVAL (4 MODES)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 So sánh Trực quan 4 Chế độ Retrieval")
    st.caption("Chạy cùng một câu hỏi qua BM25, Semantic Search, Hybrid RRF và Hybrid + Rerank. Chế độ này KHÔNG gọi Generation (0 calls).")

    q_comp_input = st.text_input(
        "Câu hỏi so sánh:",
        value=st.session_state.get("latest_question", default_question),
        key="comp_question_input",
    )

    if st.button("📊 Chạy So sánh 4 Modes", type="primary", key="run_compare_btn"):
        with st.spinner("Đang truy xuất kết quả qua 4 chế độ..."):
            comp_res = advanced_rag.compare_retrieval_modes(question=q_comp_input.strip(), strategy=strategy)
            st.session_state["compare_result"] = comp_res

    if "compare_result" in st.session_state:
        comp_res = st.session_state["compare_result"]
        table_data = comp_res.get("comparison_table", [])

        st.markdown("---")
        st.markdown("### 📋 Bảng Tổng hợp Thứ hạng Chunks qua 4 Mode")

        # Convert to DataFrame for visual table
        rows = []
        for item in table_data:
            r = item["ranks"]
            chg = item.get("rank_change", "-")
            chg_str = f"+{chg}" if isinstance(chg, int) and chg > 0 else f"{chg}"
            rows.append({
                "Chunk ID": item["chunk_id"],
                "Source": item["source"],
                "BM25 Rank": f"#{r.get('bm25')}" if r.get('bm25') else "-",
                "Semantic Rank": f"#{r.get('semantic')}" if r.get('semantic') else "-",
                "RRF Rank": f"#{r.get('hybrid')}" if r.get('hybrid') else "-",
                "Rerank Rank": f"#{r.get('hybrid_rerank')}" if r.get('hybrid_rerank') else "-",
                "Rank Movement": chg_str,
                "Final Modes": " + ".join(item["presence"]),
            })

        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("### 📌 Chi tiết Top-K Ứng viên theo từng Mode")
        mode_results = comp_res.get("mode_results", {})

        col1, col2, col3, col4 = st.columns(4)
        modes_info = [
            ("BM25 Lexical", "bm25", col1),
            ("Semantic Vector", "semantic", col2),
            ("Hybrid RRF", "hybrid", col3),
            ("Hybrid + Rerank", "hybrid_rerank", col4),
        ]

        for title, m_key, col in modes_info:
            with col:
                st.markdown(f"#### {title}")
                m_evs = mode_results.get(m_key, {}).get("evidence", [])
                for i, ev in enumerate(m_evs, start=1):
                    rank_info = ""
                    if m_key == "bm25":
                        rank_info = f"Score: {ev['bm25_score']:.2f}"
                    elif m_key == "semantic":
                        rank_info = f"Dist: {ev['semantic_distance']:.4f}"
                    elif m_key == "hybrid":
                        rank_info = f"RRF: {ev['rrf_score']:.4f}"
                    elif m_key == "hybrid_rerank":
                        rank_info = f"Rerank: {ev['rerank_score']:.4f}"

                    st.markdown(f"**[{i}] `{ev['chunk_id']}`**")
                    st.caption(f"{ev['source']} | {rank_info}")
                    st.text(ev['text'][:70] + "...")
                    st.markdown("---")


# -----------------------------------------------------------------------------
# TAB 3 — PIPELINE TRACE
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Chi tiết Pipeline Trace Nhiều Tầng")

    if "latest_query_result" not in st.session_state:
        st.info("Vui lòng thực hiện một truy vấn ở Tab 1 để xem thông tin Pipeline Trace chi tiết.")
    else:
        res = st.session_state["latest_query_result"]
        tr = res.get("trace", {})
        lat = tr.get("latency_ms", {})

        st.markdown("### 📈 Luồng Dịch chuyển Ứng viên (Candidate Flow)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("BM25 Candidates", tr.get("bm25_candidates", 0))
        m2.metric("Semantic Candidates", tr.get("semantic_candidates", 0))
        m3.metric("Union / Overlap", f"{tr.get('union', 0)} / {tr.get('overlap', 0)}")
        m4.metric("Reranked Candidates", tr.get("reranked", 0))
        m5.metric("Accepted Evidences", tr.get("accepted", 0))

        st.markdown("---")
        st.markdown("### ⏱️ Phân tích Thời gian Xử lý (Latency Breakdown)")
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("BM25 (ms)", f"{lat.get('bm25', 0):.2f}ms")
        l2.metric("Semantic (ms)", f"{lat.get('semantic', 0):.2f}ms")
        l3.metric("RRF Fusion (ms)", f"{lat.get('fusion', 0):.2f}ms")
        l4.metric("Rerank (ms)", f"{lat.get('rerank', 0):.2f}ms")
        l5.metric("Generation (ms)", f"{lat.get('generation', 0):.2f}ms")
        l6.metric("Total Latency", f"{lat.get('total', 0):.2f}ms", delta_color="inverse")

        st.markdown("---")
        st.info(
            r"""
            📌 **Hướng dẫn Đọc Chỉ số Score & Distance:**
            - **BM25 Score**: Điểm số trùng khớp từ khóa. *Càng cao càng tốt*.
            - **Cosine Distance**: Khoảng cách vector ngữ nghĩa giữa câu hỏi và chunk. *Càng thấp càng tốt* (0.0 = trùng tuyệt đối).
            - **RRF Score**: Điểm dung hợp Reciprocal Rank Fusion ($1 / (k + r)$). *Càng cao càng tốt*.
            - **Rerank Score (Sigmoid)**: Điểm tương quan trực tiếp từ Cross-Encoder đưa về dải $[0, 1]$. *Càng cao càng tốt* ($\ge 0.50$).
            - ⚠️ **Lưu ý**: Rerank score thể hiện độ tương quan ngữ nghĩa, **KHÔNG PHẢI xác suất toán học**.
            """
        )


# -----------------------------------------------------------------------------
# TAB 4 — BÁO CÁO ĐÁNH GIÁ (EVALUATION)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📈 Báo cáo Đánh giá Offline RAG Engine")

    # Read-only report reader
    reports_dir = BASE_DIR / "reports"
    report_files = list(reports_dir.glob("*.json")) if reports_dir.exists() else []

    # Check gold labels in eval/questions.json for warnings
    eval_q_file = BASE_DIR / "eval" / "questions.json"
    needs_review = False
    if eval_q_file.exists():
        try:
            with open(eval_q_file, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
                needs_review = any(q.get("needs_human_review", False) for q in questions_data)
        except Exception:
            pass

    if needs_review:
        st.warning("⚠️ **Cảnh báo:** Tập câu hỏi đánh giá (`eval/questions.json`) vẫn còn chứa các câu hỏi gắn nhãn `needs_human_review = True`. Kết quả chưa nên dùng để kết luận winner chính thức.")

    if not report_files:
        st.warning("⚠️ **Chưa tìm thấy tệp báo cáo đánh giá trong thư mục `reports/`.**")
        st.info("💡 Để tạo báo cáo đánh giá offline, vui lòng chạy lệnh CLI bên dưới từ terminal:")
        st.code("python rag_foundation/buoi_08/evaluate.py", language="bash")
        st.caption("Lưu ý: Ứng dụng web tuyệt đối không tự động chạy đánh giá hàng loạt API khi vừa mở trang.")
    else:
        selected_report = st.selectbox("Chọn báo cáo đánh giá:", [f.name for f in report_files])
        report_path = reports_dir / selected_report

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            st.json(report_data)
        except Exception as e:
            st.error(f"Lỗi khi đọc tệp báo cáo {selected_report}: {e}")

