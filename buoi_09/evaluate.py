"""
Module Đánh Giá Tự Động Retrieval Modes (Buổi 09).
So sánh 4 modes (single_flat, multi_flat, single_parent, multi_parent) trên tập câu hỏi đánh giá.
"""

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BUOI_09_DIR = Path(__file__).resolve().parent
if str(BUOI_09_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_09_DIR))

import hierarchical_rag


def calculate_recall_at_k(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Tính điểm Recall@K."""
    if not gold_ids:
        return 1.0 if not retrieved_ids else 0.0
    hits = sum(1 for gid in gold_ids if gid in retrieved_ids)
    return hits / len(gold_ids)


def calculate_mrr_at_k(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Tính điểm Mean Reciprocal Rank (MRR@K)."""
    if not gold_ids or not retrieved_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Tính điểm Normalized Discounted Cumulative Gain (nDCG@K - binary relevance)."""
    if not gold_ids or not retrieved_ids:
        return 0.0

    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = 0.0
    for rank in range(1, min(len(gold_ids), len(retrieved_ids)) + 1):
        idcg += 1.0 / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def load_evaluation_questions(
    questions_file: Optional[Path] = None,
    hierarchy_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Tải tập câu hỏi đánh giá và validate các parent_ids/child_ids đối chiếu với Hierarchy Store.
    """
    if questions_file is None:
        questions_file = BUOI_09_DIR / "eval" / "questions.json"

    questions_file = Path(questions_file)
    if not questions_file.exists():
        raise FileNotFoundError(f"Tệp câu hỏi đánh giá không tồn tại: {questions_file}")

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Validate against Hierarchy Store
    status_info = hierarchical_rag.get_hierarchy_status(output_dir=hierarchy_dir)
    needs_review = any(q.get("needs_human_review", False) for q in questions)

    if status_info.get("store_exists"):
        children_dict, parents_dict, _ = hierarchical_rag.load_hierarchy_store_data(hierarchy_dir=hierarchy_dir)
        stale_ids = []
        for q in questions:
            for pid in q.get("relevant_parent_ids", []):
                if pid not in parents_dict:
                    stale_ids.append(f"Parent_id '{pid}' trong {q['question_id']}")
            for cid in q.get("relevant_child_ids", []):
                if cid not in children_dict:
                    stale_ids.append(f"Child_id '{cid}' trong {q['question_id']}")

        if stale_ids:
            raise ValueError(f"Tập câu hỏi chứa các IDs hết hạn (stale IDs) không tồn tại trong Hierarchy Store: {stale_ids}")

    return questions, needs_review


def evaluate_retrieval_modes(
    config: Optional[Dict[str, Any]] = None,
    questions_file: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    hierarchy_dir: Optional[Path] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Thực thi đánh giá 4 Retrieval Modes trên toàn bộ tập câu hỏi (Retrieval-Only).
    """
    if config is None:
        config = hierarchical_rag.load_config()

    if reports_dir is None:
        reports_dir = BUOI_09_DIR / "reports"

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    questions, needs_human_review = load_evaluation_questions(questions_file=questions_file, hierarchy_dir=hierarchy_dir)

    if hybrid_retriever_fn is None:
        import advanced_rag
        col_status = advanced_rag.get_status()
        if not col_status.get("collection_exists") or not config.get("api_key"):
            children_dict, _, _ = hierarchical_rag.load_hierarchy_store_data(hierarchy_dir=hierarchy_dir)
            child_list = list(children_dict.values())
            def offline_bm25_retriever(q, k):
                q_words = set(q.lower().split())
                scored = []
                for c in child_list:
                    c_words = set(c.get("text", "").lower().split())
                    overlap = len(q_words.intersection(c_words))
                    scored.append((overlap, c))
                scored.sort(key=lambda x: -x[0])
                return [item[1] for item in scored[:k]]
            hybrid_retriever_fn = offline_bm25_retriever

    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    mode_aggregate_metrics = {}
    per_question_results = []

    for q_item in questions:
        qid = q_item["question_id"]
        q_text = q_item["question"]
        q_type = q_item["question_type"]
        gold_children = q_item.get("relevant_child_ids", [])
        gold_parents = q_item.get("relevant_parent_ids", [])

        q_res_record = {
            "question_id": qid,
            "question": q_text,
            "question_type": q_type,
            "gold_child_ids": gold_children,
            "gold_parent_ids": gold_parents,
            "modes": {},
        }

        for m in modes:
            t0 = time.perf_counter()
            res = hierarchical_rag.execute_query_pipeline(
                question=q_text,
                mode=m,
                config=config,
                hierarchy_dir=hierarchy_dir,
                query_generator_fn=query_generator_fn,
                hybrid_retriever_fn=hybrid_retriever_fn,
                reranker_fn=reranker_fn,
                answer_generator_fn=lambda q, ctx, cits: "EVAL_NO_GEN",
            )
            t1 = time.perf_counter()
            mode_ms = round((t1 - t0) * 1000, 2)

            accepted = res.get("accepted_evidence", [])
            retrieved_ids = [item.get("parent_id") or item.get("child_id") for item in accepted]
            retrieved_child_ids = [c.get("child_id") or c.get("chunk_id") for c in res.get("child_hits", [])]

            c_recall = calculate_recall_at_k(retrieved_child_ids, gold_children)
            p_recall = calculate_recall_at_k(retrieved_ids, gold_parents)
            mrr = calculate_mrr_at_k(retrieved_ids, gold_parents)
            ndcg = calculate_ndcg_at_k(retrieved_ids, gold_parents)

            exp_factor = res.get("expansion_trace", {}).get("context_expansion_factor", 1.0) if "expansion_trace" in res else 1.0
            context_chars = res.get("expansion_trace", {}).get("expanded_parent_chars_total", 0) if "expansion_trace" in res else sum(len(c.get("text", "")) for c in accepted)

            m_metrics = {
                "status": res["status"],
                "child_recall_at_k": round(c_recall, 4),
                "parent_recall_at_k": round(p_recall, 4),
                "mrr_at_k": round(mrr, 4),
                "ndcg_at_k": round(ndcg, 4),
                "accepted_evidence_count": len(accepted),
                "retrieved_child_count": len(child_hits := res.get("child_hits", [])),
                "context_chars": context_chars,
                "context_expansion_factor": exp_factor,
                "latency_ms": mode_ms,
                "generation_calls": res["api_call_counts"]["generation_calls"],
                "embedding_calls": res["api_call_counts"]["embedding_calls"],
            }

            q_res_record["modes"][m] = m_metrics

            # Accumulate mode stats
            if m not in mode_aggregate_metrics:
                mode_aggregate_metrics[m] = {
                    "child_recalls": [],
                    "parent_recalls": [],
                    "mrrs": [],
                    "ndcgs": [],
                    "latencies": [],
                    "expansion_factors": [],
                    "total_gen_calls": 0,
                    "total_embed_calls": 0,
                }

            mode_aggregate_metrics[m]["child_recalls"].append(c_recall)
            mode_aggregate_metrics[m]["parent_recalls"].append(p_recall)
            mode_aggregate_metrics[m]["mrrs"].append(mrr)
            mode_aggregate_metrics[m]["ndcgs"].append(ndcg)
            mode_aggregate_metrics[m]["latencies"].append(mode_ms)
            mode_aggregate_metrics[m]["expansion_factors"].append(exp_factor)
            mode_aggregate_metrics[m]["total_gen_calls"] += res["api_call_counts"]["generation_calls"]
            mode_aggregate_metrics[m]["total_embed_calls"] += res["api_call_counts"]["embedding_calls"]

        per_question_results.append(q_res_record)

    # Compute final aggregates per mode
    summary_per_mode = {}
    for m, stats in mode_aggregate_metrics.items():
        n = max(1, len(stats["latencies"]))
        lats = sorted(stats["latencies"])
        p50_lat = lats[len(lats) // 2] if lats else 0.0

        summary_per_mode[m] = {
            "mean_child_recall_at_k": round(sum(stats["child_recalls"]) / n, 4),
            "mean_parent_recall_at_k": round(sum(stats["parent_recalls"]) / n, 4),
            "mean_mrr_at_k": round(sum(stats["mrrs"]) / n, 4),
            "mean_ndcg_at_k": round(sum(stats["ndcgs"]) / n, 4),
            "mean_context_expansion_factor": round(sum(stats["expansion_factors"]) / n, 2),
            "mean_latency_ms": round(sum(stats["latencies"]) / n, 2),
            "p50_latency_ms": p50_lat,
            "total_generation_calls": stats["total_gen_calls"],
            "total_embedding_calls": stats["total_embed_calls"],
        }

    timestamp_str = datetime.now(timezone.utc).isoformat()
    report_data = {
        "timestamp": timestamp_str,
        "config": {
            "multi_query_count": config["multi_query_count"],
            "per_query_candidates": config["per_query_candidates"],
            "parent_candidates": config["parent_candidates"],
            "final_parent_top_k": config["final_parent_top_k"],
            "rerank_min_score": config["rerank_min_score"],
            "generation_model": config["generation_model"],
            "reranker_model": config["reranker_model"],
        },
        "needs_human_review": needs_human_review,
        "summary_per_mode": summary_per_mode,
        "per_question_results": per_question_results,
    }

    # Write report atomically to reports/evaluation_report.json and reports/latest_report.json
    rep_file = reports_dir / f"report_{int(time.time())}.json"
    latest_file = reports_dir / "latest_report.json"
    std_file = reports_dir / "evaluation_report.json"

    for target in [rep_file, latest_file, std_file]:
        tmp_p = target.with_suffix(".json.tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        tmp_p.replace(target)

    return report_data


if __name__ == "__main__":
    print("\n============================================================")
    print("  ĐÁNH GIÁ TỰ ĐỘNG RETRIEVAL MODES (evaluate.py)")
    print("============================================================")
    rep = evaluate_retrieval_modes()
    print(f"✅ Đã tạo báo cáo đánh giá tại: reports/evaluation_report.json")
    print(f" Timestamp           : {rep['timestamp']}")
    print(f" Needs Human Review  : {rep['needs_human_review']}")
    print("------------------------------------------------------------")
    for m, m_stats in rep["summary_per_mode"].items():
        print(f" 🔹 Mode [{m:13s}] | Child Recall: {m_stats['mean_child_recall_at_k']:.4f} | Parent Recall: {m_stats['mean_parent_recall_at_k']:.4f} | MRR: {m_stats['mean_mrr_at_k']:.4f} | nDCG: {m_stats['mean_ndcg_at_k']:.4f} | Latency: {m_stats['mean_latency_ms']}ms")
    print("============================================================\n")
