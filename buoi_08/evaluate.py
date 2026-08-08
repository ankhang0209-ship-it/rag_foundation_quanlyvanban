"""
Framework Đánh giá Offline RAG Engine - Buổi 08.
Đo lường Hit Rate@K, Recall@K, MRR@K, nDCG@K và Latency (mean & p50) giữa 4 Retrieval Modes.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Đường dẫn tĩnh độc lập với CWD
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import advanced_rag

EVAL_QUESTIONS_PATH = BASE_DIR / "eval" / "questions.json"
REPORTS_DIR = BASE_DIR / "reports"


def calculate_hit_rate(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Tính chỉ số Hit Rate@K (1.0 nếu có ít nhất 1 relevant chunk trong top K, ngược lại 0.0).
    """
    if not relevant_chunk_ids:
        return 0.0
    top_k_retrieved = retrieved_chunk_ids[:k]
    hit = any(cid in relevant_chunk_ids for cid in top_k_retrieved)
    return 1.0 if hit else 0.0


def calculate_recall(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Tính chỉ số Recall@K (Tỷ lệ các relevant chunks được tìm thấy trong top K).
    """
    if not relevant_chunk_ids:
        return 0.0
    top_k_retrieved = set(retrieved_chunk_ids[:k])
    rel_set = set(relevant_chunk_ids)
    hits = top_k_retrieved.intersection(rel_set)
    return len(hits) / float(len(rel_set))


def calculate_mrr(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Tính chỉ số Mean Reciprocal Rank (MRR@K).
    Trả về 1/rank của phần tử khớp đầu tiên trong top K (1-indexed), hoặc 0.0 nếu không khớp.
    """
    if not relevant_chunk_ids:
        return 0.0
    top_k_retrieved = retrieved_chunk_ids[:k]
    rel_set = set(relevant_chunk_ids)
    for idx, cid in enumerate(top_k_retrieved, start=1):
        if cid in rel_set:
            return 1.0 / float(idx)
    return 0.0


def calculate_ndcg(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Tính chỉ số nDCG@K (Normalized Discounted Cumulative Gain) với binary relevance (0 hoặc 1).
    """
    if not relevant_chunk_ids:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    rel_set = set(relevant_chunk_ids)

    # Tính DCG@K
    dcg = 0.0
    for idx, cid in enumerate(top_k_retrieved, start=1):
        rel = 1.0 if cid in rel_set else 0.0
        dcg += rel / math.log2(idx + 1)

    # Tính Ideal DCG@K (IDCG@K)
    ideal_hits = min(k, len(rel_set))
    idcg = 0.0
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_median(values: List[float]) -> float:
    """
    Helper tính P50 (Median) latency.
    """
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return sorted_v[mid]
    else:
        return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def run_evaluation(
    eval_file: Path = EVAL_QUESTIONS_PATH,
    output_report: Path = None,
    strategy: str = "hierarchical",
    k: int = 5,
    modes: List[str] = None,
    reranker_fn: Any = None,
) -> Dict[str, Any]:
    """
    Chạy đánh giá so sánh các Retrieval Modes trên tập câu hỏi eval_file.
    TẤT CẢ các truy vấn đều KHÔNG gọi generation (0 calls).
    """
    if not eval_file.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp câu hỏi đánh giá: {eval_file}")

    if modes is None:
        modes = ["bm25", "semantic", "hybrid", "hybrid_rerank"]

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        raise ValueError("Tập câu hỏi đánh giá rỗng.")

    needs_human_review_warning = any(q.get("needs_human_review", False) for q in questions)

    config = advanced_rag.load_config()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_report is None:
        output_report = REPORTS_DIR / "evaluation_report.json"

    query_details = []
    mode_metrics_collector = {
        m: {
            "hit_rates": [],
            "recalls": [],
            "mrrs": [],
            "ndcgs": [],
            "latencies": [],
            "failures": 0,
        }
        for m in modes
    }

    for q_item in questions:
        qid = q_item.get("query_id", "Q_UNK")
        question_text = q_item["question"]
        relevant_cids = q_item.get("relevant_chunk_ids", [])
        scope = q_item.get("scope", "in_scope")

        item_detail = {
            "query_id": qid,
            "question": question_text,
            "scope": scope,
            "relevant_chunk_ids": relevant_cids,
            "modes": {},
        }

        for m in modes:
            try:
                res = advanced_rag.query_advanced_rag(
                    question=question_text,
                    top_k=k,
                    strategy=strategy,
                    mode=m,
                    reranker_fn=reranker_fn,
                    call_generation=False,
                )

                evidences = res.get("evidence", [])
                retrieved_cids = [e["chunk_id"] for e in evidences]
                lat_total = res.get("trace", {}).get("latency_ms", {}).get("total", 0.0)

                hr = calculate_hit_rate(retrieved_cids, relevant_cids, k)
                rec = calculate_recall(retrieved_cids, relevant_cids, k)
                mrr = calculate_mrr(retrieved_cids, relevant_cids, k)
                ndcg = calculate_ndcg(retrieved_cids, relevant_cids, k)

                mode_metrics_collector[m]["hit_rates"].append(hr)
                mode_metrics_collector[m]["recalls"].append(rec)
                mode_metrics_collector[m]["mrrs"].append(mrr)
                mode_metrics_collector[m]["ndcgs"].append(ndcg)
                mode_metrics_collector[m]["latencies"].append(lat_total)

                item_detail["modes"][m] = {
                    "status": res["status"],
                    "retrieved_chunk_ids": retrieved_cids,
                    "hit_rate": hr,
                    "recall": rec,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "latency_ms": lat_total,
                }

            except Exception as e:
                mode_metrics_collector[m]["failures"] += 1
                item_detail["modes"][m] = {
                    "status": "failed",
                    "error": str(e),
                    "hit_rate": 0.0,
                    "recall": 0.0,
                    "mrr": 0.0,
                    "ndcg": 0.0,
                    "latency_ms": 0.0,
                }

        query_details.append(item_detail)

    # Tính toán tổng hợp chỉ số trung bình cho từng mode
    summary_by_mode = {}
    for m in modes:
        hrs = mode_metrics_collector[m]["hit_rates"]
        recs = mode_metrics_collector[m]["recalls"]
        mrrs = mode_metrics_collector[m]["mrrs"]
        ndcgs = mode_metrics_collector[m]["ndcgs"]
        lats = mode_metrics_collector[m]["latencies"]
        n_queries = len(hrs)

        summary_by_mode[m] = {
            "eval_count": n_queries,
            "failures_count": mode_metrics_collector[m]["failures"],
            "hit_rate_at_k": round(sum(hrs) / n_queries, 4) if n_queries > 0 else 0.0,
            "recall_at_k": round(sum(recs) / n_queries, 4) if n_queries > 0 else 0.0,
            "mrr_at_k": round(sum(mrrs) / n_queries, 4) if n_queries > 0 else 0.0,
            "ndcg_at_k": round(sum(ndcgs) / n_queries, 4) if n_queries > 0 else 0.0,
            "mean_latency_ms": round(sum(lats) / n_queries, 2) if n_queries > 0 else 0.0,
            "p50_latency_ms": round(calculate_median(lats), 2) if n_queries > 0 else 0.0,
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy,
        "k": k,
        "eval_questions_count": len(questions),
        "needs_human_review_warning": needs_human_review_warning,
        "warning_message": (
            "⚠️ Cảnh báo: Tập dữ liệu câu hỏi đánh giá có chứa câu hỏi gắn nhãn needs_human_review = True. "
            "Kết quả chỉ mang tính tham khảo, chưa dùng để tuyên bố chiến thắng chính thức giữa các mode."
            if needs_human_review_warning
            else None
        ),
        "model_identity": {
            "embedding_model": config["embedding_model"],
            "reranker_model": config["reranker_model"],
        },
        "config": {
            "bm25_candidates": config["bm25_candidates"],
            "semantic_candidates": config["semantic_candidates"],
            "rrf_k": config["rrf_k"],
            "rerank_candidates": config["rerank_candidates"],
            "final_top_k": config["final_top_k"],
        },
        "metrics_by_mode": summary_by_mode,
        "query_details": query_details,
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    """
    CLI Entry Point cho Framework Đánh giá Offline Buổi 08.
    """
    parser = argparse.ArgumentParser(description="Buổi 08 Offline RAG Evaluation CLI")
    parser.add_argument(
        "--strategy",
        type=str,
        default="hierarchical",
        choices=["fixed-size", "semantic", "hierarchical"],
        help="Chiến lược chunking đánh giá (mặc định: hierarchical)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Ngưỡng Top-K đánh giá (mặc định: 5)",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=str(EVAL_QUESTIONS_PATH),
        help="Đường dẫn tệp JSON câu hỏi đánh giá",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(REPORTS_DIR / "evaluation_report.json"),
        help="Đường dẫn tệp JSON lưu báo cáo",
    )

    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    out_path = Path(args.output_report)

    try:
        print("\n" + "=" * 75)
        print(f"  ĐANG CHẠY KHUNG ĐÁNH GIÁ OFFLINE RAG ENGINE (Strategy: {args.strategy}, K: {args.k})")
        print("=" * 75)
        print(f" Tệp câu hỏi eval : {eval_path}")
        print(f" Tệp báo cáo output: {out_path}")
        print(" Đang thực thi... (0 calls generation)\n")

        rep = run_evaluation(eval_file=eval_path, output_report=out_path, strategy=args.strategy, k=args.k)

        print("=" * 75)
        print("  KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP THEO RETRIEVAL MODE")
        print("=" * 75)
        print(f" {'Mode':<15} | {'Recall@K':<10} | {'MRR@K':<10} | {'nDCG@K':<10} | {'Mean Lat (ms)':<14} | {'P50 Lat (ms)':<12}")
        print("-" * 75)

        for m_name, m_data in rep["metrics_by_mode"].items():
            rec_str = f"{m_data['recall_at_k']:.4f}"
            mrr_str = f"{m_data['mrr_at_k']:.4f}"
            ndcg_str = f"{m_data['ndcg_at_k']:.4f}"
            mean_l = f"{m_data['mean_latency_ms']:.2f}ms"
            p50_l = f"{m_data['p50_latency_ms']:.2f}ms"
            print(f" {m_name:<15} | {rec_str:<10} | {mrr_str:<10} | {ndcg_str:<10} | {mean_l:<14} | {p50_l:<12}")

        print("-" * 75)
        if rep.get("warning_message"):
            print(f" {rep['warning_message']}")
        print(f"\n Báo cáo JSON chi tiết đã được xuất thành công vào: {out_path}\n")

    except Exception as e:
        print(f"\n❌ LỖI RUN EVALUATION: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
