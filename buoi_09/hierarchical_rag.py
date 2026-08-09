"""
Hierarchical RAG Engine - Buổi 09
Module quản lý Parent-Child Hierarchy Registry & Parent Store cho văn bản pháp luật.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

# Base Directory paths
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Storage paths
STORAGE_DIR = BASE_DIR / "storage"
HIERARCHY_DIR = STORAGE_DIR / "hierarchy"
CHILDREN_FILE = HIERARCHY_DIR / "children.json"
PARENTS_FILE = HIERARCHY_DIR / "parents.json"
MANIFEST_FILE = HIERARCHY_DIR / "manifest.json"

DEFAULT_INPUT_DIR = BASE_DIR.parent / "buoi_05" / "output" / "chunks"

# Load .env file (phụ thuộc vị trí file, không phụ thuộc CWD)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def load_config() -> Dict[str, Any]:
    """
    Tải và validate đầy đủ các tham số cấu hình Buổi 09 từ biến môi trường.
    """
    config = {
        "multi_query_count": int(os.getenv("MULTI_QUERY_COUNT", "3")),
        "multi_query_max_chars": int(os.getenv("MULTI_QUERY_MAX_CHARS", "300")),
        "multi_query_temperature": float(os.getenv("MULTI_QUERY_TEMPERATURE", "0.2")),
        "multi_query_original_weight": float(os.getenv("MULTI_QUERY_ORIGINAL_WEIGHT", "1.5")),
        "multi_query_variant_weight": float(os.getenv("MULTI_QUERY_VARIANT_WEIGHT", "1.0")),
        "multi_query_rrf_k": int(os.getenv("MULTI_QUERY_RRF_K", "60")),
        "per_query_candidates": int(os.getenv("PER_QUERY_CANDIDATES", "12")),
        "parent_max_chars": int(os.getenv("PARENT_MAX_CHARS", "6000")),
        "parent_score_child_limit": int(os.getenv("PARENT_SCORE_CHILD_LIMIT", "3")),
        "parent_rrf_k": int(os.getenv("PARENT_RRF_K", "60")),
        "parent_candidates": int(os.getenv("PARENT_CANDIDATES", "10")),
        "final_parent_top_k": int(os.getenv("FINAL_PARENT_TOP_K", "3")),
        "total_context_max_chars": int(os.getenv("TOTAL_CONTEXT_MAX_CHARS", "16000")),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        "embedding_dim": int(os.getenv("GEMINI_EMBEDDING_DIM", "768")),
        "generation_model": os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"),
        "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "rerank_min_score": float(os.getenv("RERANK_MIN_SCORE", "0.50")),
        "rerank_device": os.getenv("RERANK_DEVICE", "auto"),
        "api_key": os.getenv("GEMINI_API_KEY", ""),
    }
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Ràng buộc và kiểm tra giá trị hợp lệ cho tất cả tham số cấu hình.
    """
    if not (1 <= config["multi_query_count"] <= 5):
        raise ValueError(f"MULTI_QUERY_COUNT ({config['multi_query_count']}) phải từ 1 đến 5.")
    if not (50 <= config["multi_query_max_chars"] <= 1000):
        raise ValueError(f"MULTI_QUERY_MAX_CHARS ({config['multi_query_max_chars']}) phải từ 50 đến 1000.")
    if not (0.0 <= config["multi_query_temperature"] <= 1.0):
        raise ValueError(f"MULTI_QUERY_TEMPERATURE ({config['multi_query_temperature']}) phải từ 0.0 đến 1.0.")
    if config["multi_query_original_weight"] < 0 or config["multi_query_variant_weight"] < 0:
        raise ValueError("Trọng số Multi-Query không được âm.")
    if config["multi_query_original_weight"] == 0 and config["multi_query_variant_weight"] == 0:
        raise ValueError("Trọng số original và variant không được đồng thời bằng 0.")
    if config["multi_query_rrf_k"] <= 0 or config["parent_rrf_k"] <= 0:
        raise ValueError("RRF K phải là số nguyên dương.")
    if not (1 <= config["per_query_candidates"] <= 100):
        raise ValueError("PER_QUERY_CANDIDATES phải từ 1 đến 100.")
    if not (1 <= config["parent_candidates"] <= 100):
        raise ValueError("PARENT_CANDIDATES phải từ 1 đến 100.")
    if not (1000 <= config["parent_max_chars"] <= 20000):
        raise ValueError(f"PARENT_MAX_CHARS ({config['parent_max_chars']}) phải từ 1000 đến 20000.")
    if not (1 <= config["parent_score_child_limit"] <= 20):
        raise ValueError("PARENT_SCORE_CHILD_LIMIT phải từ 1 đến 20.")
    if config["final_parent_top_k"] > config["parent_candidates"]:
        raise ValueError(f"FINAL_PARENT_TOP_K ({config['final_parent_top_k']}) không được lớn hơn PARENT_CANDIDATES ({config['parent_candidates']}).")
    if config["total_context_max_chars"] < config["parent_max_chars"]:
        raise ValueError("TOTAL_CONTEXT_MAX_CHARS phải lớn hơn hoặc bằng PARENT_MAX_CHARS.")
    if not config["embedding_model"] or not config["generation_model"] or not config["reranker_model"]:
        raise ValueError("Tên mô hình (Embedding, Generation, Reranker) không được rỗng.")


def extract_chunk_sequence_number(chunk_id: str) -> int:
    """
    Trích xuất phần số cuối cùng của chunk_id để sắp xếp số học thay vì sắp xếp từ vựng.
    Ví dụ: 'hierarchical_TT_02_2023_NHNN_002' -> 2
    """
    match = re.search(r"(\d+)$", str(chunk_id))
    return int(match.group(1)) if match else 0


def sanitize_key(key: str) -> str:
    """
    Chuẩn hóa chuỗi ký tự phục vụ làm ID định danh an toàn.
    """
    s = re.sub(r"[^\w\.-]", "_", key.strip())
    return re.sub(r"_+", "_", s)


def parse_heading_from_text(text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Phân tích các heading Chương / Điều ở ĐẦU CHUNK TEXT.
    Chỉ công nhận heading nếu xuất hiện ở đầu dòng chính thức (không phải trích dẫn giữa câu).
    """
    chapter_found = None
    article_found = None
    warnings = []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None, None, warnings

    # Match Chapter heading at top lines
    for line in lines[:3]:
        m_chap = re.match(r"^(?:#+\s*)?(Chương\s+[0-9IVXLCDMivxlcdm]+(?:\.|\:|\s+[^\n]+)?)", line, re.IGNORECASE)
        if m_chap and not chapter_found:
            chapter_found = m_chap.group(1).strip()
            break

    # Match Article heading at top lines
    competing_articles = []
    for line in lines[:3]:
        m_art = re.match(r"^(?:#+\s*)?(Điều\s+\d+(?:\.[^\n]*)?)", line, re.IGNORECASE)
        if m_art:
            art_str = m_art.group(1).strip()
            if art_str not in competing_articles:
                competing_articles.append(art_str)

    if len(competing_articles) == 1:
        article_found = competing_articles[0]
    elif len(competing_articles) > 1:
        article_found = competing_articles[0]
        warnings.append(f"multiple_competing_headings: Tìm thấy nhiều heading Điều ở đầu chunk: {competing_articles}")

    return chapter_found, article_found, warnings


def resolve_hierarchy_for_chunks(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Phân giải mối quan hệ thứ bậc (Chương/Điều/Khoản/Điểm) cho danh sách child chunks.
    Sử dụng thứ tự ưu tiên: Metadata -> Heading Inferred -> Carried Forward -> Document Fallback.
    """
    # 1. Kiểm tra tính hợp lệ và lọc strategy hierarchical
    filtered_chunks = []
    seen_ids = set()
    for c in raw_chunks:
        if c.get("strategy") != "hierarchical":
            continue
        cid = c.get("chunk_id")
        if not cid:
            raise ValueError("Phát hiện record thiếu 'chunk_id'.")
        if cid in seen_ids:
            raise ValueError(f"Phát hiện trùng lặp chunk_id: '{cid}'.")
        seen_ids.add(cid)

        # Validate required fields
        for field in ["source", "page_start", "page_end", "text"]:
            if field not in c:
                raise ValueError(f"Record '{cid}' thiếu trường bắt buộc '{field}'.")
        
        p_start = c["page_start"]
        p_end = c["page_end"]
        if not isinstance(p_start, int) or not isinstance(p_end, int) or p_start < 1 or p_start > p_end:
            raise ValueError(f"Record '{cid}' có page range không hợp lệ: page_start={p_start}, page_end={p_end}.")
        
        if not str(c["text"]).strip():
            raise ValueError(f"Record '{cid}' có nội dung text rỗng.")

        filtered_chunks.append(c)

    # 2. Nhóm theo source và sắp xếp theo số thứ tự chuỗi
    grouped = {}
    for c in filtered_chunks:
        grouped.setdefault(c["source"], []).append(c)

    resolved_children = []

    for source, source_chunks in grouped.items():
        # Sắp xếp số học theo sequence number cuối chunk_id
        source_chunks.sort(key=lambda x: extract_chunk_sequence_number(x["chunk_id"]))

        last_chapter = None
        last_article = None

        for c in source_chunks:
            cid = c["chunk_id"]
            meta = c.get("metadata", {})
            st = meta.get("structure") if isinstance(meta, dict) and isinstance(meta.get("structure"), dict) else None

            ch_label = None
            art_label = None
            cl_label = None
            pt_label = None
            method = "document_fallback"
            ambiguous = False
            warnings = []

            # 1. Ưu tiên Metadata nếu có
            if st and (st.get("article") or st.get("chapter")):
                ch_label = st.get("chapter")
                art_label = st.get("article")
                cl_label = st.get("clause")
                pt_label = st.get("point")
                method = "metadata"

            # 2. Inferred Heading từ dòng đầu text
            inf_chap, inf_art, inf_warns = parse_heading_from_text(c["text"])
            warnings.extend(inf_warns)
            if inf_warns:
                ambiguous = True

            if method == "metadata":
                # Kiểm tra xung đột giữa Metadata và Heading Inferred
                if inf_art and art_label and (inf_art.split(".")[0].strip() != art_label.split(".")[0].strip()):
                    ambiguous = True
                    warnings.append(f"metadata_heading_conflict: Metadata ({art_label}) xung đột với Text Heading ({inf_art})")
            elif inf_art or inf_chap:
                method = "heading_inferred"
                if inf_art:
                    art_label = inf_art
                if inf_chap:
                    ch_label = inf_chap

            # 3. Carry Forward trong cùng source
            if not art_label:
                if last_article:
                    art_label = last_article
                    if last_chapter and not ch_label:
                        ch_label = last_chapter
                    method = "carried_forward"

            # 4. Fallback cấp Document nếu vẫn không tìm thấy Article
            if not art_label:
                method = "document_fallback"

            # Cập nhật trạng thái vết cho các chunk sau trong cùng source
            if art_label:
                last_article = art_label
            if ch_label:
                last_chapter = ch_label

            resolved_rec = {
                "child_id": cid,
                "parent_id": "",  # Sẽ gán sau khi build parent windows
                "source": source,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "text": c["text"],
                "structural_path": {
                    "chapter": ch_label,
                    "article": art_label,
                    "clause": cl_label,
                    "point": pt_label,
                },
                "resolution_method": method,
                "ambiguous": ambiguous,
                "warnings": warnings,
            }
            resolved_children.append(resolved_rec)

    return resolved_children


def build_parent_documents(resolved_children: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Gom nhóm các child chunks thành các Parent Article Windows thỏa mãn PARENT_MAX_CHARS.
    Gán parent_id chính thức cho từng child chunk.
    """
    if config is None:
        config = load_config()

    max_chars = config["parent_max_chars"]

    # Nhóm theo (source, resolved_article_key)
    groups = {}
    for c in resolved_children:
        source = c["source"]
        art = c["structural_path"]["article"]
        art_key = art if art else "doc_fallback"
        key = (source, art_key)
        groups.setdefault(key, []).append(c)

    parents = []

    for (source, art_key), children_in_group in groups.items():
        current_window_children = []
        current_char_count = 0
        window_index = 1

        for child in children_in_group:
            child_len = len(child["text"])

            # Xử lý trường hợp child đơn lẻ vượt PARENT_MAX_CHARS
            if child_len > max_chars:
                # Đóng window hiện tại nếu có
                if current_window_children:
                    parent_doc = _create_parent_window(
                        source=source,
                        art_key=art_key,
                        window_index=window_index,
                        children=current_window_children,
                    )
                    parents.append(parent_doc)
                    window_index += 1
                    current_window_children = []
                    current_char_count = 0

                # Tạo parent window riêng cho child quá khổ
                oversized_parent = _create_parent_window(
                    source=source,
                    art_key=art_key,
                    window_index=window_index,
                    children=[child],
                    extra_warning=f"oversized_single_child: Child chunk '{child['child_id']}' dài {child_len} chars vượt PARENT_MAX_CHARS ({max_chars}).",
                )
                parents.append(oversized_parent)
                window_index += 1
                continue

            # Thêm child vào window hiện tại nếu không vượt quá giới hạn
            if current_char_count + child_len <= max_chars:
                current_window_children.append(child)
                current_char_count += child_len
            else:
                # Ngắt window và tạo window mới
                parent_doc = _create_parent_window(
                    source=source,
                    art_key=art_key,
                    window_index=window_index,
                    children=current_window_children,
                )
                parents.append(parent_doc)
                window_index += 1

                current_window_children = [child]
                current_char_count = child_len

        # Đóng window cuối cùng
        if current_window_children:
            parent_doc = _create_parent_window(
                source=source,
                art_key=art_key,
                window_index=window_index,
                children=current_window_children,
            )
            parents.append(parent_doc)

    return parents


def _create_parent_window(
    source: str,
    art_key: str,
    window_index: int,
    children: List[Dict[str, Any]],
    extra_warning: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hàm bổ trợ khởi tạo một bản ghi ParentDocument hoàn chỉnh và gán parent_id cho children.
    """
    safe_src = sanitize_key(Path(source).stem)
    safe_art = sanitize_key(art_key)
    parent_id = f"parent_{safe_src}_{safe_art}_w{window_index}"

    # Ghép văn bản không trùng lặp theo thứ tự
    parent_text = "\n\n".join([c["text"] for c in children])
    char_count = len(parent_text)
    p_start = min(c["page_start"] for c in children)
    p_end = max(c["page_end"] for c in children)
    ambiguous_count = sum(1 for c in children if c.get("ambiguous", False))

    warnings = []
    if extra_warning:
        warnings.append(extra_warning)

    # Gán parent_id ngược lại cho từng child
    for c in children:
        c["parent_id"] = parent_id

    return {
        "parent_id": parent_id,
        "source": source,
        "page_start": p_start,
        "page_end": p_end,
        "article_key": art_key,
        "window_index": window_index,
        "child_ids": [c["child_id"] for c in children],
        "text": parent_text,
        "char_count": char_count,
        "ambiguous_child_count": ambiguous_count,
        "warnings": warnings,
    }


def build_hierarchy_store(
    input_path: Path = None,
    output_dir: Path = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Thực thi quy trình build Hierarchy Registry & Parent Store hoàn chỉnh.
    Ghi atomically bằng tệp tạm và replace để tránh hư hỏng dữ liệu.
    """
    if config is None:
        config = load_config()

    if input_path is None:
        input_path = DEFAULT_INPUT_DIR

    if output_dir is None:
        output_dir = HIERARCHY_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục/tệp input chunks: {input_path}")

    # Đọc raw chunks
    files_to_read = []
    if input_path.is_file():
        files_to_read = [input_path]
    else:
        files_to_read = sorted(list(input_path.glob("*_chunks.json")))

    if not files_to_read:
        raise FileNotFoundError(f"Không tìm thấy tệp *_chunks.json nào tại: {input_path}")

    raw_chunks = []
    input_fingerprints = []

    for f in files_to_read:
        file_bytes = f.read_bytes()
        sha = hashlib.sha256(file_bytes).hexdigest()
        input_fingerprints.append({"filename": f.name, "sha256": sha})

        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            if isinstance(data, dict) and "hierarchical_chunks" in data:
                raw_chunks.extend(data["hierarchical_chunks"])
            elif isinstance(data, list):
                raw_chunks.extend([c for c in data if isinstance(c, dict) and c.get("strategy") == "hierarchical"])

    # 1. Resolve Children Hierarchy
    resolved_children = resolve_hierarchy_for_chunks(raw_chunks)

    # 2. Build Parent Documents
    parent_docs = build_parent_documents(resolved_children, config)

    # 3. Tổng hợp Manifest Statistics
    total_children = len(resolved_children)
    total_parents = len(parent_docs)
    ambiguous_children = sum(1 for c in resolved_children if c["ambiguous"])
    total_warnings = sum(len(c["warnings"]) for c in resolved_children) + sum(len(p["warnings"]) for p in parent_docs)

    manifest = {
        "schema_version": "1.0",
        "input_files_fingerprint": input_fingerprints,
        "strategy": "hierarchical",
        "config_identity": {
            "parent_max_chars": config["parent_max_chars"],
            "embedding_model": config["embedding_model"],
            "embedding_dim": config["embedding_dim"],
        },
        "total_children_count": total_children,
        "total_parents_count": total_parents,
        "ambiguous_children_count": ambiguous_children,
        "total_warnings_count": total_warnings,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 4. Ghi Atomic qua tệp tạm (.tmp)
    tmp_children = output_dir / ".children.json.tmp"
    tmp_parents = output_dir / ".parents.json.tmp"
    tmp_manifest = output_dir / ".manifest.json.tmp"

    with open(tmp_children, "w", encoding="utf-8") as f:
        json.dump(resolved_children, f, ensure_ascii=False, indent=2)

    with open(tmp_parents, "w", encoding="utf-8") as f:
        json.dump(parent_docs, f, ensure_ascii=False, indent=2)

    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Replace nguyên tử
    os.replace(tmp_children, output_dir / "children.json")
    os.replace(tmp_parents, output_dir / "parents.json")
    os.replace(tmp_manifest, output_dir / "manifest.json")

    return {
        "status": "success",
        "children_count": total_children,
        "parents_count": total_parents,
        "ambiguous_count": ambiguous_children,
        "warnings_count": total_warnings,
        "output_dir": str(output_dir),
    }


def get_hierarchy_status(output_dir: Path = None) -> Dict[str, Any]:
    """
    Kiểm tra trạng thái Hierarchy Store theo cơ chế READ-ONLY.
    Tuyệt đối không mkdir, không ghi file, không sửa timestamp.
    """
    if output_dir is None:
        output_dir = HIERARCHY_DIR

    manifest_p = Path(output_dir) / "manifest.json"
    children_p = Path(output_dir) / "children.json"
    parents_p = Path(output_dir) / "parents.json"

    if not manifest_p.exists() or not children_p.exists() or not parents_p.exists():
        return {
            "store_exists": False,
            "children_count": 0,
            "parents_count": 0,
            "ambiguous_count": 0,
            "warnings_count": 0,
            "manifest_file": str(manifest_p),
        }

    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        return {
            "store_exists": True,
            "schema_version": manifest_data.get("schema_version", "1.0"),
            "strategy": manifest_data.get("strategy", "hierarchical"),
            "children_count": manifest_data.get("total_children_count", 0),
            "parents_count": manifest_data.get("total_parents_count", 0),
            "ambiguous_count": manifest_data.get("ambiguous_children_count", 0),
            "warnings_count": manifest_data.get("total_warnings_count", 0),
            "build_timestamp": manifest_data.get("build_timestamp", ""),
            "config_identity": manifest_data.get("config_identity", {}),
            "manifest_file": str(manifest_p),
        }
    except Exception as e:
        return {
            "store_exists": False,
            "error": f"Lỗi đọc manifest.json: {e}",
            "manifest_file": str(manifest_p),
        }


# -----------------------------------------------------------------------------
# MULTI-QUERY EXPANSION GENERATOR
# -----------------------------------------------------------------------------
_MULTI_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


def normalize_query_text(text: str) -> str:
    """Chuẩn hóa NFC và xóa khoảng trắng thừa."""
    import unicodedata
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)


def compute_query_dedup_key(text: str) -> str:
    """Tính dedup key (NFC + casefold + xóa khoảng trắng và dấu câu)."""
    norm = normalize_query_text(text).casefold()
    clean = re.sub(r"[^\w]", "", norm)
    return clean


def extract_legal_references(text: str) -> List[str]:
    """Trích xuất các tham chiếu pháp lý như 'Điều 7', 'Khoản 2', 'Thông tư 39/2016'."""
    refs = []
    matches = re.findall(r"(?:Điều|Khoản|Điểm|Thông tư|Nghị định|Luật)\s+\d+(?:/\d+/[A-Z-]+)?", text, re.IGNORECASE)
    for m in matches:
        m_str = m.strip()
        if m_str not in refs:
            refs.append(m_str)
    return refs


def generate_multi_queries(
    question: str,
    config: Dict[str, Any] = None,
    query_generator_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Sinh các biến thể truy vấn (Multi-Query Expansion Generator).
    Trả về Dict gồm Q0 (nguyên văn gốc) và Q1..Qn (các biến thể).
    """
    import time
    if config is None:
        config = load_config()

    t_start = time.perf_counter()

    # 1. Standardize Q0
    q0_clean = normalize_query_text(question)
    if not q0_clean:
        return {
            "original_question": "",
            "queries": [],
            "model": config["generation_model"],
            "generation_latency_ms": 0.0,
            "status": "query_generation_unavailable",
            "warnings": ["Câu hỏi gốc rỗng."],
            "cache_hit": False,
        }

    max_chars = config["multi_query_max_chars"]
    count_requested = config["multi_query_count"]
    model_name = config["generation_model"]
    temp = config["multi_query_temperature"]

    # 2. Check Cache
    cache_raw = f"{q0_clean}_{model_name}_{temp}_{count_requested}"
    cache_key = hashlib.sha256(cache_raw.encode("utf-8")).hexdigest()

    if query_generator_fn is None and cache_key in _MULTI_QUERY_CACHE:
        cached = dict(_MULTI_QUERY_CACHE[cache_key])
        cached["cache_hit"] = True
        return cached

    q0_obj = {
        "query_id": "Q0",
        "text": q0_clean,
        "origin": "original",
        "focus": "original_intent",
    }

    warnings = []

    # 3. Generate Variants
    raw_variants = []
    if query_generator_fn is not None:
        try:
            raw_variants = query_generator_fn(q0_clean, count_requested)
        except Exception as e:
            t_end = time.perf_counter()
            return {
                "original_question": q0_clean,
                "queries": [q0_obj],
                "model": model_name,
                "generation_latency_ms": round((t_end - t_start) * 1000, 2),
                "status": "query_generation_unavailable",
                "warnings": [f"Lỗi query_generator_fn: {e}"],
                "cache_hit": False,
            }
    else:
        api_key = config.get("api_key")
        if not api_key:
            t_end = time.perf_counter()
            return {
                "original_question": q0_clean,
                "queries": [q0_obj],
                "model": model_name,
                "generation_latency_ms": round((t_end - t_start) * 1000, 2),
                "status": "query_generation_unavailable",
                "warnings": ["Lỗi: GEMINI_API_KEY chưa được cấu hình. Không thể sinh multi-queries."],
                "cache_hit": False,
            }

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = (
                f"Bạn là chuyên gia tra cứu văn bản pháp luật ngân hàng Việt Nam.\n"
                f"Hãy tạo đúng {count_requested} câu hỏi tìm kiếm biến thể (search query variants) cho câu hỏi gốc dưới đây:\n"
                f"Câu hỏi gốc: \"{q0_clean}\"\n\n"
                f"Yêu cầu bắt buộc:\n"
                f"1. Tạo đúng {count_requested} câu hỏi tra cứu bằng tiếng Việt ngắn gọn, súc tích.\n"
                f"2. Bao gồm các khía cạnh: (a) thuật ngữ pháp lý chính xác (exact_legal_terms), (b) cách diễn đạt tương đương (paraphrase), (c) khía cạnh còn thiếu (missing_aspect).\n"
                f"3. NẾU câu hỏi gốc có đề cập số Điều, Khoản, Điểm, số Thông tư/Nghị định thì BẮT BUỘC giữ nguyên tham chiếu đó trong ít nhất 1 biến thể.\n"
                f"4. KHÔNG tự bịa ra số Điều/Khoản mới không có trong câu hỏi gốc.\n"
                f"5. CHỈ trả về dữ liệu định dạng JSON theo đúng schema yêu cầu. KHÔNG trả lời câu hỏi và KHÔNG thêm văn bản ngoài JSON.\n"
            )

            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "queries": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "text": {"type": "STRING"},
                                "focus": {"type": "STRING"}
                            },
                            "required": ["text", "focus"]
                        }
                    }
                },
                "required": ["queries"]
            }

            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temp,
            )

            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )

            if not res or not hasattr(res, "text") or not res.text:
                raise ValueError("Gemini API trả về câu phản hồi rỗng.")

            resp_json = json.loads(res.text)
            raw_variants = resp_json.get("queries", [])
        except Exception as e:
            t_end = time.perf_counter()
            return {
                "original_question": q0_clean,
                "queries": [q0_obj],
                "model": model_name,
                "generation_latency_ms": round((t_end - t_start) * 1000, 2),
                "status": "query_generation_unavailable",
                "warnings": [f"Lỗi Gemini API Multi-Query Generation: {e}"],
                "cache_hit": False,
            }

    # 4. Process and Sanitize Generated Variants
    valid_queries = [q0_obj]
    seen_dedup_keys = {compute_query_dedup_key(q0_clean)}

    dropped_duplicate_count = 0
    variant_idx = 1

    for v in raw_variants:
        if len(valid_queries) - 1 >= count_requested:
            break

        v_text = v.get("text", "") if isinstance(v, dict) else str(v)
        v_focus = v.get("focus", "paraphrase") if isinstance(v, dict) else "paraphrase"

        norm_v = normalize_query_text(v_text)
        if not norm_v:
            continue

        if len(norm_v) > max_chars:
            norm_v = norm_v[:max_chars].strip()
            warnings.append(f"Query variant bị cắt bớt do vượt {max_chars} chars.")

        dedup_k = compute_query_dedup_key(norm_v)
        if dedup_k in seen_dedup_keys:
            dropped_duplicate_count += 1
            continue

        # Check for fabricated article numbers
        inv_arts = re.findall(r"Điều\s+(\d+)", norm_v, re.IGNORECASE)
        q0_arts = re.findall(r"Điều\s+(\d+)", q0_clean, re.IGNORECASE)
        fabricated = [art for art in inv_arts if art not in q0_arts]
        if fabricated and q0_arts:
            warnings.append(f"Bỏ qua query variant do bịa số Điều mới ({fabricated}) không có trong câu hỏi gốc.")
            continue

        seen_dedup_keys.add(dedup_k)
        qid = f"Q{variant_idx}"
        valid_queries.append({
            "query_id": qid,
            "text": norm_v,
            "origin": "generated",
            "focus": v_focus,
        })
        variant_idx += 1

    if dropped_duplicate_count > 0:
        warnings.append(f"dropped_duplicate_count: {dropped_duplicate_count}")

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 2)

    result_set = {
        "original_question": q0_clean,
        "queries": valid_queries,
        "model": model_name,
        "generation_latency_ms": latency_ms,
        "status": "ready",
        "warnings": warnings,
        "cache_hit": False,
    }

    # Save to Cache
    if query_generator_fn is None:
        _MULTI_QUERY_CACHE[cache_key] = dict(result_set)
    return result_set


# -----------------------------------------------------------------------------
# CROSS-QUERY RRF FUSION & MULTI-QUERY CHILD SEARCH
# -----------------------------------------------------------------------------
def cross_query_rrf_fusion(
    per_query_results: Dict[str, Dict[str, Any]],
    query_set: List[Dict[str, Any]],
    config: Dict[str, Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """
    Tầng 2: Dung hợp kết quả tìm kiếm đa nhánh Cross-Query RRF Fusion.
    """
    if config is None:
        config = load_config()

    rrf_k = config["multi_query_rrf_k"]
    w_orig = config["multi_query_original_weight"]
    w_var = config["multi_query_variant_weight"]

    # Build query weight map
    query_weights = {}
    for q_item in query_set:
        qid = q_item["query_id"]
        if q_item.get("origin") == "original" or qid == "Q0":
            query_weights[qid] = w_orig
        else:
            query_weights[qid] = w_var

    merged_map = {}

    for qid, q_res in per_query_results.items():
        if q_res.get("status") != "success":
            continue
        hits = q_res.get("results", [])
        w_q = query_weights.get(qid, 1.0)

        for rank, child in enumerate(hits, start=1):
            cid = child.get("child_id") or child.get("chunk_id")
            if not cid:
                continue

            score_contrib = w_q / (rrf_k + rank)

            if cid not in merged_map:
                merged_map[cid] = {
                    "child_id": cid,
                    "text": child["text"],
                    "source": child["source"],
                    "page_start": child["page_start"],
                    "page_end": child["page_end"],
                    "multi_query_rrf_score": score_contrib,
                    "support_query_ids": [qid],
                    "per_query_ranks": {qid: rank},
                    "per_query_trace": {
                        qid: {
                            "bm25_rank": child.get("bm25_rank"),
                            "semantic_rank": child.get("semantic_rank"),
                            "inner_rrf_rank": child.get("fused_rank", rank),
                        }
                    },
                }
            else:
                existing = merged_map[cid]
                # Metadata mismatch check
                if existing["source"] != child["source"] or existing["text"] != child["text"]:
                    raise ValueError(f"Metadata mismatch cho child_id '{cid}' giữa các queries.")

                existing["multi_query_rrf_score"] += score_contrib
                if qid not in existing["support_query_ids"]:
                    existing["support_query_ids"].append(qid)
                existing["per_query_ranks"][qid] = rank
                existing["per_query_trace"][qid] = {
                    "bm25_rank": child.get("bm25_rank"),
                    "semantic_rank": child.get("semantic_rank"),
                    "inner_rrf_rank": child.get("fused_rank", rank),
                }

    merged_list = list(merged_map.values())
    for item in merged_list:
        item["support_query_count"] = len(item["support_query_ids"])
        item["best_query_rank"] = min(item["per_query_ranks"].values())
        item["multi_query_rrf_score"] = round(item["multi_query_rrf_score"], 6)

    # Sort: multi_query_rrf_score (desc) -> support_query_count (desc) -> best_query_rank (asc) -> child_id (asc)
    merged_list.sort(
        key=lambda x: (
            -x["multi_query_rrf_score"],
            -x["support_query_count"],
            x["best_query_rank"],
            x["child_id"],
        )
    )

    for rank, item in enumerate(merged_list, start=1):
        item["multi_query_rank"] = rank

    overlap_dist = {}
    for item in merged_list:
        c = item["support_query_count"]
        overlap_dist[c] = overlap_dist.get(c, 0) + 1

    return merged_list, overlap_dist


def search_multi_query_child(
    question: str,
    top_k: Optional[int] = None,
    strategy: str = "hierarchical",
    config: Dict[str, Any] = None,
    input_dir: Optional[Path] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Thực hiện quy trình Per-Query Hybrid Retrieval + Cross-Query RRF Fusion hoàn chỉnh.
    """
    import time
    if config is None:
        config = load_config()

    t_start = time.perf_counter()
    per_candidate_k = top_k if top_k is not None else config["per_query_candidates"]

    # 1. Multi-Query Expansion
    t_gen_start = time.perf_counter()
    expansion_res = generate_multi_queries(
        question=question,
        config=config,
        query_generator_fn=query_generator_fn,
    )
    t_gen_end = time.perf_counter()
    gen_ms = round((t_gen_end - t_gen_start) * 1000, 2)

    query_set = expansion_res.get("queries", [])
    if not query_set:
        raise ValueError("Multi-query expansion không trả về query nào.")

    # 2. Per-Query Hybrid Retrieval
    t_ret_start = time.perf_counter()
    per_query_results = {}
    per_query_stats = []
    executed_count = 0
    failed_count = 0
    warnings = list(expansion_res.get("warnings", []))

    for q_item in query_set:
        qid = q_item["query_id"]
        q_text = q_item["text"]
        t_q_start = time.perf_counter()

        try:
            executed_count += 1
            if hybrid_retriever_fn is not None:
                q_hits = hybrid_retriever_fn(q_text, per_candidate_k)
            else:
                import advanced_rag
                try:
                    trace = advanced_rag.search_hybrid(
                        question=q_text,
                        top_k=per_candidate_k,
                        strategy=strategy,
                        input_dir=input_dir,
                    )
                    q_hits = trace.get("results", [])
                except Exception as ex:
                    # Fallback to BM25 search if Chroma collection is missing or fails
                    chunks_cache, _ = advanced_rag.rag.load_chunks(input_path=input_dir, target_strategy=strategy)
                    trace = advanced_rag.search_bm25(
                        question=q_text,
                        chunks=chunks_cache,
                        candidate_k=per_candidate_k,
                    )
                    q_hits = trace if isinstance(trace, list) else trace.get("results", [])
                    warnings.append(f"Chroma Vector DB chưa nạp collection ({ex}). Hệ thống tự động fallback sang BM25 search cho query [{qid}].")

            t_q_end = time.perf_counter()
            q_ms = round((t_q_end - t_q_start) * 1000, 2)

            per_query_results[qid] = {
                "status": "success",
                "results": q_hits,
                "latency_ms": q_ms,
            }
            per_query_stats.append({
                "query_id": qid,
                "text": q_text,
                "candidate_count": len(q_hits),
                "latency_ms": q_ms,
                "status": "success",
            })
        except Exception as e:
            t_q_end = time.perf_counter()
            q_ms = round((t_q_end - t_q_start) * 1000, 2)
            failed_count += 1

            if qid == "Q0":
                raise RuntimeError(f"Lỗi truy xuất Per-Query Hybrid cho câu hỏi gốc Q0: {e}")

            warnings.append(f"Lỗi truy xuất query {qid} ({q_text}): {e}")
            per_query_results[qid] = {
                "status": "failed",
                "error": str(e),
                "results": [],
                "latency_ms": q_ms,
            }
            per_query_stats.append({
                "query_id": qid,
                "text": q_text,
                "candidate_count": 0,
                "latency_ms": q_ms,
                "status": "failed",
                "error": str(e),
            })

    t_ret_end = time.perf_counter()
    ret_ms = round((t_ret_end - t_ret_start) * 1000, 2)

    # 3. Cross-Query RRF Fusion
    t_fusion_start = time.perf_counter()
    fused_child_hits, overlap_dist = cross_query_rrf_fusion(
        per_query_results=per_query_results,
        query_set=query_set,
        config=config,
    )
    t_fusion_end = time.perf_counter()
    fusion_ms = round((t_fusion_end - t_fusion_start) * 1000, 2)
    t_end = time.perf_counter()
    total_ms = round((t_end - t_start) * 1000, 2)

    status_code = "ready"
    if failed_count > 0 or expansion_res.get("status") != "ready":
        status_code = "multi_query_partial"

    return {
        "question": expansion_res.get("original_question", question),
        "strategy": strategy,
        "status": status_code,
        "query_counts": {
            "requested": config["multi_query_count"],
            "valid": len(query_set),
            "executed": executed_count,
            "failed": failed_count,
        },
        "latency_ms": {
            "multi_query_gen_ms": gen_ms,
            "per_query_retrieval_ms": ret_ms,
            "fusion_ms": fusion_ms,
            "total_ms": total_ms,
        },
        "per_query_stats": per_query_stats,
        "union_child_count": len(fused_child_hits),
        "overlap_distribution": overlap_dist,
        "warnings": warnings,
        "results": fused_child_hits,
    }


# -----------------------------------------------------------------------------
# CLI INTERFACE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI Quản lý Hierarchy Registry & Parent Store (Buổi 09)")
    subparsers = parser.add_subparsers(dest="command", help="Danh sách subcommands")

    subparsers.add_parser("hierarchy-audit", help="Kiểm tra read-only phân giải hierarchy của input chunks")
    subparsers.add_parser("build-hierarchy", help="Xây dựng Hierarchy Registry & Parent Store")
    subparsers.add_parser("hierarchy-status", help="Kiểm tra trạng thái read-only của Store hiện tại")

    cmd_expand = subparsers.add_parser("expand-query", help="Sinh và hiển thị các biến thể câu hỏi (Multi-Query Expansion)")
    cmd_expand.add_argument("--question", type=str, default="Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?", help="Nội dung câu hỏi gốc cần sinh biến thể")

    cmd_mc = subparsers.add_parser("multi-child", help="Thực thi Multi-Query Per-Query Hybrid + Cross-Query RRF Child Search")
    cmd_mc.add_argument("--question", type=str, default="Điều kiện vay vốn và các trường hợp không được cho vay là gì?", help="Câu hỏi gốc")

    args = parser.parse_args()

    if args.command == "hierarchy-audit":
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO AUDIT HIERARCHY RESOLUTION (Read-only)")
        print("============================================================")
        res = build_hierarchy_store(config=config)
        
        # Đọc chi tiết để hiển thị statistics
        with open(HIERARCHY_DIR / "children.json", "r", encoding="utf-8") as f:
            children = json.load(f)
        with open(HIERARCHY_DIR / "parents.json", "r", encoding="utf-8") as f:
            parents = json.load(f)

        method_counts = {}
        for c in children:
            m = c["resolution_method"]
            method_counts[m] = method_counts.get(m, 0) + 1

        print(f" Tổng số Child Chunks        : {len(children)}")
        print(f" Phân bổ Resolution Methods  : {method_counts}")
        print(f" Số Child bị Ambiguous      : {sum(1 for c in children if c['ambiguous'])}")
        print(f" Tổng số Parent Documents    : {len(parents)}")
        
        parent_lens = [p["char_count"] for p in parents]
        if parent_lens:
            print(f" Độ dài Parent Text (chars)  : Min={min(parent_lens)}, Mean={sum(parent_lens)/len(parent_lens):.1f}, Max={max(parent_lens)}")
        print("============================================================\n")

    elif args.command == "build-hierarchy":
        config = load_config()
        print("\n⚡ Đang thực thi build Hierarchy Registry & Parent Store...")
        res = build_hierarchy_store(config=config)
        print(f"✅ Xây dựng thành công! Store lưu tại: {res['output_dir']}")
        print(f"   Children: {res['children_count']} | Parents: {res['parents_count']} | Ambiguous: {res['ambiguous_count']} | Warnings: {res['warnings_count']}\n")

    elif args.command == "hierarchy-status":
        st_info = get_hierarchy_status()
        print("\n============================================================")
        print("  TRẠNG THÁI HIERARCHY STORE (Read-only)")
        print("============================================================")
        print(f" Store Tồn Tại         : {st_info['store_exists']}")
        if st_info['store_exists']:
            print(f" Schema Version        : {st_info.get('schema_version')}")
            print(f" Total Children Count  : {st_info.get('children_count')}")
            print(f" Total Parents Count   : {st_info.get('parents_count')}")
            print(f" Ambiguous Children    : {st_info.get('ambiguous_count')}")
            print(f" Warnings Count        : {st_info.get('warnings_count')}")
            print(f" Build Timestamp (UTC) : {st_info.get('build_timestamp')}")
        print("============================================================\n")

    elif args.command == "expand-query":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO MULTI-QUERY EXPANSION (Command: expand-query)")
        print("============================================================")
        print(f" Câu hỏi gốc (Q0) : {q}")
        res = generate_multi_queries(question=q, config=config)
        print(f" Status            : {res['status']}")
        print(f" Model             : {res['model']}")
        print(f" Latency           : {res['generation_latency_ms']} ms")
        print(f" Cache Hit         : {res.get('cache_hit', False)}")
        print("------------------------------------------------------------")
        for q_item in res["queries"]:
            print(f" [{q_item['query_id']}] ({q_item['origin']}/{q_item['focus']}): {q_item['text']}")
        if res.get("warnings"):
            print(" Warnings          :", res["warnings"])
        print("============================================================\n")

    elif args.command == "multi-child":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO MULTI-QUERY CHILD RETRIEVAL (Command: multi-child)")
        print("============================================================")
        print(f" Câu hỏi gốc      : {q}")
        res = search_multi_query_child(question=q, config=config)
        print(f" Status           : {res['status']}")
        print(f" Query Counts     : {res['query_counts']}")
        print(f" Latency Trace    : Gen={res['latency_ms']['multi_query_gen_ms']}ms | Ret={res['latency_ms']['per_query_retrieval_ms']}ms | Fusion={res['latency_ms']['fusion_ms']}ms | Total={res['latency_ms']['total_ms']}ms")
        print(f" Union Child Hits : {res['union_child_count']}")
        print(f" Overlap Dist     : {res['overlap_distribution']}")
        print("------------------------------------------------------------")
        print(" Top Fused Child Hits:")
        for child in res["results"][:10]:
            print(f" [{child['multi_query_rank']:02d}] MQ-RRF: {child['multi_query_rrf_score']:.6f} | Support: {child['support_query_count']} ({child['support_query_ids']}) | Ranks: {child['per_query_ranks']} | ID: {child['child_id']}")
        if res.get("warnings"):
            print(" Warnings         :", res["warnings"])
        print("============================================================\n")

    else:
        parser.print_help()


# -----------------------------------------------------------------------------
# PARENT DOCUMENT EXPANSION & PARENT AGGREGATION
# -----------------------------------------------------------------------------
def load_hierarchy_store_data(hierarchy_dir: Path = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Tải dữ liệu children.json, parents.json và manifest.json từ Hierarchy Store.
    """
    if hierarchy_dir is None:
        hierarchy_dir = HIERARCHY_DIR

    hierarchy_dir = Path(hierarchy_dir)
    c_file = hierarchy_dir / "children.json"
    p_file = hierarchy_dir / "parents.json"
    m_file = hierarchy_dir / "manifest.json"

    if not c_file.exists() or not p_file.exists() or not m_file.exists():
        raise FileNotFoundError(f"Hierarchy store không sẵn sàng tại: {hierarchy_dir}")

    with open(c_file, "r", encoding="utf-8") as f:
        children_list = json.load(f)
    with open(p_file, "r", encoding="utf-8") as f:
        parents_list = json.load(f)
    with open(m_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    children_dict = {c["child_id"]: c for c in children_list}
    parents_dict = {p["parent_id"]: p for p in parents_list}

    return children_dict, parents_dict, manifest_data


def expand_children_to_parents(
    child_hits: List[Dict[str, Any]],
    children_dict: Dict[str, Dict[str, Any]],
    parents_dict: Dict[str, Dict[str, Any]],
    config: Dict[str, Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Mở rộng Child Hits thành Parent Documents, thực hiện Parent Aggregation và Context Budgeting.
    """
    import time
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    score_child_limit = config["parent_score_child_limit"]
    parent_rrf_k = config["parent_rrf_k"]
    parent_candidates_limit = config["parent_candidates"]
    total_context_max_chars = config["total_context_max_chars"]

    # 1. Group child hits by parent_id
    parent_groups = {}
    mapping_table = []
    child_chars_total = 0

    for child_hit in child_hits:
        cid = child_hit.get("child_id") or child_hit.get("chunk_id")
        if not cid:
            continue

        child_chars_total += len(child_hit.get("text", ""))

        if cid not in children_dict:
            raise KeyError(f"Child_id '{cid}' không tìm thấy trong Hierarchy Children Registry.")

        reg_child = children_dict[cid]
        pid = reg_child.get("parent_id")
        if not pid or pid not in parents_dict:
            raise KeyError(f"Parent_id '{pid}' của child '{cid}' không tìm thấy trong Parent Store.")

        mq_rank = child_hit.get("multi_query_rank") or child_hit.get("fused_rank") or 1
        sup_queries = child_hit.get("support_query_ids", ["Q0"])

        mapping_table.append({
            "child_id": cid,
            "parent_id": pid,
            "multi_query_rank": mq_rank,
            "support_queries": sup_queries,
        })

        if pid not in parent_groups:
            parent_groups[pid] = {
                "parent_doc": parents_dict[pid],
                "structural_path": reg_child.get("structural_path", {}),
                "child_hits": [],
            }
        parent_groups[pid]["child_hits"].append(child_hit)

    # 2. Parent Aggregation Score
    aggregated_parents = []
    parent_score_components = []

    for pid, group in parent_groups.items():
        pdoc = group["parent_doc"]
        chits = group["child_hits"]

        # Sort children by multi_query_rank ascending (rank 1 is best)
        chits.sort(key=lambda c: c.get("multi_query_rank") or c.get("fused_rank") or 999)

        # Separate scoring vs supporting children
        scoring_chits = chits[:score_child_limit]
        scoring_ids = [c.get("child_id") or c.get("chunk_id") for c in scoring_chits]
        supporting_ids = [c.get("child_id") or c.get("chunk_id") for c in chits]
        anchor_id = scoring_ids[0] if scoring_ids else ""

        # Parent RRF Score calculation
        p_score = 0.0
        for c in scoring_chits:
            r = c.get("multi_query_rank") or c.get("fused_rank") or 1
            p_score += 1.0 / (parent_rrf_k + r)

        p_score = round(p_score, 6)

        # Collect unique support query_ids across all supporting children
        all_sup_queries = []
        for c in chits:
            for qid in c.get("support_query_ids", ["Q0"]):
                if qid not in all_sup_queries:
                    all_sup_queries.append(qid)

        best_c_rank = min(c.get("multi_query_rank") or c.get("fused_rank") or 999 for c in chits)

        parent_cand = {
            "parent_id": pid,
            "source": pdoc["source"],
            "page_start": pdoc["page_start"],
            "page_end": pdoc["page_end"],
            "structural_path": group["structural_path"],
            "article_key": pdoc.get("article_key", ""),
            "text": pdoc["text"],
            "parent_rrf_score": p_score,
            "anchor_child_id": anchor_id,
            "scoring_child_ids": scoring_ids,
            "supporting_child_ids": supporting_ids,
            "support_query_ids": all_sup_queries,
            "support_query_count": len(all_sup_queries),
            "best_child_rank": best_c_rank,
            "char_count": pdoc["char_count"],
            "ambiguous": pdoc.get("ambiguous_child_count", 0) > 0,
            "warnings": list(pdoc.get("warnings", [])),
        }
        aggregated_parents.append(parent_cand)
        parent_score_components.append({
            "parent_id": pid,
            "parent_rrf_score": p_score,
            "scoring_child_ids": scoring_ids,
        })

    # 3. Sort Parent Candidates: parent_rrf_score (desc) -> support_query_count (desc) -> best_child_rank (asc) -> parent_id (asc)
    aggregated_parents.sort(
        key=lambda x: (
            -x["parent_rrf_score"],
            -x["support_query_count"],
            x["best_child_rank"],
            x["parent_id"],
        )
    )

    for rank, p in enumerate(aggregated_parents, start=1):
        p["parent_rank"] = rank

    # 4. Candidate Limit & Context Budgeting
    dropped_parents = []
    
    candidates_within_limit = aggregated_parents[:parent_candidates_limit]
    for p in aggregated_parents[parent_candidates_limit:]:
        dropped_parents.append({"parent_id": p["parent_id"], "reason": "candidate_limit"})

    budgeted_parents = []
    current_context_chars = 0
    warnings_budget = []

    for p in candidates_within_limit:
        p_chars = p["char_count"]

        # Oversized first parent check
        if not budgeted_parents and p_chars > total_context_max_chars:
            budgeted_parents.append(p)
            current_context_chars += p_chars
            warnings_budget.append(
                f"oversized_first_parent_budget_exceeded: Parent đầu tiên '{p['parent_id']}' dài {p_chars} chars vượt TOTAL_CONTEXT_MAX_CHARS ({total_context_max_chars}). Giữ nguyên parent đầu tiên."
            )
            break

        if current_context_chars + p_chars <= total_context_max_chars:
            budgeted_parents.append(p)
            current_context_chars += p_chars
        else:
            dropped_parents.append({"parent_id": p["parent_id"], "reason": "context_budget"})

    t_end = time.perf_counter()
    mapping_ms = round((t_end - t_start) * 1000, 2)

    expanded_parent_chars_total = sum(p["char_count"] for p in budgeted_parents)
    expansion_factor = round(expanded_parent_chars_total / max(1, child_chars_total), 2)

    children_per_parent = {p["parent_id"]: len(p["supporting_child_ids"]) for p in aggregated_parents}

    trace = {
        "input_child_hit_count": len(child_hits),
        "unique_parent_count": len(aggregated_parents),
        "children_per_parent": children_per_parent,
        "child_to_parent_mapping_table": mapping_table,
        "parent_score_components": parent_score_components,
        "dropped_parents": dropped_parents,
        "child_chars_total": child_chars_total,
        "expanded_parent_chars_total": expanded_parent_chars_total,
        "context_expansion_factor": expansion_factor,
        "ambiguous_count": sum(1 for p in budgeted_parents if p["ambiguous"]),
        "warnings": warnings_budget,
        "mapping_latency_ms": mapping_ms,
    }

    return budgeted_parents, aggregated_parents, trace


def search_parent_documents(
    question: str,
    top_k: Optional[int] = None,
    strategy: str = "hierarchical",
    mode: str = "multi_parent",
    config: Dict[str, Any] = None,
    input_dir: Optional[Path] = None,
    hierarchy_dir: Optional[Path] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Thực hiện quy trình Truy xuất Child -> Mở rộng Parent Document đầy đủ kèm Trace.
    """
    import time
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    # 1. Precondition Check: Check Hierarchy Store
    status_info = get_hierarchy_status(output_dir=hierarchy_dir)
    if not status_info.get("store_exists"):
        t_end = time.perf_counter()
        return {
            "status": "hierarchy_not_ready",
            "question": question,
            "mode": mode,
            "strategy": strategy,
            "parents": [],
            "warnings": ["Hierarchy Store chưa được tạo hoặc tệp manifest.json bị thiếu. Vui lòng chạy 'build-hierarchy' trước."],
            "latency_ms": {"total_ms": round((t_end - t_start) * 1000, 2)},
        }

    # Load store data
    try:
        children_dict, parents_dict, manifest_data = load_hierarchy_store_data(hierarchy_dir=hierarchy_dir)
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "status": "hierarchy_not_ready",
            "question": question,
            "mode": mode,
            "strategy": strategy,
            "parents": [],
            "warnings": [f"Lỗi nạp Hierarchy Store: {e}"],
            "latency_ms": {"total_ms": round((t_end - t_start) * 1000, 2)},
        }

    # 2. Child Retrieval Phase based on mode
    child_search_res = {}
    warnings = []
    if mode == "single_parent":
        # Single query Q0 retrieval
        q0_clean = normalize_query_text(question)
        t_ret_start = time.perf_counter()
        if hybrid_retriever_fn is not None:
            raw_hits = hybrid_retriever_fn(q0_clean, config["per_query_candidates"])
        else:
            import advanced_rag
            try:
                trace = advanced_rag.search_hybrid(
                    question=q0_clean,
                    top_k=config["per_query_candidates"],
                    strategy=strategy,
                    input_dir=input_dir,
                )
                raw_hits = trace.get("results", [])
            except Exception as ex:
                chunks_cache, _ = advanced_rag.rag.load_chunks(input_path=input_dir, target_strategy=strategy)
                trace = advanced_rag.search_bm25(
                    question=q0_clean,
                    chunks=chunks_cache,
                    candidate_k=config["per_query_candidates"],
                )
                raw_hits = trace if isinstance(trace, list) else trace.get("results", [])
                warnings.append(f"Chroma Vector DB chưa nạp collection ({ex}). Hệ thống tự động fallback sang BM25 search cho Q0.")
        t_ret_end = time.perf_counter()

        # Build Q0 child hits format
        fused_child_hits = []
        for rank, c in enumerate(raw_hits, start=1):
            cid = c.get("child_id") or c.get("chunk_id")
            fused_child_hits.append({
                "child_id": cid,
                "text": c["text"],
                "source": c["source"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "multi_query_rank": rank,
                "multi_query_rrf_score": round(1.0 / (config["multi_query_rrf_k"] + rank), 6),
                "support_query_count": 1,
                "support_query_ids": ["Q0"],
                "per_query_ranks": {"Q0": rank},
            })

        child_search_res = {
            "question": q0_clean,
            "status": "ready",
            "results": fused_child_hits,
            "warnings": [],
            "latency_ms": {"total_ms": round((t_ret_end - t_ret_start) * 1000, 2)},
        }
    else:
        # Full multi_parent mode
        child_search_res = search_multi_query_child(
            question=question,
            top_k=top_k,
            strategy=strategy,
            config=config,
            input_dir=input_dir,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
        )

    fused_child_hits = child_search_res.get("results", [])
    warnings = list(child_search_res.get("warnings", []))

    # 3. Expand Children to Parent Documents
    budgeted_parents, all_aggregated_parents, expansion_trace = expand_children_to_parents(
        child_hits=fused_child_hits,
        children_dict=children_dict,
        parents_dict=parents_dict,
        config=config,
    )

    warnings.extend(expansion_trace.get("warnings", []))

    t_end = time.perf_counter()
    total_ms = round((t_end - t_start) * 1000, 2)

    status_code = child_search_res.get("status", "ready")

    return {
        "question": question,
        "strategy": strategy,
        "mode": mode,
        "status": status_code,
        "child_retrieval_trace": child_search_res,
        "parents": budgeted_parents,
        "all_parent_candidates": all_aggregated_parents,
        "expansion_trace": expansion_trace,
        "warnings": warnings,
        "latency_ms": {
            "child_search_ms": child_search_res.get("latency_ms", {}).get("total_ms", 0.0),
            "parent_expansion_ms": expansion_trace.get("mapping_latency_ms", 0.0),
            "total_ms": total_ms,
        },
    }


# -----------------------------------------------------------------------------
# CLI INTERFACE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI Quản lý Hierarchy Registry & Parent Store (Buổi 09)")
    subparsers = parser.add_subparsers(dest="command", help="Danh sách subcommands")

    subparsers.add_parser("hierarchy-audit", help="Kiểm tra read-only phân giải hierarchy của input chunks")
    subparsers.add_parser("build-hierarchy", help="Xây dựng Hierarchy Registry & Parent Store")
    subparsers.add_parser("hierarchy-status", help="Kiểm tra trạng thái read-only của Store hiện tại")

    cmd_expand = subparsers.add_parser("expand-query", help="Sinh và hiển thị các biến thể câu hỏi (Multi-Query Expansion)")
    cmd_expand.add_argument("--question", type=str, default="Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?", help="Nội dung câu hỏi gốc cần sinh biến thể")

    cmd_mc = subparsers.add_parser("multi-child", help="Thực thi Multi-Query Per-Query Hybrid + Cross-Query RRF Child Search")
    cmd_mc.add_argument("--question", type=str, default="Điều kiện vay vốn và các trường hợp không được cho vay là gì?", help="Câu hỏi gốc")

    cmd_pr = subparsers.add_parser("parent-retrieve", help="Thực thi Parent Expansion & Parent Aggregation Retrieval")
    cmd_pr.add_argument("--mode", type=str, default="multi_parent", choices=["single_parent", "multi_parent"], help="Mode truy xuất parent")
    cmd_pr.add_argument("--question", type=str, default="Điều kiện vay vốn và các trường hợp không được cho vay là gì?", help="Câu hỏi gốc")

    args = parser.parse_args()

    if args.command == "hierarchy-audit":
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO AUDIT HIERARCHY RESOLUTION (Read-only)")
        print("============================================================")
        res = build_hierarchy_store(config=config)
        
        # Đọc chi tiết để hiển thị statistics
        with open(HIERARCHY_DIR / "children.json", "r", encoding="utf-8") as f:
            children = json.load(f)
        with open(HIERARCHY_DIR / "parents.json", "r", encoding="utf-8") as f:
            parents = json.load(f)

        method_counts = {}
        for c in children:
            m = c["resolution_method"]
            method_counts[m] = method_counts.get(m, 0) + 1

        print(f" Tổng số Child Chunks        : {len(children)}")
        print(f" Phân bổ Resolution Methods  : {method_counts}")
        print(f" Số Child bị Ambiguous      : {sum(1 for c in children if c['ambiguous'])}")
        print(f" Tổng số Parent Documents    : {len(parents)}")
        
        parent_lens = [p["char_count"] for p in parents]
        if parent_lens:
            print(f" Độ dài Parent Text (chars)  : Min={min(parent_lens)}, Mean={sum(parent_lens)/len(parent_lens):.1f}, Max={max(parent_lens)}")
        print("============================================================\n")

    elif args.command == "build-hierarchy":
        config = load_config()
        print("\n⚡ Đang thực thi build Hierarchy Registry & Parent Store...")
        res = build_hierarchy_store(config=config)
        print(f"✅ Xây dựng thành công! Store lưu tại: {res['output_dir']}")
        print(f"   Children: {res['children_count']} | Parents: {res['parents_count']} | Ambiguous: {res['ambiguous_count']} | Warnings: {res['warnings_count']}\n")

    elif args.command == "hierarchy-status":
        st_info = get_hierarchy_status()
        print("\n============================================================")
        print("  TRẠNG THÁI HIERARCHY STORE (Read-only)")
        print("============================================================")
        print(f" Store Tồn Tại         : {st_info['store_exists']}")
        if st_info['store_exists']:
            print(f" Schema Version        : {st_info.get('schema_version')}")
            print(f" Total Children Count  : {st_info.get('children_count')}")
            print(f" Total Parents Count   : {st_info.get('parents_count')}")
            print(f" Ambiguous Children    : {st_info.get('ambiguous_count')}")
            print(f" Warnings Count        : {st_info.get('warnings_count')}")
            print(f" Build Timestamp (UTC) : {st_info.get('build_timestamp')}")
        print("============================================================\n")

    elif args.command == "expand-query":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO MULTI-QUERY EXPANSION (Command: expand-query)")
        print("============================================================")
        print(f" Câu hỏi gốc (Q0) : {q}")
        res = generate_multi_queries(question=q, config=config)
        print(f" Status            : {res['status']}")
        print(f" Model             : {res['model']}")
        print(f" Latency           : {res['generation_latency_ms']} ms")
        print(f" Cache Hit         : {res.get('cache_hit', False)}")
        print("------------------------------------------------------------")
        for q_item in res["queries"]:
            print(f" [{q_item['query_id']}] ({q_item['origin']}/{q_item['focus']}): {q_item['text']}")
        if res.get("warnings"):
            print(" Warnings          :", res["warnings"])
        print("============================================================\n")

    elif args.command == "multi-child":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO MULTI-QUERY CHILD RETRIEVAL (Command: multi-child)")
        print("============================================================")
        print(f" Câu hỏi gốc      : {q}")
        res = search_multi_query_child(question=q, config=config)
        print(f" Status           : {res['status']}")
        print(f" Query Counts     : {res['query_counts']}")
        print(f" Latency Trace    : Gen={res['latency_ms']['multi_query_gen_ms']}ms | Ret={res['latency_ms']['per_query_retrieval_ms']}ms | Fusion={res['latency_ms']['fusion_ms']}ms | Total={res['latency_ms']['total_ms']}ms")
        print(f" Union Child Hits : {res['union_child_count']}")
        print(f" Overlap Dist     : {res['overlap_distribution']}")
        print("------------------------------------------------------------")
        print(" Top Fused Child Hits:")
        for child in res["results"][:10]:
            print(f" [{child['multi_query_rank']:02d}] MQ-RRF: {child['multi_query_rrf_score']:.6f} | Support: {child['support_query_count']} ({child['support_query_ids']}) | Ranks: {child['per_query_ranks']} | ID: {child['child_id']}")
        if res.get("warnings"):
            print(" Warnings         :", res["warnings"])
        print("============================================================\n")

    elif args.command == "parent-retrieve":
        q = args.question
        m = args.mode
        config = load_config()
        print("\n============================================================")
        print(f"  BÁO CÁO PARENT DOCUMENT RETRIEVAL (Mode: {m})")
        print("============================================================")
        print(f" Câu hỏi gốc            : {q}")
        res = search_parent_documents(question=q, mode=m, config=config)
        print(f" Status                 : {res['status']}")
        print(f" Latency Trace          : ChildSearch={res['latency_ms']['child_search_ms']}ms | ParentExpand={res['latency_ms']['parent_expansion_ms']}ms | Total={res['latency_ms']['total_ms']}ms")
        print(f" Selected Parents Count : {len(res['parents'])}")
        if "expansion_trace" in res:
            exp = res["expansion_trace"]
            print(f" Context Expansion      : {exp['child_chars_total']} chars -> {exp['expanded_parent_chars_total']} chars (Factor: {exp['context_expansion_factor']}x)")
        print("------------------------------------------------------------")
        print(" Tree View Ánh Xạ Parent └── Child └── Query Support:")
        for p in res["parents"]:
            print(f"\n 🏢 Parent [{p['parent_id']}] (Rank #{p['parent_rank']} | Score: {p['parent_rrf_score']:.6f} | Support Qs: {p['support_query_ids']})")
            print(f"    Source: {p['source']} (Trang {p['page_start']}-{p['page_end']}) | Article: {p['article_key']}")
            for cid in p["supporting_child_ids"]:
                print(f"    └── 📄 Child [{cid}]")
        if res.get("warnings"):
            print("\n Warnings               :", res["warnings"])
        print("============================================================\n")

    else:
        parser.print_help()


# -----------------------------------------------------------------------------
# PARENT RERANK, EVIDENCE GATE, CITATIONS & PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def build_query_child_matrix(child_hits: List[Dict[str, Any]], query_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tạo ma trận Query-Child hiển thị rank của child chunk trong từng query (Q0, Q1...).
    """
    matrix_rows = []
    q_ids = [q["query_id"] for q in query_set]

    for c in child_hits:
        cid = c.get("child_id") or c.get("chunk_id")
        per_ranks = c.get("per_query_ranks", {})
        row = {
            "child_id": cid,
            "source": c.get("source", ""),
            "support_count": c.get("support_query_count", len(c.get("support_query_ids", []))),
            "mq_rrf_score": c.get("multi_query_rrf_score", 0.0),
        }
        for qid in q_ids:
            row[qid] = per_ranks.get(qid, "—")
        matrix_rows.append(row)

    return matrix_rows


def format_parent_tree_node(parent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuyển đổi dữ liệu Parent Candidate thành cấu trúc Cây hiển thị UI.
    """
    return {
        "parent_id": parent.get("parent_id"),
        "rank_before": parent.get("parent_rank", 1),
        "rank_after": parent.get("parent_rerank_rank", 1),
        "rank_change": parent.get("parent_rank_change", 0),
        "rrf_score": parent.get("parent_rrf_score", 0.0),
        "rerank_score": parent.get("parent_rerank_score", 0.0),
        "source": parent.get("source", ""),
        "pages": f"Trang {parent.get('page_start', 1)}-{parent.get('page_end', 1)}",
        "structural_path": parent.get("structural_path", {}),
        "supporting_children_count": len(parent.get("supporting_child_ids", [])),
        "support_queries": parent.get("support_query_ids", []),
        "anchor_child": parent.get("anchor_child_id", ""),
        "ambiguous": parent.get("ambiguous", False),
        "warnings": parent.get("warnings", []),
    }


def map_status_to_ui_alert(status: str) -> Dict[str, str]:
    """
    Ánh xạ status code sang giao diện cảnh báo báo lỗi thân thiện.
    """
    mapping = {
        "hierarchy_not_ready": {
            "type": "error",
            "title": "Hierarchy Store Chưa Tồn Tại",
            "action": "Vui lòng nhấn nút 'Build Hierarchy Store' ở Sidebar để khởi tạo.",
        },
        "collection_not_ready": {
            "type": "error",
            "title": "Chroma Collection Chưa Sẵn Sàng",
            "action": "Vui lòng chuẩn bị dữ liệu vector database ở Sidebar.",
        },
        "query_generation_unavailable": {
            "type": "warning",
            "title": "Không Thể Sinh Multi-Query Expansion",
            "action": "Hệ thống tự động sử dụng câu hỏi gốc Q0 để tiếp tục retrieval.",
        },
        "multi_query_partial": {
            "type": "warning",
            "title": "Một Số Query Biến Thể Bị Lỗi",
            "action": "Kết quả được dung hợp từ các câu hỏi thành công khả dụng.",
        },
        "reranker_unavailable": {
            "type": "error",
            "title": "Reranker Model Không Khả Dụng",
            "action": "Vui lòng kiểm tra môi trường chạy hoặc bộ nhớ RAM/GPU.",
        },
        "insufficient_evidence": {
            "type": "warning",
            "title": "Không Đủ Bằng Chứng Pháp Lý Tin Cậy",
            "action": "Không có tài liệu nào vượt ngưỡng RERANK_MIN_SCORE. Gemini Answer Generation không được kích hoạt.",
        },
        "ready": {
            "type": "success",
            "title": "Thực Thi Thành Công",
            "action": "Đã hoàn thành truy xuất và tổng hợp câu trả lời.",
        },
    }
    return mapping.get(status, {"type": "info", "title": f"Trạng thái: {status}", "action": ""})


def predict_reranker_scores(pairs: List[Tuple[str, str]], config: Dict[str, Any] = None) -> List[float]:
    """
    Dự đoán điểm số Cross-Encoder Reranker cho danh sách các cặp (query, document_text).
    """
    if not pairs:
        return []
    import advanced_rag
    if config is None:
        config = load_config()

    tokenizer, model, device_used = advanced_rag.get_reranker_model(config)
    import torch
    max_length = config.get("reranker_max_length", 512)
    batch_size = config.get("rerank_batch_size", 4)
    raw_scores = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = [[p[0], p[1]] for p in pairs[i : i + batch_size]]
        inputs = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device_used)

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            scores = logits.view(-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            raw_scores.extend(scores)
    return raw_scores


def rerank_parents(
    question: str,
    parent_candidates: List[Dict[str, Any]],
    config: Dict[str, Any] = None,
    reranker_fn: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Rerank danh sách Parent Candidates bằng Cross-Encoder (dùng Q0 + parent_text).
    """
    import math
    import time
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    q0_clean = normalize_query_text(question)

    if not parent_candidates:
        return [], 0.0

    pairs = [(q0_clean, p["text"]) for p in parent_candidates]

    if reranker_fn is not None:
        raw_scores = reranker_fn(pairs)
    else:
        raw_scores = predict_reranker_scores(pairs, config)

    reranked_parents = []
    for p, raw_s in zip(parent_candidates, raw_scores):
        raw_val = float(raw_s)
        sigmoid_score = round(1.0 / (1.0 + math.exp(-raw_val)), 6)
        p_copy = dict(p)
        p_copy["parent_rerank_raw_score"] = round(raw_val, 4)
        p_copy["parent_rerank_score"] = sigmoid_score
        reranked_parents.append(p_copy)

    # Sort: parent_rerank_score (desc) -> parent_rank (asc) -> parent_id (asc)
    reranked_parents.sort(
        key=lambda x: (
            -x["parent_rerank_score"],
            x["parent_rank"],
            x["parent_id"],
        )
    )

    for r_rank, p in enumerate(reranked_parents, start=1):
        p["parent_rerank_rank"] = r_rank
        p["parent_rank_change"] = p["parent_rank"] - r_rank

    t_end = time.perf_counter()
    rerank_ms = round((t_end - t_start) * 1000, 2)

    return reranked_parents, rerank_ms


def generate_rag_answer(
    question: str,
    accepted_evidence: List[Dict[str, Any]],
    config: Dict[str, Any] = None,
    answer_generator_fn: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]], float]:
    """
    Sinh câu trả lời từ Accepted Evidence và tạo danh sách Citations [P1], [P2]...
    """
    import time
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    q0_clean = normalize_query_text(question)

    if not accepted_evidence:
        t_end = time.perf_counter()
        return (
            "Dựa trên các văn bản quy định hiện tại, không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.",
            [],
            round((t_end - t_start) * 1000, 2),
        )

    # Build Citations List and Context Prompt
    citations = []
    context_blocks = []

    for idx, ev in enumerate(accepted_evidence, start=1):
        label = f"P{idx}"
        pid = ev.get("parent_id") or ev.get("child_id") or f"doc_{idx}"
        anchor_id = ev.get("anchor_child_id") or ev.get("child_id") or pid
        sup_ids = ev.get("supporting_child_ids") or [anchor_id]

        cit_obj = {
            "evidence_id": label,
            "parent_id": pid,
            "anchor_child_id": anchor_id,
            "supporting_child_ids": sup_ids,
            "source": ev.get("source", ""),
            "page_start": ev.get("page_start", 1),
            "page_end": ev.get("page_end", 1),
            "structural_path": ev.get("structural_path", {}),
            "parent_rerank_score": ev.get("parent_rerank_score") or ev.get("rerank_score") or 0.0,
            "ambiguous": ev.get("ambiguous", False),
            "warnings": list(ev.get("warnings", [])),
        }
        citations.append(cit_obj)

        c_text = (
            f"[{label}] Nguồn: {ev.get('source', '')} | {ev.get('article_key', '')}\n"
            f"{ev.get('text', '')}"
        )
        context_blocks.append(c_text)

    full_context_str = "\n\n---\n\n".join(context_blocks)

    if answer_generator_fn is not None:
        raw_ans = answer_generator_fn(q0_clean, full_context_str, citations)
    else:
        api_key = config.get("api_key")
        if not api_key:
            t_end = time.perf_counter()
            return (
                "Lỗi: GEMINI_API_KEY chưa được cấu hình. Không thể sinh câu trả lời.",
                citations,
                round((t_end - t_start) * 1000, 2),
            )

        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            prompt = (
                f"Bạn là chuyên gia tư vấn quy định ngân hàng Việt Nam.\n"
                f"Nhiệm vụ: Trả lời câu hỏi dưới đây CHỈ DỰA TRÊN NGỮ CẢNH TRÍCH DẪN [P1], [P2]... ĐÃ ĐƯỢC CẤP.\n\n"
                f"Câu hỏi gốc: \"{q0_clean}\"\n\n"
                f"Dữ liệu Ngữ cảnh Bằng chứng:\n"
                f"{full_context_str}\n\n"
                f"Quy tắc bắt buộc khi trả lời:\n"
                f"1. CHỈ sử dụng thông tin có trong Ngữ cảnh Bằng chứng. KHÔNG tự suy diễn hoặc bịa đặt quy định ngoài tài liệu.\n"
                f"2. Với MỖI nhận định/thông tin trích xuất, BẮT BUỘC ghi rõ nhãn nguồn trích dẫn tương ứng như [P1], [P2]...\n"
                f"3. NẾU Ngữ cảnh chứa cảnh báo hoặc có thông tin mâu thuẫn/chưa rõ ràng, hãy ghi rõ giới hạn đó trong câu trả lời.\n"
                f"4. Trả lời mạch lạc, rõ ràng bằng tiếng Việt chuẩn mực.\n"
            )

            res = client.models.generate_content(
                model=config["generation_model"],
                contents=prompt,
            )

            if not res or not hasattr(res, "text") or not res.text:
                raise ValueError("Gemini Generation API trả về nội dung rỗng.")

            raw_ans = res.text.strip()
        except Exception as e:
            raw_ans = f"Lỗi Gemini Generation API: {e}"

    t_end = time.perf_counter()
    ans_ms = round((t_end - t_start) * 1000, 2)

    return raw_ans, citations, ans_ms


def execute_query_pipeline(
    question: str,
    mode: str = "multi_parent",
    strategy: str = "hierarchical",
    config: Dict[str, Any] = None,
    input_dir: Optional[Path] = None,
    hierarchy_dir: Optional[Path] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
    answer_generator_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Thực thi Pipeline hoàn chỉnh cho cả 4 modes: single_flat, multi_flat, single_parent, multi_parent.
    """
    import time
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    q0_clean = normalize_query_text(question)
    min_score = config["rerank_min_score"]
    final_top_k = config["final_parent_top_k"]

    generation_calls = 0
    embedding_calls = 0

    warnings = []
    errors = []

    # 1. Pipeline Routing by Mode
    if mode in ["single_flat", "multi_flat"]:
        if mode == "single_flat":
            query_set = [{"query_id": "Q0", "text": q0_clean, "origin": "original", "focus": "original_intent"}]
            t_ret_start = time.perf_counter()
            embedding_calls += 1
            if hybrid_retriever_fn is not None:
                raw_hits = hybrid_retriever_fn(q0_clean, config["per_query_candidates"])
            else:
                import advanced_rag
                try:
                    trace = advanced_rag.search_hybrid(
                        question=q0_clean,
                        top_k=config["per_query_candidates"],
                        strategy=strategy,
                        input_dir=input_dir,
                    )
                    raw_hits = trace.get("results", [])
                except Exception as ex:
                    # Fallback to BM25 search if Chroma collection is missing or fails
                    chunks_cache, _ = advanced_rag.rag.load_chunks(input_path=input_dir, target_strategy=strategy)
                    trace = advanced_rag.search_bm25(
                        question=q0_clean,
                        chunks=chunks_cache,
                        candidate_k=config["per_query_candidates"],
                    )
                    raw_hits = trace if isinstance(trace, list) else trace.get("results", [])
                    warnings.append(f"Chroma Vector DB chưa nạp collection ({ex}). Hệ thống tự động fallback sang BM25 search cho Q0.")
            t_ret_end = time.perf_counter()

            child_hits = raw_hits
            ret_trace = {"latency_ms": {"total_ms": round((t_ret_end - t_ret_start) * 1000, 2)}}
        else:
            generation_calls += 1
            exp_res = generate_multi_queries(question=q0_clean, config=config, query_generator_fn=query_generator_fn)
            query_set = exp_res.get("queries", [])
            if exp_res.get("status") != "ready":
                warnings.extend(exp_res.get("warnings", []))

            child_search_res = search_multi_query_child(
                question=q0_clean,
                top_k=config["per_query_candidates"],
                strategy=strategy,
                config=config,
                input_dir=input_dir,
                query_generator_fn=query_generator_fn,
                hybrid_retriever_fn=hybrid_retriever_fn,
            )
            child_hits = child_search_res.get("results", [])
            embedding_calls += len(query_set)
            ret_trace = child_search_res

        # Rerank flat child hits using Q0
        pairs = [(q0_clean, c["text"]) for c in child_hits]
        t_rerank_start = time.perf_counter()
        if reranker_fn is not None:
            raw_scores = reranker_fn(pairs)
        else:
            raw_scores = predict_reranker_scores(pairs, config)
        t_rerank_end = time.perf_counter()
        rerank_ms = round((t_rerank_end - t_rerank_start) * 1000, 2)

        import math
        reranked_children = []
        for c, raw_s in zip(child_hits, raw_scores):
            raw_val = float(raw_s)
            sig_s = round(1.0 / (1.0 + math.exp(-raw_val)), 6)
            c_copy = dict(c)
            c_copy["rerank_score"] = sig_s
            c_copy["parent_rerank_score"] = sig_s
            c_copy["anchor_child_id"] = c.get("child_id") or c.get("chunk_id")
            c_copy["supporting_child_ids"] = [c_copy["anchor_child_id"]]
            c_copy["support_query_ids"] = c.get("support_query_ids", ["Q0"])
            reranked_children.append(c_copy)

        reranked_children.sort(key=lambda x: -x["rerank_score"])
        for r_rank, c in enumerate(reranked_children, start=1):
            c["rerank_rank"] = r_rank

        accepted_evidence = [c for c in reranked_children if c["rerank_score"] >= min_score][:final_top_k]

        parent_candidates = reranked_children
        parent_rerank_ms = rerank_ms
        expansion_ms = 0.0
        pipeline_status = child_search_res.get("status", "ready") if mode == "multi_flat" else "ready"

    else:
        # Parent Modes Pipeline (single_parent, multi_parent)
        parent_ret_res = search_parent_documents(
            question=q0_clean,
            top_k=config["per_query_candidates"],
            strategy=strategy,
            mode=mode,
            config=config,
            input_dir=input_dir,
            hierarchy_dir=hierarchy_dir,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
        )

        if parent_ret_res.get("status") == "hierarchy_not_ready":
            t_end = time.perf_counter()
            return {
                "question": q0_clean,
                "strategy": strategy,
                "mode": mode,
                "status": "hierarchy_not_ready",
                "query_set": [{"query_id": "Q0", "text": q0_clean, "origin": "original"}],
                "child_hits": [],
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": "Không thể thực hiện truy xuất do Hierarchy Store chưa được tạo.",
                "citations": [],
                "latency_ms": {"total_ms": round((t_end - t_start) * 1000, 2)},
                "api_call_counts": {"generation_calls": 0, "embedding_calls": 0},
                "warnings": parent_ret_res.get("warnings", []),
                "errors": ["Hierarchy Store missing"],
            }

        child_hits = parent_ret_res.get("child_retrieval_trace", {}).get("results", [])
        parent_cands = parent_ret_res.get("parents", [])
        warnings.extend(parent_ret_res.get("warnings", []))
        pipeline_status = parent_ret_res.get("status", "ready")

        if mode == "multi_parent":
            generation_calls += 1
            query_set = parent_ret_res.get("child_retrieval_trace", {}).get("per_query_stats", [])
            embedding_calls += len(query_set) if query_set else 1
        else:
            query_set = [{"query_id": "Q0", "text": q0_clean, "origin": "original"}]
            embedding_calls += 1

        # Rerank parents using Q0
        try:
            reranked_parents, parent_rerank_ms = rerank_parents(
                question=q0_clean,
                parent_candidates=parent_cands,
                config=config,
                reranker_fn=reranker_fn,
            )
        except Exception as e:
            t_end = time.perf_counter()
            return {
                "question": q0_clean,
                "strategy": strategy,
                "mode": mode,
                "status": "reranker_unavailable",
                "query_set": query_set,
                "child_hits": child_hits,
                "parent_candidates": parent_cands,
                "accepted_evidence": [],
                "answer": f"Reranker model không sẵn sàng: {e}",
                "citations": [],
                "latency_ms": {"total_ms": round((t_end - t_start) * 1000, 2)},
                "api_call_counts": {"generation_calls": generation_calls, "embedding_calls": embedding_calls},
                "warnings": warnings,
                "errors": [f"Lỗi reranker: {e}"],
            }

        parent_candidates = reranked_parents
        accepted_evidence = [p for p in reranked_parents if p["parent_rerank_score"] >= min_score][:final_top_k]
        expansion_ms = parent_ret_res.get("latency_ms", {}).get("parent_expansion_ms", 0.0)

    # 2. Evidence Gate Check & Answer Generation
    if not accepted_evidence:
        t_end = time.perf_counter()
        return {
            "question": q0_clean,
            "strategy": strategy,
            "mode": mode,
            "status": "insufficient_evidence",
            "query_set": query_set,
            "child_hits": child_hits,
            "parent_candidates": parent_candidates,
            "accepted_evidence": [],
            "answer": "Dựa trên các văn bản quy định hiện tại, không tìm thấy bằng chứng pháp lý nào vượt ngưỡng tin cậy để trả lời câu hỏi của bạn.",
            "citations": [],
            "latency_ms": {
                "parent_rerank_ms": parent_rerank_ms,
                "answer_generation_ms": 0.0,
                "total_ms": round((t_end - t_start) * 1000, 2),
            },
            "api_call_counts": {
                "generation_calls": generation_calls,
                "embedding_calls": embedding_calls,
            },
            "warnings": warnings + ["Ngưỡng tin cậy Evidence Gate không đạt. Không gọi Gemini Answer Generation."],
            "errors": errors,
        }

    # Answer Generation Call
    generation_calls += 1
    answer_text, citations, ans_ms = generate_rag_answer(
        question=q0_clean,
        accepted_evidence=accepted_evidence,
        config=config,
        answer_generator_fn=answer_generator_fn,
    )

    t_end = time.perf_counter()
    total_ms = round((t_end - t_start) * 1000, 2)

    return {
        "question": q0_clean,
        "strategy": strategy,
        "mode": mode,
        "status": pipeline_status,
        "query_set": query_set,
        "child_hits": child_hits,
        "parent_candidates": parent_candidates,
        "accepted_evidence": accepted_evidence,
        "answer": answer_text,
        "citations": citations,
        "latency_ms": {
            "parent_expansion_ms": expansion_ms,
            "parent_rerank_ms": parent_rerank_ms,
            "answer_generation_ms": ans_ms,
            "total_ms": total_ms,
        },
        "api_call_counts": {
            "generation_calls": generation_calls,
            "embedding_calls": embedding_calls,
        },
        "system_info": {
            "corpus": "banking_legal_vn",
            "generation_model": config["generation_model"],
            "reranker_model": config["reranker_model"],
            "hierarchy_store_ready": True,
        },
        "warnings": warnings,
        "errors": errors,
    }


def compare_retrieval_modes(
    question: str,
    strategy: str = "hierarchical",
    config: Dict[str, Any] = None,
    input_dir: Optional[Path] = None,
    hierarchy_dir: Optional[Path] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    So sánh hiệu năng 4 Retrieval Modes mà KHÔNG gọi Answer Generation LLM.
    """
    import time
    t_start = time.perf_counter()

    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    comparison_results = {}

    for m in modes:
        res = execute_query_pipeline(
            question=question,
            mode=m,
            strategy=strategy,
            config=config,
            input_dir=input_dir,
            hierarchy_dir=hierarchy_dir,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            reranker_fn=reranker_fn,
            answer_generator_fn=lambda q, ctx, cits: "MOCK_COMPARE_NO_GEN",
        )
        res["answer"] = "[COMPARE_MODE_NO_ANSWER_GENERATION]"
        res["api_call_counts"]["generation_calls"] = min(res["api_call_counts"]["generation_calls"], 1)
        comparison_results[m] = res

    t_end = time.perf_counter()
    return {
        "question": question,
        "strategy": strategy,
        "modes": comparison_results,
        "total_compare_ms": round((t_end - t_start) * 1000, 2),
    }


# -----------------------------------------------------------------------------
# CLI INTERFACE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI Quản lý Hierarchy Registry & Parent Store (Buổi 09)")
    subparsers = parser.add_subparsers(dest="command", help="Danh sách subcommands")

    subparsers.add_parser("hierarchy-audit", help="Kiểm tra read-only phân giải hierarchy của input chunks")
    subparsers.add_parser("build-hierarchy", help="Xây dựng Hierarchy Registry & Parent Store")
    subparsers.add_parser("hierarchy-status", help="Kiểm tra trạng thái read-only của Store hiện tại")

    cmd_expand = subparsers.add_parser("expand-query", help="Sinh và hiển thị các biến thể câu hỏi (Multi-Query Expansion)")
    cmd_expand.add_argument("--question", type=str, default="Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?", help="Nội dung câu hỏi gốc cần sinh biến thể")

    cmd_mc = subparsers.add_parser("multi-child", help="Thực thi Multi-Query Per-Query Hybrid + Cross-Query RRF Child Search")
    cmd_mc.add_argument("--question", type=str, default="Điều kiện vay vốn và các trường hợp không được cho vay là gì?", help="Câu hỏi gốc")

    cmd_pr = subparsers.add_parser("parent-retrieve", help="Thực thi Parent Expansion & Parent Aggregation Retrieval")
    cmd_pr.add_argument("--mode", type=str, default="multi_parent", choices=["single_parent", "multi_parent"], help="Mode truy xuất parent")
    cmd_pr.add_argument("--question", type=str, default="Điều kiện vay vốn và các trường hợp không được cho vay là gì?", help="Câu hỏi gốc")

    cmd_q = subparsers.add_parser("query", help="Chạy quy trình RAG Pipeline hoàn chỉnh (Query -> Retrieval -> Rerank -> Generation)")
    cmd_q.add_argument("--mode", type=str, default="multi_parent", choices=["single_flat", "multi_flat", "single_parent", "multi_parent"], help="Mode thực thi RAG")
    cmd_q.add_argument("--question", type=str, default="Điều kiện vay vốn và các nhu cầu vốn không được cho vay được quy định thế nào?", help="Câu hỏi gốc")

    cmd_cmp = subparsers.add_parser("compare", help="So sánh 4 Retrieval Modes (Không gọi Answer Generation)")
    cmd_cmp.add_argument("--question", type=str, default="Điều kiện vay vốn và các nhu cầu vốn không được cho vay được quy định thế nào?", help="Câu hỏi gốc")

    args = parser.parse_args()

    if args.command == "hierarchy-audit":
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO AUDIT HIERARCHY RESOLUTION (Read-only)")
        print("============================================================")
        res = build_hierarchy_store(config=config)
        
        # Đọc chi tiết để hiển thị statistics
        with open(HIERARCHY_DIR / "children.json", "r", encoding="utf-8") as f:
            children = json.load(f)
        with open(HIERARCHY_DIR / "parents.json", "r", encoding="utf-8") as f:
            parents = json.load(f)

        method_counts = {}
        for c in children:
            m = c["resolution_method"]
            method_counts[m] = method_counts.get(m, 0) + 1

        print(f" Tổng số Child Chunks        : {len(children)}")
        print(f" Phân bổ Resolution Methods  : {method_counts}")
        print(f" Số Child bị Ambiguous      : {sum(1 for c in children if c['ambiguous'])}")
        print(f" Tổng số Parent Documents    : {len(parents)}")
        
        parent_lens = [p["char_count"] for p in parents]
        if parent_lens:
            print(f" Độ dài Parent Text (chars)  : Min={min(parent_lens)}, Mean={sum(parent_lens)/len(parent_lens):.1f}, Max={max(parent_lens)}")
        print("============================================================\n")

    elif args.command == "build-hierarchy":
        config = load_config()
        print("\n⚡ Đang thực thi build Hierarchy Registry & Parent Store...")
        res = build_hierarchy_store(config=config)
        print(f"✅ Xây dựng thành công! Store lưu tại: {res['output_dir']}")
        print(f"   Children: {res['children_count']} | Parents: {res['parents_count']} | Ambiguous: {res['ambiguous_count']} | Warnings: {res['warnings_count']}\n")

    elif args.command == "hierarchy-status":
        st_info = get_hierarchy_status()
        print("\n============================================================")
        print("  TRẠNG THÁI HIERARCHY STORE (Read-only)")
        print("============================================================")
        print(f" Store Tồn Tại         : {st_info['store_exists']}")
        if st_info['store_exists']:
            print(f" Schema Version        : {st_info.get('schema_version')}")
            print(f" Total Children Count  : {st_info.get('children_count')}")
            print(f" Total Parents Count   : {st_info.get('parents_count')}")
            print(f" Ambiguous Children    : {st_info.get('ambiguous_count')}")
            print(f" Warnings Count        : {st_info.get('warnings_count')}")
            print(f" Build Timestamp (UTC) : {st_info.get('build_timestamp')}")
        print("============================================================\n")

    elif args.command == "expand-query":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO MULTI-QUERY EXPANSION (Command: expand-query)")
        print("============================================================")
        print(f" Câu hỏi gốc (Q0) : {q}")
        res = generate_multi_queries(question=q, config=config)
        print(f" Status            : {res['status']}")
        print(f" Model             : {res['model']}")
        print(f" Latency           : {res['generation_latency_ms']} ms")
        print(f" Cache Hit         : {res.get('cache_hit', False)}")
        print("------------------------------------------------------------")
        for q_item in res["queries"]:
            print(f" [{q_item['query_id']}] ({q_item['origin']}/{q_item['focus']}): {q_item['text']}")
        if res.get("warnings"):
            print(" Warnings          :", res["warnings"])
        print("============================================================\n")

    elif args.command == "multi-child":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO MULTI-QUERY CHILD RETRIEVAL (Command: multi-child)")
        print("============================================================")
        print(f" Câu hỏi gốc      : {q}")
        res = search_multi_query_child(question=q, config=config)
        print(f" Status           : {res['status']}")
        print(f" Query Counts     : {res['query_counts']}")
        print(f" Latency Trace    : Gen={res['latency_ms']['multi_query_gen_ms']}ms | Ret={res['latency_ms']['per_query_retrieval_ms']}ms | Fusion={res['latency_ms']['fusion_ms']}ms | Total={res['latency_ms']['total_ms']}ms")
        print(f" Union Child Hits : {res['union_child_count']}")
        print(f" Overlap Dist     : {res['overlap_distribution']}")
        print("------------------------------------------------------------")
        print(" Top Fused Child Hits:")
        for child in res["results"][:10]:
            print(f" [{child['multi_query_rank']:02d}] MQ-RRF: {child['multi_query_rrf_score']:.6f} | Support: {child['support_query_count']} ({child['support_query_ids']}) | Ranks: {child['per_query_ranks']} | ID: {child['child_id']}")
        if res.get("warnings"):
            print(" Warnings         :", res["warnings"])
        print("============================================================\n")

    elif args.command == "parent-retrieve":
        q = args.question
        m = args.mode
        config = load_config()
        print("\n============================================================")
        print(f"  BÁO CÁO PARENT DOCUMENT RETRIEVAL (Mode: {m})")
        print("============================================================")
        print(f" Câu hỏi gốc            : {q}")
        res = search_parent_documents(question=q, mode=m, config=config)
        print(f" Status                 : {res['status']}")
        print(f" Latency Trace          : ChildSearch={res['latency_ms']['child_search_ms']}ms | ParentExpand={res['latency_ms']['parent_expansion_ms']}ms | Total={res['latency_ms']['total_ms']}ms")
        print(f" Selected Parents Count : {len(res['parents'])}")
        if "expansion_trace" in res:
            exp = res["expansion_trace"]
            print(f" Context Expansion      : {exp['child_chars_total']} chars -> {exp['expanded_parent_chars_total']} chars (Factor: {exp['context_expansion_factor']}x)")
        print("------------------------------------------------------------")
        print(" Tree View Ánh Xạ Parent └── Child └── Query Support:")
        for p in res["parents"]:
            print(f"\n 🏢 Parent [{p['parent_id']}] (Rank #{p['parent_rank']} | Score: {p['parent_rrf_score']:.6f} | Support Qs: {p['support_query_ids']})")
            print(f"    Source: {p['source']} (Trang {p['page_start']}-{p['page_end']}) | Article: {p['article_key']}")
            for cid in p["supporting_child_ids"]:
                print(f"    └── 📄 Child [{cid}]")
        if res.get("warnings"):
            print("\n Warnings               :", res["warnings"])
        print("============================================================\n")

    elif args.command == "query":
        q = args.question
        m = args.mode
        config = load_config()
        print("\n============================================================")
        print(f"  BÁO CÁO HIERARCHICAL RAG PIPELINE (Mode: {m})")
        print("============================================================")
        print(f" Câu hỏi gốc       : {q}")
        res = execute_query_pipeline(question=q, mode=m, config=config)
        print(f" Status            : {res['status']}")
        print(f" API Calls         : Generation={res['api_call_counts']['generation_calls']} | Embedding={res['api_call_counts']['embedding_calls']}")
        print(f" Latency Breakdown : {res['latency_ms']}")
        print("------------------------------------------------------------")
        print(f" Accepted Evidence : {len(res['accepted_evidence'])} items")
        for idx, ev in enumerate(res["accepted_evidence"], start=1):
            print(f"   [P{idx}] Rerank Score: {ev.get('parent_rerank_score', ev.get('rerank_score'))} | ID: {ev.get('parent_id', ev.get('child_id'))}")
        print("------------------------------------------------------------")
        print(" Câu Trả Lời Sinh Ra:")
        print(res["answer"])
        if res.get("warnings"):
            print("\n Warnings          :", res["warnings"])
        print("============================================================\n")

    elif args.command == "compare":
        q = args.question
        config = load_config()
        print("\n============================================================")
        print("  BÁO CÁO SO SÁNH 4 RETRIEVAL MODES (Command: compare)")
        print("============================================================")
        print(f" Câu hỏi gốc : {q}")
        res = compare_retrieval_modes(question=q, config=config)
        print(f" Total Compare Latency : {res['total_compare_ms']} ms")
        print("------------------------------------------------------------")
        for m_name, m_res in res["modes"].items():
            ev_count = len(m_res["accepted_evidence"])
            top_score = m_res["accepted_evidence"][0].get("parent_rerank_score", m_res["accepted_evidence"][0].get("rerank_score")) if ev_count > 0 else 0.0
            print(f" 🔹 Mode [{m_name:13s}] | Status: {m_res['status']:22s} | Accepted Evidence: {ev_count} | Top Score: {top_score:.4f} | Latency: {m_res['latency_ms'].get('total_ms', 0)}ms")
        print("============================================================\n")

    else:
        parser.print_help()



