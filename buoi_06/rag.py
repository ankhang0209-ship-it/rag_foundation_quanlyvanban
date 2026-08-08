import os
import sys
import json
import sqlite3
from pathlib import Path

# Tự động nạp venv site-packages của Buổi 05 vào sys.path
BASE_DIR = Path(__file__).parent.resolve()
VENV_SITE_PACKAGES = BASE_DIR.parent / "buoi_05" / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

from dotenv import load_dotenv  # type: ignore
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH = STORAGE_DIR / "chunks.db"
CHROMA_DIR = STORAGE_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# 1. Database Connection & Helper (PostgreSQL / Fallback SQLite .db)
# ----------------------------------------------------
def get_db_connection():
    """Thử kết nối PostgreSQL; nếu chưa chạy thì tự động fallback lưu ra disk local dạng file .db (SQLite)."""
    try:
        import psycopg  # type: ignore
        conn = psycopg.connect(
            host=POSTGRES_HOST,
            port=int(POSTGRES_PORT),
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB,
            connect_timeout=3
        )
        return conn, "postgres"
    except Exception:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        return conn, "sqlite"

def init_db():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    if db_type == "postgres":
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                source VARCHAR(255),
                text TEXT,
                metadata TEXT
            );
        """)
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT,
                text TEXT,
                metadata TEXT
            );
        """)
    conn.commit()
    conn.close()

def save_chunks_to_db(chunk_list):
    init_db()
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    for item in chunk_list:
        cid = str(item.get("chunk_id"))
        source = str(item.get("source", ""))
        text = str(item.get("text", ""))
        meta_str = json.dumps(item.get("metadata", {}), ensure_ascii=False)
        if db_type == "postgres":
            cursor.execute("""
                INSERT INTO chunks (chunk_id, source, text, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE 
                SET source = EXCLUDED.source, text = EXCLUDED.text, metadata = EXCLUDED.metadata;
            """, (cid, source, text, meta_str))
        else:
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (chunk_id, source, text, metadata)
                VALUES (?, ?, ?, ?);
            """, (cid, source, text, meta_str))
    conn.commit()
    conn.close()

def get_chunks_from_db(chunk_ids):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    results = {}
    if not chunk_ids:
        conn.close()
        return results

    if db_type == "postgres":
        placeholders = ", ".join(["%s"] * len(chunk_ids))
        cursor.execute(f"SELECT chunk_id, source, text, metadata FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
    else:
        placeholders = ", ".join(["?"] * len(chunk_ids))
        cursor.execute(f"SELECT chunk_id, source, text, metadata FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
    
    rows = cursor.fetchall()
    for row in rows:
        cid, source, text, meta_raw = row
        meta = json.loads(meta_raw) if meta_raw else {}
        results[cid] = {"chunk_id": cid, "source": source, "text": text, "metadata": meta}
    conn.close()
    return results

# ----------------------------------------------------
# 2. Embedding với Gemini (Dimensions = 384) & ChromaDB
# ----------------------------------------------------
def get_chroma_collection():
    import chromadb  # type: ignore
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name="buoi_06_rag", metadata={"hnsw:space": "cosine"})

def get_gemini_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT"):
    """Tạo embedding 384 chiều bằng google-genai (gemini-embedding-2 / text-embedding-004)."""
    if not GEMINI_API_KEY:
        return [0.0] * 384
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        client = genai.Client(api_key=GEMINI_API_KEY)
        res = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=384,
                task_type=task_type
            )
        )
        if hasattr(res, "embeddings") and res.embeddings:
            return res.embeddings[0].values
        return [0.0] * 384
    except Exception:
        return [0.0] * 384

# ----------------------------------------------------
# Core Function 1: index()
# ----------------------------------------------------
def index():
    """
    Đọc các file JSON trong RAG/rag_foundation/buoi_05/output/chunks/
    Tạo embedding 384 chiều, lưu text vào Database (PostgreSQL hoặc SQLite .db local)
    và lưu embedding vào ChromaDB.
    """
    chunks_dir = BASE_DIR.parent / "buoi_05" / "output" / "chunks"
    if not chunks_dir.exists():
        return {"status": "error", "message": f"Không tìm thấy thư mục: {chunks_dir}"}

    json_files = list(chunks_dir.glob("*.json"))
    if not json_files:
        return {"status": "error", "message": f"Không tìm thấy tệp JSON nào trong {chunks_dir}"}

    all_chunks = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        all_chunks.append(item)
            elif isinstance(data, dict):
                for key in ["fixed_size_chunks", "semantic_chunks", "hierarchical_chunks", "chunks"]:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            if isinstance(item, dict) and "text" in item:
                                all_chunks.append(item)
        except Exception:
            continue

    if not all_chunks:
        return {"status": "error", "message": "Không đọc được dữ liệu chunk hợp lệ từ các tệp JSON."}

    # 1. Lưu text vào Database (PostgreSQL hoặc SQLite .db local)
    save_chunks_to_db(all_chunks)

    # 2. Tạo embeddings 384 chiều song song (10 workers) và lưu vào ChromaDB
    collection = get_chroma_collection()
    ids, embeddings, metadatas, documents = [], [], [], []

    def embed_chunk(item):
        cid = str(item.get("chunk_id"))
        text = str(item.get("text", ""))
        source = str(item.get("source", ""))
        vec = get_gemini_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        meta = {
            "source": source,
            "page_start": item.get("page_start", 1),
            "strategy": item.get("strategy", "unknown")
        }
        return cid, vec, text, meta

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        batch_results = list(executor.map(embed_chunk, all_chunks))

    for cid, vec, text, meta in batch_results:
        ids.append(cid)
        embeddings.append(vec)
        documents.append(text)
        metadatas.append(meta)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return {
        "status": "success",
        "processed_files": [f.name for f in json_files],
        "indexed_chunks": len(all_chunks)
    }

# ----------------------------------------------------
# Core Function 2: ask(question, top_k=5)
# ----------------------------------------------------
def ask(question: str, top_k: int = 5):
    """
    Embedding câu hỏi (dim=384), tìm top-k trong ChromaDB,
    đọc text tương ứng từ DB (PostgreSQL/.db), gửi cho Gemini LLM sinh câu trả lời.
    """
    collection = get_chroma_collection()
    
    # 1. Embedding câu hỏi (384 chiều)
    q_vec = get_gemini_embedding(question, task_type="RETRIEVAL_QUERY")
    
    # 2. Query top-k trong ChromaDB
    query_res = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k
    )

    retrieved_ids = query_res.get("ids", [[]])[0]
    if not retrieved_ids:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong dữ liệu.",
            "retrieved_chunks": []
        }

    # 3. Lấy text tương ứng từ Database (PostgreSQL hoặc SQLite local)
    chunks_map = get_chunks_from_db(retrieved_ids)
    
    retrieved_chunks = []
    context_passages = []
    for idx, cid in enumerate(retrieved_ids):
        chunk_data = chunks_map.get(cid, {"chunk_id": cid, "text": "", "source": "unknown"})
        retrieved_chunks.append(chunk_data)
        context_passages.append(f"[{idx+1}] Source: {chunk_data.get('source')}\nNội dung: {chunk_data.get('text')}")

    context_str = "\n\n".join(context_passages)

    # 4. Nếu thiếu GEMINI_API_KEY: Cho phép retrieval, không gọi LLM
    if not GEMINI_API_KEY:
        return {
            "answer": "[CẢNH BÁO: Chưa cấu hình GEMINI_API_KEY trong .env]\nĐã thực hiện trích xuất vector thành công (Retrieval Only). Dưới đây là các đoạn văn bản liên quan nhất:",
            "retrieved_chunks": retrieved_chunks
        }

    # 5. Gửi cho Gemini LLM (model gemini-flash-lite-latest)
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""Bạn là chuyên viên tư vấn dựa trên tài liệu. Hãy trả lời câu hỏi dưới đây chỉ dựa vào ngữ cảnh (Context) được cung cấp. Bắt buộc dẫn nguồn cụ thể [1], [2] tương ứng.

Context:
{context_str}

Câu hỏi: {question}

Trả lời:"""

        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt
        )
        answer = response.text
    except Exception as e:
        answer = f"Lỗi khi sinh câu trả lời từ Gemini API: {str(e)}"

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }

# ----------------------------------------------------
# Core Function 3: status()
# ----------------------------------------------------
def status():
    """
    Trả về số lượng document độc bản và số lượng chunk hiện có trong DB.
    """
    try:
        init_db()
        conn, _ = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT source), COUNT(*) FROM chunks")
        doc_count, chunk_count = cursor.fetchone()
        conn.close()
        return {
            "documents": doc_count or 0,
            "chunks": chunk_count or 0
        }
    except Exception:
        return {"documents": 0, "chunks": 0}
