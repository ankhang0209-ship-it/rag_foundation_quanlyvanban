import os
import sys
import json
import urllib.request
from pathlib import Path

# Tự động nạp venv site-packages của Buổi 05 vào sys.path
BASE_DIR = Path(__file__).parent.resolve()
VENV_SITE_PACKAGES = BASE_DIR.parent / "buoi_05" / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

from dotenv import load_dotenv  # type: ignore
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

results = {
    "python_path": sys.executable,
    "packages": {},
    "chroma_status": None,
    "postgres_status": None,
    "rag_db_status": None,
    "user_actions": []
}

# 1. Package check
packages_to_check = [
    ("streamlit", "streamlit"),
    ("google-genai", "google.genai"),
    ("chromadb", "chromadb"),
    ("psycopg", "psycopg"),
    ("python-dotenv", "dotenv")
]

for pkg_name, import_name in packages_to_check:
    try:
        if pkg_name == "google-genai":
            from google import genai  # type: ignore
        else:
            __import__(import_name)
        results["packages"][pkg_name] = "PASS"
    except Exception as e:
        results["packages"][pkg_name] = f"FAIL: {str(e)}"

# 2. ChromaDB check
try:
    chroma_server_found = False
    for url in ["http://localhost:8000", "http://localhost:8001"]:
        try:
            req = urllib.request.urlopen(f"{url}/api/v1/heartbeat", timeout=2)
            if req.status == 200:
                results["chroma_status"] = f"Server (Đang chạy tại {url})"
                chroma_server_found = True
                break
        except Exception:
            pass

    if not chroma_server_found:
        import chromadb  # type: ignore
        storage_dir = Path(__file__).parent / "storage" / "chroma"
        storage_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(storage_dir))
        results["chroma_status"] = f"Embedded Local (Persistent Client lưu tại storage/chroma/)"
except Exception as e:
    results["chroma_status"] = f"ERROR: {str(e)}"

# 3. PostgreSQL check
pg_host = os.getenv("POSTGRES_HOST", "localhost")
pg_port = os.getenv("POSTGRES_PORT", "5432")
pg_user = os.getenv("POSTGRES_USER", "postgres")
pg_pass = os.getenv("POSTGRES_PASSWORD", "")
pg_db = os.getenv("POSTGRES_DB", "rag_db")

try:
    import psycopg  # type: ignore
    # Try connecting to default 'postgres' db
    try:
        conn = psycopg.connect(
            host=pg_host,
            port=int(pg_port),
            user=pg_user,
            password=pg_pass,
            dbname="postgres",
            autocommit=True,
            connect_timeout=3
        )
        results["postgres_status"] = "CONNECTED (PostgreSQL Service đang hoạt động)"

        # Check if rag_db exists
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (pg_db,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f'CREATE DATABASE "{pg_db}";')
                results["rag_db_status"] = f'CREATED (Đã tự động tạo cơ sở dữ liệu "{pg_db}")'
            else:
                results["rag_db_status"] = f'EXISTS (Database "{pg_db}" đã tồn tại)'
        conn.close()

        # Connect to rag_db
        conn_rag = psycopg.connect(
            host=pg_host,
            port=int(pg_port),
            user=pg_user,
            password=pg_pass,
            dbname=pg_db,
            connect_timeout=3
        )
        conn_rag.close()
        results["rag_db_status"] += " -> Kết nối lại tới rag_db: PASS"

    except Exception as pg_err:
        err_msg = str(pg_err)
        if "password authentication failed" in err_msg.lower() or "fe_sendauth" in err_msg.lower():
            results["postgres_status"] = "FAIL (Mật khẩu PostgreSQL không khớp hoặc chưa điền)"
            results["user_actions"].append("Điền chính xác POSTGRES_PASSWORD vào tệp RAG/rag_foundation/buoi_06/.env")
        else:
            results["postgres_status"] = f"CHƯA CÀI ĐẶT HOẶC DỊCH VỤ CHƯA BẬT"
            results["user_actions"].extend([
                "1. Tải PostgreSQL từ trang chính thức: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads",
                "2. Cài đặt và ghi nhớ mật khẩu của user postgres",
                "3. Điền mật khẩu đó vào POSTGRES_PASSWORD trong file RAG/rag_foundation/buoi_06/.env"
            ])
        results["rag_db_status"] = "CHỜ DỊCH VỤ POSTGRESQL (Cần PostgreSQL kết nối thành công trước)"
except Exception as e:
    results["postgres_status"] = f"ERROR: {str(e)}"

print(json.dumps(results, indent=2, ensure_ascii=False))
