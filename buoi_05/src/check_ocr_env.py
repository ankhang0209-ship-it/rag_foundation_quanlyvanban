"""
KỊCH BẢN KIỂM TRA MÔI TRƯỜNG CHO BÀI THỰC HÀNH BUỔI 5 (RAG OCR & CHUNKING)
File: RAG/rag_foundation/buoi_05/src/check_ocr_env.py
"""

import sys
import os
from pathlib import Path

# Đường dẫn tới file .env trong thư mục src
SRC_DIR = Path(__file__).parent.resolve()
ENV_FILE = SRC_DIR / ".env"

def check_package(module_name: str) -> tuple[bool, str]:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "Đã cài đặt")
        return True, str(version)
    except ImportError:
        return False, "Chưa cài đặt"

def run_checks():
    print("=" * 72)
    print(" BẢNG KIỂM TRA MÔI TRƯỜNG XỬ LÝ VĂN BẢN & OCR (BUỔI 05)")
    print("=" * 72)
    
    results = []
    
    # 1. Kiểm tra Python
    py_version = sys.version.split()[0]
    py_pass = sys.version_info >= (3, 10)
    results.append(("Python >= 3.10", py_version, "PASS" if py_pass else "FAIL", "Nâng cấp Python nếu < 3.10"))
    
    # 2. PyMuPDF (fitz)
    status, ver = check_package("fitz")
    results.append(("PyMuPDF (fitz)", ver, "PASS" if status else "FAIL", "pymupdf"))
    
    # 3. Pillow (PIL)
    status, ver = check_package("PIL")
    results.append(("Pillow (PIL)", ver, "PASS" if status else "FAIL", "pillow"))
    
    # 4. Llama Cloud (llama_cloud)
    status, ver = check_package("llama_cloud")
    results.append(("llama-cloud", ver, "PASS" if status else "FAIL", "llama-cloud"))
    
    # 5. Pydantic
    status, ver = check_package("pydantic")
    results.append(("Pydantic", ver, "PASS" if status else "FAIL", "pydantic"))
    
    # 6. Streamlit
    status, ver = check_package("streamlit")
    results.append(("Streamlit", ver, "PASS" if status else "FAIL", "streamlit"))
    
    # 7. python-dotenv
    status, ver = check_package("dotenv")
    results.append(("python-dotenv", ver, "PASS" if status else "FAIL", "python-dotenv"))
    
    # 8. File .env & Key Check (Bảo mật: Không in value secret)
    env_exists = ENV_FILE.exists()
    api_key_set = False
    key_status_msg = "Không tìm thấy file .env"
    
    if env_exists:
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=ENV_FILE)
        except Exception:
            pass
        key = os.getenv("LLAMA_CLOUD_API_KEY", "").strip()
        if key and key != "KEY CỦA BẠN":
            api_key_set = True
            key_status_msg = "Đã có Key (bảo mật: ẩn giá trị)"
        elif key == "KEY CỦA BẠN":
            key_status_msg = "Key mẫu ('KEY CỦA BẠN')"
        else:
            key_status_msg = "Chưa có giá trị Key"
            
    env_pass = env_exists and api_key_set
    results.append((".env API Key", key_status_msg, "PASS" if env_pass else "WARNING", "Cập nhật LLAMA_CLOUD_API_KEY trong src/.env"))

    # In kết quả dạng bảng
    header = f"| {'Công cụ / Kiểm tra':<20} | {'Trạng thái / Version':<30} | {'Kết quả':<8} |"
    divider = "+" + "-"*22 + "+" + "-"*32 + "+" + "-"*10 + "+"
    print(divider)
    print(header)
    print(divider)
    
    fail_count = 0
    missing_packages = []
    
    for item, status_val, res, pkg_name in results:
        status_str = status_val[:30]
        print(f"| {item:<20} | {status_str:<30} | {res:<8} |")
        if res == "FAIL":
            fail_count += 1
            missing_packages.append(pkg_name)
                
    print(divider)
    print("=" * 72)
    
    if fail_count > 0:
        print(f"\n[KHẮC PHỤC] Phát hiện {fail_count} thư viện chưa được cài đặt.")
        print("Lệnh cài đặt bổ sung cho môi trường Python hiện tại:")
        print(f"   pip install {' '.join(missing_packages)}\n")
    else:
        print("\n[THÀNH CÔNG] Tất cả các thư viện bắt buộc đã sẵn sàng!\n")

if __name__ == "__main__":
    run_checks()
