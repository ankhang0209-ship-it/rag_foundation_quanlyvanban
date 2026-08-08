"""
MODULE XỬ LÝ ĐỌC VĂN BẢN PDF VÀ OCR (BUỔI 5)
File: RAG/rag_foundation/buoi_05/src/ocr_processor.py
"""

import os
import re
import asyncio
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load môi trường từ src/.env
SRC_DIR = Path(__file__).parent.resolve()
ENV_FILE = SRC_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)

def normalize_vietnamese_nfc(text: str) -> str:
    """
    Chuẩn hóa chuỗi ký tự tiếng Việt về chuẩn Unicode NFC.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', normalized)
    return normalized.strip()

def check_text_quality(text: str) -> Tuple[bool, str]:
    """
    Kiểm tra chất lượng văn bản trích xuất từ PyMuPDF text layer.
    Phát hiện rỗng, lỗi font (mojibake), mất dấu tiếng Việt (Didu, Chuang, Ph4m, tli6u).
    """
    if not text or len(text.strip()) < 30:
        return False, "Văn bản rỗng hoặc quá ngắn (dưới 30 ký tự)"
    
    # 1. Kiểm tra ký tự lỗi font / encoding dạng mojibake hoặc thay thế \ufffd
    garbage_chars = text.count('\ufffd') + text.count('?')
    if garbage_chars / max(len(text), 1) > 0.10:
        return False, "Số lượng ký tự không đọc được (lỗi font/encoding) chiếm > 10%"
    
    # 2. Phát hiện lỗi font mã hóa tiếng Việt dạng "Didu", "Chuang", "Ph4m", "tli6u", "thdng"
    font_corruption_patterns = [
        r'\bDidu\b', r'\bChuang\b', r'\bPh4m\b', r'\btli6u\b', r'\bthdng\b',
        r'\bndy\b', r'\bho4t\b', r'\bkh6khin\b', r'\bt6 chirc\b', r'\bngdn hirng\b'
    ]
    corrupted_matches = 0
    for pat in font_corruption_patterns:
        if re.search(pat, text, re.IGNORECASE):
            corrupted_matches += 1
            
    if corrupted_matches >= 2:
        return False, f"Phát hiện lỗi mã hóa font tiếng Việt (Mojibake: Didu/Chuang/Ph4m/tli6u - {corrupted_matches} mẫu phát hiện)"
    
    # 3. Kiểm tra tỷ lệ chữ cái tiếng Việt chuẩn có dấu (Unicode NFC/NFD)
    vietnamese_accents = re.findall(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]', text)
    accent_ratio = len(vietnamese_accents) / max(len(text), 1)
    
    # Một văn bản tiếng Việt chuẩn thường có tỷ lệ nguyên âm có dấu > 4%
    if accent_ratio < 0.03:
        return False, f"Tỷ lệ ký tự tiếng Việt có dấu quá thấp ({accent_ratio:.1%}), văn bản bị sai mã hóa font."

    return True, "Chất lượng text layer đạt chuẩn"

def extract_text_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Trích xuất text layer từng trang bằng PyMuPDF.
    """
    pages_data = []
    doc = fitz.open(pdf_path)
    
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        raw_text = page.get_text("text")
        clean_text = normalize_vietnamese_nfc(raw_text)
        is_valid, reason = check_text_quality(clean_text)
        
        pages_data.append({
            "page_number": page_idx + 1,
            "text": clean_text,
            "is_valid": is_valid,
            "quality_reason": reason,
            "ocr_used": False
        })
    doc.close()
    return pages_data

async def ocr_with_llamaparse(pdf_path: Path) -> str:
    """
    Thực hiện OCR toàn bộ file PDF bằng LlamaParse từ llama-cloud.
    """
    api_key = os.getenv("LLAMA_CLOUD_API_KEY", "").strip()
    if not api_key or api_key == "KEY CỦA BẠN":
        print(f"   [CẢNH BÁO OCR] API Key mẫu chưa được thiết lập trong src/.env. Không thể gọi LlamaParse API.")
        return ""
    
    try:
        from llama_cloud import AsyncLlamaCloud
        client = AsyncLlamaCloud(api_key=api_key)
        
        print(f"   [LLAMAPARSE] Đang gửi tệp {pdf_path.name} lên LlamaCloud để OCR...")
        file_obj = await client.files.create(file=str(pdf_path), purpose="parse")
        
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version='latest',
            expand=["markdown_full"],
        )
        raw_markdown = result.markdown_full or ""
        return normalize_vietnamese_nfc(raw_markdown)
    except Exception as e:
        print(f"   [LỖI OCR] Lỗi kết nối LlamaParse cho {pdf_path.name}: {str(e)}")
        return ""

async def process_pdf_document(pdf_path: Path, force_ocr: bool = False) -> Tuple[List[Dict[str, Any]], bool, str]:
    """
    Luồng xử lý hoàn chỉnh 1 file PDF:
    (1) Đọc PyMuPDF
    (2) Kiểm tra chất lượng text layer & lỗi mã hóa font
    (3) Fallback OCR bằng LlamaParse khi text lỗi hoặc khi force_ocr=True
    (4) Chuẩn hóa Unicode NFC
    """
    pdf_name = pdf_path.name
    print(f"\n---> Đang xử lý tài liệu: {pdf_name}")
    
    pages = extract_text_pymupdf(pdf_path)
    invalid_pages = [p for p in pages if not p["is_valid"]]
    
    overall_ocr_used = False
    full_text = ""
    
    should_run_ocr = force_ocr or (len(invalid_pages) > 0)
    
    if should_run_ocr:
        if force_ocr:
            print(f"   [PHÁT HIỆN] Yêu cầu ép buộc OCR bằng LlamaParse cho tệp {pdf_name}.")
        else:
            print(f"   [PHÁT HIỆN] Tìm thấy {len(invalid_pages)}/{len(pages)} trang bị lỗi font/mojibake/rỗng.")
            print(f"   [LÝ DO LỖI FONT] {invalid_pages[0]['quality_reason']}")
        
        # Gọi LlamaParse API
        ocr_text = await ocr_with_llamaparse(pdf_path)
        if ocr_text:
            overall_ocr_used = True
            full_text = ocr_text
            pages = [{
                "page_number": 1,
                "text": ocr_text,
                "is_valid": True,
                "quality_reason": "Đã OCR qua LlamaParse thành công (Đạt chuẩn Unicode NFC)",
                "ocr_used": True
            }]
        else:
            print("   [CHÚ Ý] LlamaParse không phản hồi. Giữ lại kết quả PyMuPDF tốt nhất sẵn có.")
            full_text = "\n\n".join([p["text"] for p in pages if p["text"]])
    else:
        print(f"   [THÀNH CÔNG] Tất cả {len(pages)} trang có text layer PyMuPDF đạt chuẩn. Tránh OCR không cần thiết.")
        full_text = "\n\n".join([p["text"] for p in pages])
        
    full_text = normalize_vietnamese_nfc(full_text)
    return pages, overall_ocr_used, full_text
