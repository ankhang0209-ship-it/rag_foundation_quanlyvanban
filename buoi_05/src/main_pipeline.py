"""
KỊCH BẢN CHÍNH THỰC THI PIPELINE OCR VÀ CHUNKING (BUỔI 5)
File: RAG/rag_foundation/buoi_05/src/main_pipeline.py
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Ensure import relative from src
SRC_DIR = Path(__file__).parent.resolve()
BUOI_05_DIR = SRC_DIR.parent.resolve()
sys.path.insert(0, str(SRC_DIR))

from ocr_processor import process_pdf_document
from chunker import chunk_fixed_size, chunk_semantic, chunk_hierarchical, ChunkItem

DATA_DEMO_DIR = BUOI_05_DIR / "datademo"
OUTPUT_DIR = BUOI_05_DIR / "output"

def compute_stats(chunks: List[ChunkItem]) -> Dict[str, Any]:
    if not chunks:
        return {"count": 0, "min_len": 0, "max_len": 0, "avg_len": 0}
    lengths = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 1)
    }

async def run_pipeline(dry_run: bool = True, target_pdf: str = None):
    print("=" * 75)
    mode_str = "DRY-RUN (CHỈ PREVIEW & THỐNG KÊ)" if dry_run else "WRITE (GHI DỮ LIỆU VÀO OUTPUT)"
    print(f" KHỞI CHẠY LUỒNG RAG OCR & CHUNKING BUỔI 05 — CHẾ ĐỘ: {mode_str}")
    print("=" * 75)
    
    if not DATA_DEMO_DIR.exists():
        print(f"[LỖI CRITICAL] Không tìm thấy thư mục dữ liệu: {DATA_DEMO_DIR}")
        return

    pdf_files = list(DATA_DEMO_DIR.glob("*.pdf"))
    if target_pdf:
        pdf_files = [f for f in pdf_files if f.name == target_pdf or f.stem == target_pdf]

    if not pdf_files:
        print(f"[LỖI] Không tìm thấy tệp PDF phù hợp trong {DATA_DEMO_DIR}")
        return

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[THÔNG BÁO] Thư mục đầu ra sẵn sàng: {OUTPUT_DIR}")

    all_reports = []

    for pdf_path in pdf_files:
        # Step 1-4: Read PDF, Quality Check, Fallback OCR, NFC Normalize
        pages_info, ocr_used, raw_text = await process_pdf_document(pdf_path)
        
        # Step 5: Chunking với 3 chiến lược
        fixed_chunks = chunk_fixed_size(raw_text, pdf_path.name, ocr_used=ocr_used)
        semantic_chunks = chunk_semantic(raw_text, pdf_path.name, ocr_used=ocr_used)
        hierarchical_chunks = chunk_hierarchical(raw_text, pdf_path.name, ocr_used=ocr_used)
        
        fixed_stats = compute_stats(fixed_chunks)
        semantic_stats = compute_stats(semantic_chunks)
        hierarchical_stats = compute_stats(hierarchical_chunks)
        
        report = {
            "file_name": pdf_path.name,
            "ocr_used": ocr_used,
            "raw_text_length": len(raw_text),
            "strategies": {
                "fixed-size": fixed_stats,
                "semantic": semantic_stats,
                "hierarchical": hierarchical_stats
            }
        }
        all_reports.append(report)
        
        # In thống kê từng file
        print(f"\n--- BÁO CÁO CHUNKING CHO: {pdf_path.name} ---")
        print(f" - Độ dài văn bản thô (NFC): {len(raw_text):,} ký tự")
        print(f" - Chế độ OCR được kích hoạt: {ocr_used}")
        print(f" - [Fixed-Size]   : {fixed_stats['count']} chunks | Min: {fixed_stats['min_len']} | Max: {fixed_stats['max_len']} | Avg: {fixed_stats['avg_len']}")
        print(f" - [Semantic]     : {semantic_stats['count']} chunks | Min: {semantic_stats['min_len']} | Max: {semantic_stats['max_len']} | Avg: {semantic_stats['avg_len']}")
        print(f" - [Hierarchical] : {hierarchical_stats['count']} chunks | Min: {hierarchical_stats['min_len']} | Max: {hierarchical_stats['max_len']} | Avg: {hierarchical_stats['avg_len']}")
        
        # Save output if write mode
        if not dry_run:
            file_stem = pdf_path.stem
            
            # Save raw text
            raw_out_path = OUTPUT_DIR / f"{file_stem}_raw.txt"
            raw_out_path.write_text(raw_text, encoding="utf-8")
            
            # Save chunks JSON
            chunks_payload = {
                "source": pdf_path.name,
                "ocr_used": ocr_used,
                "fixed_size_chunks": [c.model_dump() for c in fixed_chunks],
                "semantic_chunks": [c.model_dump() for c in semantic_chunks],
                "hierarchical_chunks": [c.model_dump() for c in hierarchical_chunks]
            }
            chunks_out_path = OUTPUT_DIR / f"{file_stem}_chunks.json"
            chunks_out_path.write_text(json.dumps(chunks_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   [ĐÃ GHI] -> Raw: {raw_out_path.name} | Chunks: {chunks_out_path.name}")

    print("\n" + "=" * 75)
    print(" TỔNG HỢP PIPELINE HOÀN TẤT")
    print("=" * 75)
    
    if dry_run:
        print("\n[LƯU Ý] Đây là chạy thử (DRY-RUN). Để lưu tệp vào folder output/, vui lòng thêm cờ `--write`.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG OCR & Chunking Buổi 5")
    parser.add_argument("--write", action="store_true", help="Ghi kết quả ra thư mục output/")
    parser.add_argument("--dry-run", action="store_true", help="Chạy xem trước và in thống kê mà không ghi tệp")
    parser.add_argument("--pdf", type=str, default=None, help="Tên tệp PDF cụ thể cần chạy")
    
    args = parser.parse_args()
    is_write = args.write
    dry_run = not is_write
    
    asyncio.run(run_pipeline(dry_run=dry_run, target_pdf=args.pdf))

if __name__ == "__main__":
    main()
