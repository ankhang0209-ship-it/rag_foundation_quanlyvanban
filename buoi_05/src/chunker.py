"""
MODULE CHUNKING VĂN BẢN VỚI 3 CHIẾN LƯỢC (BUỔI 5)
File: RAG/rag_foundation/buoi_05/src/chunker.py
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    ocr_used: bool = False
    language: str = "vi"
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None

class ChunkItem(BaseModel):
    chunk_id: str
    strategy: str
    source: str
    page_start: int
    page_end: int
    text: str
    metadata: ChunkMetadata

def Path_basename(path_str: str) -> str:
    from pathlib import Path
    return Path(path_str).stem

# ---------------------------------------------------------
# 1. CHIẾN LƯỢC FIXED-SIZE
# ---------------------------------------------------------
def chunk_fixed_size(
    text: str,
    source: str,
    ocr_used: bool = False,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[ChunkItem]:
    chunks = []
    if not text:
        return chunks
        
    start = 0
    step = max(chunk_size - overlap, 50)
    idx = 1
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end].strip()
        
        if chunk_str:
            c_id = f"fixed_{Path_basename(source)}_{idx:03d}"
            chunks.append(ChunkItem(
                chunk_id=c_id,
                strategy="fixed-size",
                source=source,
                page_start=1,
                page_end=1,
                text=chunk_str,
                metadata=ChunkMetadata(ocr_used=ocr_used, language="vi")
            ))
            idx += 1
            
        start += step
        if end >= len(text):
            break
            
    return chunks

# ---------------------------------------------------------
# 2. CHIẾN LƯỢC SEMANTIC
# ---------------------------------------------------------
def chunk_semantic(
    text: str,
    source: str,
    ocr_used: bool = False,
    max_chunk_size: int = 600
) -> List[ChunkItem]:
    chunks = []
    if not text:
        return chunks
        
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    current_chunk = []
    current_length = 0
    idx = 1
    
    for p in paragraphs:
        if current_length + len(p) <= max_chunk_size:
            current_chunk.append(p)
            current_length += len(p) + 2
        else:
            if current_chunk:
                c_text = "\n\n".join(current_chunk)
                c_id = f"semantic_{Path_basename(source)}_{idx:03d}"
                chunks.append(ChunkItem(
                    chunk_id=c_id,
                    strategy="semantic",
                    source=source,
                    page_start=1,
                    page_end=1,
                    text=c_text,
                    metadata=ChunkMetadata(ocr_used=ocr_used, language="vi")
                ))
                idx += 1
            
            if len(p) > max_chunk_size:
                sentences = re.split(r'(?<=[.?!;])\s+', p)
                sub_chunk = []
                sub_len = 0
                for s in sentences:
                    if sub_len + len(s) <= max_chunk_size:
                        sub_chunk.append(s)
                        sub_len += len(s) + 1
                    else:
                        if sub_chunk:
                            c_text = " ".join(sub_chunk)
                            c_id = f"semantic_{Path_basename(source)}_{idx:03d}"
                            chunks.append(ChunkItem(
                                chunk_id=c_id,
                                strategy="semantic",
                                source=source,
                                page_start=1,
                                page_end=1,
                                text=c_text,
                                metadata=ChunkMetadata(ocr_used=ocr_used, language="vi")
                            ))
                            idx += 1
                        sub_chunk = [s]
                        sub_len = len(s)
                if sub_chunk:
                    current_chunk = sub_chunk
                    current_length = sub_len
            else:
                current_chunk = [p]
                current_length = len(p)

    if current_chunk:
        c_text = "\n\n".join(current_chunk)
        c_id = f"semantic_{Path_basename(source)}_{idx:03d}"
        chunks.append(ChunkItem(
            chunk_id=c_id,
            strategy="semantic",
            source=source,
            page_start=1,
            page_end=1,
            text=c_text,
            metadata=ChunkMetadata(ocr_used=ocr_used, language="vi")
        ))
        
    return chunks

# ---------------------------------------------------------
# 3. CHIẾN LƯỢC HIERARCHICAL (HỖ TRỢ CẢ MARKDOWN HEADERS & PLAIN TEXT)
# ---------------------------------------------------------
def chunk_hierarchical(
    text: str,
    source: str,
    ocr_used: bool = False
) -> List[ChunkItem]:
    chunks = []
    if not text:
        return chunks
        
    # Regex nhận diện Chương, Mục, Điều (bao gồm cả ký tự Markdown #, ##, ###)
    re_chapter = re.compile(r'^(?:#+\s*)?(?:CHƯƠNG|Chương)\s+([IVXLCDM\d]+)', re.MULTILINE)
    re_section = re.compile(r'^(?:#+\s*)?(?:MỤC|Mục)\s+(\d+|[IVXLCDM]+)', re.MULTILINE)
    re_article = re.compile(r'^(?:#+\s*)?(?:ĐIỀU|Điều)\s+(\d+)[\.\:]?', re.MULTILINE)
    
    articles_found = list(re_article.finditer(text))
    
    if not articles_found:
        print(f"   [CẢNH BẢO HIERARCHICAL] Tệp '{source}' không tìm thấy mốc 'Điều...'. Không tự bịa heading. Ngắt theo đoạn tự nhiên.")
        return chunk_semantic(text, source, ocr_used=ocr_used, max_chunk_size=800)
        
    lines = text.split('\n')
    idx = 1
    
    curr_chapter = None
    curr_section = None
    curr_article = None
    curr_lines = []
    
    def save_hierarchical_chunk():
        nonlocal idx, curr_lines
        if curr_lines:
            c_text = "\n".join(curr_lines).strip()
            if c_text:
                c_id = f"hierarchical_{Path_basename(source)}_{idx:03d}"
                chunks.append(ChunkItem(
                    chunk_id=c_id,
                    strategy="hierarchical",
                    source=source,
                    page_start=1,
                    page_end=1,
                    text=c_text,
                    metadata=ChunkMetadata(
                        ocr_used=ocr_used,
                        language="vi",
                        chapter=curr_chapter,
                        section=curr_section,
                        article=curr_article
                    )
                ))
                idx += 1
            curr_lines = []

    for line in lines:
        line_str = line.strip()
        
        m_chap = re_chapter.match(line_str)
        if m_chap:
            save_hierarchical_chunk()
            curr_chapter = line_str.lstrip('#').strip()
            continue
            
        m_sec = re_section.match(line_str)
        if m_sec:
            save_hierarchical_chunk()
            curr_section = line_str.lstrip('#').strip()
            continue
            
        m_art = re_article.match(line_str)
        if m_art:
            save_hierarchical_chunk()
            curr_article = line_str.lstrip('#').strip()
            curr_lines.append(line_str)
            continue
            
        curr_lines.append(line)
        
    save_hierarchical_chunk()
    return chunks
