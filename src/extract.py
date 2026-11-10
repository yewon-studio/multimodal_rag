"""① 문서 파싱: PDF → 텍스트 / 표 / 이미지 분리."""
import os
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from PIL import Image
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(pdf_path, figures_dir, vertical_pad=110):
    """PDF에서 텍스트 블록·표를 추출하고, 이미지는 figures_dir에 .jpg로 저장합니다."""
    os.makedirs(figures_dir, exist_ok=True)
    # 이 보고서는 차트 제목("[그림 13] 배당소득에 대한 품목별 소비반응")이 그림 바로 위 텍스트라
    # 기본 crop에는 제목이 빠집니다. 위아래로 여백을 줘서 제목·출처까지 이미지에 포함시킵니다.
    # (제목이 없으면 VLM 요약도, 답변 모델도 어떤 차트인지 구분하지 못함)
    os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = str(vertical_pad)
    os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "20"
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                    # 레이아웃 분석 모델 사용 (표·이미지 정확도 핵심)
        languages=["kor", "eng"],             # 한국어 보도자료 → OCR 언어 지정
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],  # 그래프/그림 블록을 파일로 저장
        extract_image_block_output_dir=str(figures_dir),
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )


def categorize_elements(raw_pdf_elements):
    """unstructured 요소를 텍스트(CompositeElement)와 표(Table)로 분류합니다."""
    texts, tables = [], []
    for element in raw_pdf_elements:
        type_name = str(type(element))
        if "unstructured.documents.elements.Table" in type_name:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in type_name:
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트를 토큰 기준으로 다시 분할합니다 (경계에서 문맥이 끊기지 않도록 overlap)."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))


def remove_small_images(figures_dir, min_side=150):
    """로고·아이콘처럼 작은 이미지는 검색 노이즈이므로 삭제합니다. 삭제한 파일 목록을 반환."""
    removed = []
    for p in sorted(Path(figures_dir).glob("*.jpg")):
        with Image.open(p) as im:
            w, h = im.size
        if min(w, h) < min_side:
            p.unlink()
            removed.append((p.name, w, h))
    return removed
