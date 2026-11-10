"""② 요약 생성: 텍스트·표는 LLM, 이미지는 VLM(gpt-4o-mini)으로 검색용 요약을 만듭니다."""
import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

TEXT_PROMPT_KOR = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
지표 이름, 기간(분기·연도), 핵심 수치는 반드시 포함하세요.
요약할 표 또는 텍스트: {element}"""

IMAGE_PROMPT_KOR = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
Include the chart title, axis units, series names, time range, and the key values or trends.
The summary should be written in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=False, model_name="gpt-4o-mini"):
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT_KOR)
    model = ChatOpenAI(temperature=0, model=model_name)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries, table_summaries = [], []
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    return text_summaries, table_summaries


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_summarize(img_base64, prompt, model_name="gpt-4o-mini"):
    chat = ChatOpenAI(model=model_name, max_tokens=1024, temperature=0)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """폴더의 이미지를 base64로 인코딩하고 VLM 요약을 만듭니다. (파일명, base64, 요약) 순서 유지."""
    img_files, img_base64_list, image_summaries = [], [], []
    for img_file in sorted(os.listdir(path)):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            b64 = encode_image(os.path.join(path, img_file))
            img_files.append(img_file)
            img_base64_list.append(b64)
            image_summaries.append(image_summarize(b64, IMAGE_PROMPT_KOR))
    return img_files, img_base64_list, image_summaries
