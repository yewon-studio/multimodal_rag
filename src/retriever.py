"""③ 저장·검색·답변: MultiVectorRetriever(요약으로 검색 → 원본 반환)와 최종 체인."""
import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ID_KEY = "doc_id"


def create_multi_vector_retriever(collection_name, k=4, backend="memory"):
    """요약은 벡터 저장소(색인 카드 서랍), 원본은 InMemoryStore(서가)에 저장하는 검색기.

    backend="chroma"  : 교안 구성. 단, 이 환경(chromadb 1.x, Windows)에서는 노트북 커널에
                        hi_res 레이아웃 모델이 올라간 뒤 similarity_search가 간헐적으로 빈 결과를
                        반환하는 문제가 있었습니다 (3회 실행 중 2회, 단독 스크립트에서는 재현 안 됨).
    backend="memory"  : langchain_core.InMemoryVectorStore (기본값). 문서 수가 66개로 작아
                        전수 코사인 비교로도 충분히 빠르고, 결과가 항상 결정적입니다.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if backend == "chroma":
        vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings)
    else:
        vectorstore = InMemoryVectorStore(embedding=embeddings)
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
        search_kwargs={"k": k},
    )


def add_documents(retriever, doc_summaries, doc_contents, kind):
    """요약과 원본을 같은 doc_id로 짝지어 저장합니다. kind는 근거 종류 추적용(text/table/image)."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i], "kind": kind})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def looks_like_base64(s):
    return re.match(r"^[A-Za-z0-9+/]{100,}={0,2}$", s) is not None


def split_image_text_types(docs):
    b64_images, texts = [], []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        if looks_like_base64(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict, language="한국어"):
    messages = []
    for image in data_dict["context"]["images"]:
        # detail="high": 막대 위 작은 숫자 라벨을 읽으려면 고해상도 타일 분석이 필요
        # (auto일 때 산업별 취업자수 차트에서 인접 막대 값을 잘못 읽는 오류가 있었음)
        messages.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}}
        )
    formatted_texts = "\n\n".join(data_dict["context"]["texts"])
    messages.append(
        {
            "type": "text",
            "text": (
                f"당신은 한국은행 경제상황 평가 보고서를 분석하는 애널리스트입니다.\n"
                f"아래 자료(이미지 차트와 텍스트)만 근거로 질문에 {language}로 답하세요.\n"
                "이미지가 여러 장이면 각 이미지 상단의 차트 제목을 먼저 읽고, 질문과 제목이 일치하는 차트만 근거로 쓰세요.\n"
                "차트에서 읽은 수치는 어느 차트(제목)에서 읽었는지 밝히세요.\n"
                "자료에 없는 내용은 모른다고 답하세요.\n\n"
                f"질문: {data_dict['question']}\n\n"
                f"참고 텍스트:\n{formatted_texts}"
            ),
        }
    )
    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever, language="한국어", model_name="gpt-4o-mini"):
    model = ChatOpenAI(temperature=0, model=model_name, max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(lambda d: img_prompt_func(d, language))
        | model
        | StrOutputParser()
    )


def build_translate_chain(model=None):
    """영문 답변 → 한국어 번역 체인. 기본은 gpt-4o-mini, 로컬 Ollama 모델을 넘겨도 됩니다."""
    prompt = ChatPromptTemplate.from_template(
        "다음 영문 답변을 자연스러운 한국어로 번역하세요. 숫자와 고유명사는 그대로 두세요.\n\n{answer}"
    )
    model = model or ChatOpenAI(temperature=0, model="gpt-4o-mini")
    return {"answer": RunnablePassthrough()} | prompt | model | StrOutputParser()
