import streamlit as st

import hashlib
import io

from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from prompt import build_prompt
from llm import generate_answer

from pypdf import PdfReader

from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD
)


st.set_page_config(
    page_title="Simple RAG",
)


st.title("Simple RAG")

st.write("一个基于PDF文档的简单RAG问答系统")


@st.cache_resource(show_spinner=False)
def initialize_rag(file_bytes, skip_pages):
    pdf_file = io.BytesIO(file_bytes)

    pages = load_pdf(
        pdf_file,
        skip_pages=list(skip_pages)
        )

    chunks = split_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()[:16]
    
    skip_key = "-".join(
        str(page) for page in skip_pages
    )

    if not skip_key:
        skip_key = "none"

    retriever = Retriever(
        chunks,
        model_name=EMBEDDING_MODEL,
        top_k=TOP_K,
        threshold=SIMILARITY_THRESHOLD,
        cache_dir=f"cache/uploads/{file_hash}/{skip_key}"
    )

    return retriever, pages, chunks


uploaded_file = st.file_uploader(
    "上传PDF文档",
    type=["pdf"]
)


if uploaded_file is None:
    st.info("请先上传一个PDF文档。")
    st.stop()


file_bytes = uploaded_file.getvalue()

pdf_reader = PdfReader(
    io.BytesIO(file_bytes)
)

total_pages = len(pdf_reader.pages)

st.write(
    f"PDF共{total_pages}页"
)

skip_pages = st.multiselect(
    "选择需要跳过的页面：",
    options=list(
        range(1, total_pages + 1)
    ),
    format_func=lambda page: f"第{page}页"
)

with st.spinner("正在解析PDF并建立知识库..."):
    try:
        retriever, pages, chunks = initialize_rag(
            file_bytes,
            tuple(skip_pages)
        )

    except Exception as error:
        st.error(
            f"PDF处理失败：{error}"
        )
        st.stop()


st.success("PDF知识库建立完成")

st.write(
    f"文件：{uploaded_file.name}"
)

st.write(
    f"实际读取{len(pages)}页，生成{len(chunks)}个Chunk"
)

question = st.text_input(
    "请输入你的问题：",
    placeholder="请输入与当前PDF相关的问题"
)

if st.button("提问", type="primary"):

    if not question.strip():
        st.warning("请先输入一个问题。")

    else:
        with st.spinner("正在检索相关内容..."):
            results = retriever.retrieve(question)

        if len(results) == 0:
            st.warning(
                "没有找到足够相关的资料。"
            )

        else:
            prompt = build_prompt(
                question,
                results
            )

            with st.spinner("正在生成回答..."):
                answer = generate_answer(prompt)

            st.subheader("回答")

            st.write(answer)


            source_pages = sorted(
                set(
                    result["page"]
                    for result in results
                )
            )

            source_text = "、".join(
                f"第{page}页"
                for page in source_pages
            )

            st.caption(
                f"参考来源：{source_text}"
            )


            with st.expander(
                "查看检索结果"
            ):

                for rank, result in enumerate(
                    results,
                    start=1
                ):
                    st.markdown(
                        f"#### 第{rank}条"
                    )

                    st.write(
                        f"页码：第{result['page']}页"
                    )

                    st.write(
                        f"相似度："
                        f"{result['score']:.4f}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()