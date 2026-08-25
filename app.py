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

# 侧边栏
st.sidebar.header("RAG参数")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

overlap = st.sidebar.slider(
    "Overlap",
    min_value=0,
    max_value=100,
    value=50,
    step=10
)

top_k = st.sidebar.slider(
    "Top-K",
    min_value=1,
    max_value=5,
    value=3
)

threshold = st.sidebar.slider(
    "相似度阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.54,
    step=0.01
)

if overlap >= chunk_size:
    st.error("Overlap必须小于Chunk Size。")
    st.stop()
    

@st.cache_resource(show_spinner=False)
def initialize_rag(
    file_bytes,
    skip_pages,
    chunk_size,
    overlap,
    top_k,
    threshold
):
    pdf_file = io.BytesIO(file_bytes)

    pages = load_pdf(
        pdf_file,
        skip_pages=list(skip_pages)
        )

    chunks = split_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()[:16]
    
    skip_key = "-".join(
        str(page) for page in skip_pages
    )

    if not skip_key:
        skip_key = "none"
        
    cache_dir = (
        f"cache/uploads/{file_hash}/{skip_key}"
        f"chunk_{chunk_size}_"
        f"overlap_{overlap}"
    )

    retriever = Retriever(
        chunks,
        model_name=EMBEDDING_MODEL,
        top_k=top_k,
        threshold=threshold,
        cache_dir=cache_dir
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
            tuple(skip_pages),
            chunk_size,
            overlap,
            top_k,
            threshold
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

st.caption(
    f"Chunk Size={chunk_size} | "
    f"Overlap={overlap} | "
    f"Top-K={top_k} | "
    f"Threshold={threshold:.2f}"
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