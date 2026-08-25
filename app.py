import streamlit as st

from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from prompt import build_prompt
from llm import generate_answer

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

st.write(
    "一个基于PDF文档的简单RAG问答系统"
)


@st.cache_resource
def initialize_rag():
    pages = load_pdf(
        PDF_PATH,
        skip_pages=[1]
    )

    chunks = split_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    retriever = Retriever(
        chunks,
        model_name=EMBEDDING_MODEL,
        top_k=TOP_K,
        threshold=SIMILARITY_THRESHOLD
    )

    return retriever


with st.spinner("正在初始化RAG系统..."):
    retriever = initialize_rag()


st.success("RAG系统初始化完成")


question = st.text_input(
    "请输入你的问题：",
    placeholder="例如：Cache利用了什么原理？"
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