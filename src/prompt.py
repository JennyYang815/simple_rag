def build_prompt(question, results):
    context_parts = []

    for result in results:
        context_parts.append(
            f"[第 {result['page']} 页]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
你是一个文档问答助手。
请仅根据下面提供的参考资料回答问题。
如果参考资料中没有足够的信息，请回答“根据当前资料无法回答”，不要自己编造内容。

用户问题：
{question}

参考资料：
{context}

请回答：
"""

    return prompt