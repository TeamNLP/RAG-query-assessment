RAG_SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question.
Refer to the passage below and answer the following question shortly and concisely.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

RAG_PROMPT_TEMPLATE = "Passages: {passages} \n\n Question: {query} \n\n Answer: "
RAG_PROMPT_TEMPLATE_SUFFIX = "Answer: "
