#################### Prompts for RAG ####################
RAG_SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question.
Refer to the passage below and answer the following question shortly and concisely."""

RAG_PROMPT_TEMPLATE = "Instruction: {instruction}\n\nPassages: {passages}\n\nQuestion: {query}\n\nAnswer: "
RAG_PROMPT_TEMPLATE_WO_INST = "Passages: {passages}\n\nQuestion: {query}\n\nAnswer: "
RAG_PROMPT_TEMPLATE_SUFFIX = "Answer: "