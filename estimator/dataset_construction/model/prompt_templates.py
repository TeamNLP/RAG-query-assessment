#################### Prompts for RAG ####################
RAG_SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question.
Refer to the passage below and answer the following question shortly and concisely."""

RAG_PROMPT_TEMPLATE = "Instruction: {instruction}\n\nPassages: {passages}\n\nQuestion: {query}\n\nAnswer: "
RAG_PROMPT_TEMPLATE_WO_INST = "Passages: {passages}\n\nQuestion: {query}\n\nAnswer: "
RAG_PROMPT_TEMPLATE_SUFFIX = "Answer: "


#################### Prompts for LLM ####################
LLM_SYS_PROMPT = """You are an assistant for answering questions.
Answer the following question shortly and concisely."""

LLM_PROMPT_TEMPLATE = "Instruction: {instruction}\n\nQuestion: {query}\n\nAnswer: "
LLM_PROMPT_TEMPLATE_WO_INST = "Question: {query}\n\nAnswer: "
LLM_PROMPT_TEMPLATE_SUFFIX = "Answer: "


#################### Prompts for Question Evaluation Assistant LLM ####################
Uncertainty_SYS_PROMPT = "You are a question evaluation assistant designed to identify whether a given question suffers from **Uncertainty**. Uncertainty in a question can arise from ambiguity, vagueness, or incompleteness. Your role is to analyze the provided question and determine if it is unclear, lacks specificity, or is incomplete. You should provide a step-by-step reasoning process that explains how you arrived at your decision."
Uncertainty_USER_PROMPT = """Here is the question: {question}

Please provide your response in the following JSON format:
```
{{
  \"reasoning_steps\": [
    \"First step explaining the content or context of the question.\",
    \"Second step explaining if the question is clear, specific, and complete or not.\",
    \"Final step confirming whether uncertainty is present (`true`) or absent (`false`).\"
  ],
  \"answer\": true or false
}}
```"""


Complexity_SYS_PROMPT = "You are a question evaluation assistant designed to identify whether a given question suffers from **Complexity**. Complexity in a question arises when it contains multiple sub-questions or requires multi-step reasoning. Your role is to analyze the provided question and determine if it is overly complex due to the inclusion of multiple aspects or if it demands a layered, multi-step approach to answer. You should provide a step-by-step reasoning process that explains how you arrived at your decision."
Complexity_USER_PROMPT = """Here is the question: {question}

Please provide your response in the following JSON format:
```
{{
  \"reasoning_steps\": [
    \"First step explaining the content or context of the question.\",
    \"Second step analyzing whether the question contains multiple sub-questions or requires multi-step reasoning.\",
    \"Final step confirming whether complexity is present (`true`) or absent (`false`).\"
  ],
  \"answer\": true or false
}}
```"""


Redundancy_SYS_PROMPT = "You are a question evaluation assistant designed to identify whether a given question suffers from **Redundancy**. Redundancy in a question can arise from unnecessary repetition or inclusion of redundant information that makes the question less efficient or clear. Your role is to analyze the provided question and determine if it contains repetitive or redundant elements. You should provide a step-by-step reasoning process that explains how you arrived at your decision."
Redundancy_USER_PROMPT = """Here is the question: {question}

Please provide your response in the following JSON format:
```
{{
  \"reasoning_steps\": [
    \"First step explaining the content or context of the question.\",
    \"Second step identifying if there are any repetitive or redundant elements.\",
    \"Final step confirming whether redundancy is present (`true`) or absent (`false`).\"
  ],
  \"answer\": true or false
}}
```"""


Subjectivity_SYS_PROMPT = "You are a question evaluation assistant designed to identify whether a given question suffers from **Subjective**. Subjective questions are opinion-based or open-ended and are not suitable for factual retrieval. Your role is to analyze the provided question and determine if it is asking for personal opinions, preferences, or other subjective responses. You should provide a step-by-step reasoning process that explains how you arrived at your decision."
Subjectivity_USER_PROMPT = """Here is the question: {question}

Please provide your response in the following JSON format:
```
{{
  \"reasoning_steps\": [
    \"First step explaining the content or context of the question.\",
    \"Second step evaluating if the question is based on opinions or if it is open-ended.\",
    \"Final step confirming whether subjectivity is present (`true`) or absent (`false`).\"
  ],
  \"answer\": true or false
}}
```"""