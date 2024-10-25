#################### Prompts for RAG ####################
RAG_SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question.
Refer to the passage below and answer the following question shortly and concisely."""

RAG_PROMPT_TEMPLATE = "Instruction: {instruction}\n\nPassages: {passages}\n\nQuestion: {query}\n\nAnswer: "
RAG_PROMPT_TEMPLATE_WO_INST = "Passages: {passages}\n\nQuestion: {query}\n\nAnswer: "
RAG_PROMPT_TEMPLATE_SUFFIX = "Answer: "


#################### Prompts for Query Rewriter ####################
REWRITER_SYSTEM_PROMPT = "You are a Query Rewriter, tasked with reshaping user's query to fit specific requirements for a Retrieval Augmented Generation (RAG) system."
REWRITER_GENERAL_SYSTEM_PROMPT = (
    REWRITER_SYSTEM_PROMPT +
    "\n" +
    "Your job involves reconstructing, supplementing, splitting, adding new words, removing unnecessary words, changing the order, adjusting word sequence, "
    "and controlling sentence length in order to refine the user's query for more accurate responses.\n"
    "You must follow your user's requirements."
)
REWRITER_MULTIFACETED_SYSTEM_PROMPT = (
    REWRITER_SYSTEM_PROMPT +
    "\n" +
    "You must follow your user's requirements."
)

REWRITER_GENERAL_PROMPT = (
    "Requirements are as follows:\n"
)
REWRITER_MULTIFACETED_PROMPT = (
    "I will provide ambiguous questions that can have multiple answers based on their different possible interpretations.\n"
    "Clarify the given question into several disambiguated questions.\n"
    "Here are six common types of multifaceted questions.\n"
    "Please categorize the given question into one of these six types and create \"one\" disambiguation question that matches the type.\n"
    "six types of multifaceted questions : Conditional, Set-Valued, Time Dependent, Underspecified Reference, Underspecified Type, Needs Elaboration\n"
    "Here are some examples.\n"
)

REWRITE_PROMPT_METHOD_1 = (
    REWRITER_GENERAL_PROMPT +
    "\n" +
    "Main Condition: Please make the question more specific.\n"
    "Sub Conditions:\n"
    "Understand the intention behind the question and supplement it with additional explanations or remove unnecessary words to facilitate finding answers in documents.\n"
    "If the question is ambiguous, clarify it through changing the word order, supplementing, or splitting it.\n"
    "If there are typos or grammatical errors in the question, proceed with corrections.\n"
    "\n" +
    "Examples:\n"
    "1. what animal is a possum -> What kind of animal is a possum?\n"
    "2. what animal is a possum -> To which species of animals does a possum belong?\n"
    "3. What is rba? -> What does RBA stand for?\n"
    "4. Please tell me how much the fee is to handle over 5 tons of waste concrete from a construction site. -> Household waste refers to all waste excluding industrial waste. Therefore, the treatment of household waste involves the collection, transport, storage, recycling, and disposal of such waste. Please tell me how much the fee is to handle over 5 tons of waste concrete from a construction site.\n"
)
INPUT_TEMPLATE_METHOD_1 = (
    "Query to rewrite: {query}\n"
    "Rewritten Query:"
)

REWRITE_PROMPT_METHOD_2 = (
    REWRITER_GENERAL_PROMPT +
    "Main Condition: Breaking down a question into steps\n"
    "Sub Conditions:\n"
    "Identify the Main Question: Determine the core problem or question that needs to be answered.\n"
    "Divide into Sub-questions: Break the main question into smaller, more manageable sub-questions.\n"
    "Prioritize Sub-questions: Arrange the sub-questions in a logical order that makes sense for solving the overall problem.\n"
    "\n" +
    "Example1) Query: \"can you take left hand lane to turn right on a dual carriageway roundabout\"\n"
    "Divide into sub-questions:\n"
    "What are the general rules for lane usage on a dual carriageway roundabout?\n"
    "Specifically, can the left lane be used to turn right on such a roundabout?\n"
    "Are there any exceptions or specific signs to look for that allow or disallow such a maneuver?\n"
    "Prioritize the sub-questions in a logical sequence:\n"
    "What are the general rules for lane usage on a dual carriageway roundabout?\n"
    "Are there any exceptions or specific signs to look for that allow or disallow using the left lane to turn right?\n"
    "Specifically, can the left lane be used to turn right on such a roundabout?\n"
    "\n" +
    "Example2) Query: \"what is the salary of a person with a biology degree\"\n"
    "Divide into sub-questions:\n"
    "What are the common career paths for someone with a biology degree?\n"
    "What is the entry-level salary for these careers?\n"
    "How does the salary progress with experience in biology-related careers?\n"
    "Prioritize the sub-questions in a logical sequence:\n"
    "What are the common career paths for someone with a biology degree?\n"
    "What is the entry-level salary for these careers?\n"
    "How does the salary progress with experience in biology-related careers?\n"
)
INPUT_TEMPLATE_METHOD_2 = (
    "Query to rewrite: {query}\n"
    "Divide into sub-questions:\n"
    "Prioritize the sub-questions in a logical sequence:"
)

REWRITE_PROMPT_METHOD_3 = (
    REWRITER_GENERAL_PROMPT +
    "Main Condition: Repeat the following questions while maintaining the same content.\n"
    "Sub Conditions:\n"
    "It is essential to preserve the main content and intention of the question."
    "You may transform the structure of the original question (e.g., from a direct to an indirect question).\n"
    "Feel free to use synonyms or similar words for the keywords in the question, but avoid introducing new concepts or sentences.\n"
    "You may emphasize different aspects or words in the question than originally highlighted. For instance, if the original question focused on a specific word, now you can emphasize it in a different context or with different words.\n"
    "\n" +
    "Example1) Query: \"how long can you leave a cooked meal in the fridge\"\n"
    "Rewritten Query: What is the maximum duration a cooked meal can be stored in the fridge?\n"
    "For how many days can you safely store a cooked meal in the refrigerator?\n"
    "How many days can a cooked meal remain in the fridge before it spoils?\n"
    "\n" +
    "Example2) Query: \"what is palladium metal\"\n"
    "Rewritten Query: What exactly is palladium metal?\n"
    "Can you explain what palladium metal is?\n"
    "What is the definition of palladium metal?\n"
    "How would you describe palladium metal?\n"
)
INPUT_TEMPLATE_METHOD_3 = (
    "Query to rewrite: {query}\n"
    "Rewritten Query:"
)

REWRITE_PROMPT_METHOD_4 = (
    REWRITER_MULTIFACETED_PROMPT +
    "(a) Conditional: The original question needs to be refined by specifying additional conditions that may be specifications or constraints.\n"
    "Query: When did movies start being made in color?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: When was the first film made that utilized any type of color? A: September 1, 1902\n"
    "When did the first feature length film come out that was made entirely in three-strip Technicolor? A: June 13, 1935\n"
    "\n" +
    "(b) Set-Valued: The answer to the question is an unstructured collection of size two or greater.\n"
    "Query: What are the neighboring countries of South Korea? \n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: What are the neighboring countries to the North of South Korea? A: North Korea\n"
    "What are the neighboring countries to the South of South Korea? A: Japan\n"
    "\n" +
    "(c) Time Dependent: The answer depends on the time at which the question was asked, or changed over time in the past.\n"
    "Query: Where was indian independence league formed in 1942?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: Where was indian independence league brought together in March 1942? A: Tokyo\n"
    "Where was indian independence league brought together in June 1942? A: Bangkok Conference\n"
    "\n" +
    "(d) Underspecified Reference: There is a noun phrase in the question which may be resolved in multiple ways.\n"
    "Query: When did bat out of hell come out?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: When did the album bat out of hell come out? A: October 21, 1977\n"
    "When did the TV series bat out of hell come out? A: 26 November 1966\n"
    "\n" +
    "(e) Underspecified Type: The entity type or sub-type is not specified in the question.\n"
    "Query: Who is the mayor in horton hears a who?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: Who plays the mayor in the 2008 film Horton Hears a Who? A: Steve Carell\n"
    "Who is the mayor in the 2008 film Horton Hears a Who? A: Mayor Ned McDodd\n"
    "\n" +
    "(f) Needs Elaboration: The answer needs to be elaborated to fully answer the question\n"
    "Query: Where did \“you can’t have your cake and eat it too\” come from?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: Where was the early recording of the phrase found? A: in a letter on 14 March 1538\n"
    "Who sent the letter? A: Thomas, Duke of Norfolk"
    "To whom was the letter sent to? A: Thomas Cromwell\n"
    "How was it phrased in the letter: A: \“a man cannot have his cake and eat his cake\”\n"
    "\n" +
    "DO NOT answer your Rewritten Query. Only provide the Rewritten Query."
)
INPUT_TEMPLATE_METHOD_4 = (
    "Query to rewrite: {query}\n"
    "Type: \n"
    "Rewritten Query:"
)

REWRITE_PROMPT_METHOD_5 = (
    REWRITER_MULTIFACETED_PROMPT +
    "(a) Conditional: The original question needs to be refined by specifying additional conditions that may be specifications or constraints.\n"
    "Query: When did movies start being made in color?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: When was the first film made that utilized any type of color?\n"
    "When did the first feature length film come out that was made entirely in three-strip Technicolor?\n"
    "\n" +
    "(b) Set-Valued: The answer to the question is an unstructured collection of size two or greater.\n"
    "Query: What are the neighboring countries of South Korea? \n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: What are the neighboring countries to the North of South Korea?\n"
    "What are the neighboring countries to the South of South Korea?\n"
    "\n" +
    "(c) Time Dependent: The answer depends on the time at which the question was asked, or changed over time in the past.\n"
    "Query: Where was indian independence league formed in 1942?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: Where was indian independence league brought together in March 1942?\n"
    "Rewritten Query: Where was indian independence league brought together in June 1942?\n"
    "\n" +
    "(d) Underspecified Reference: There is a noun phrase in the question which may be resolved in multiple ways.\n"
    "Query: When did bat out of hell come out?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: When did the album bat out of hell come out?\n"
    "When did the TV series bat out of hell come out?\n"
    "\n" +
    "(e) Underspecified Type: The entity type or sub-type is not specified in the question.\n"
    "Query: Who is the mayor in horton hears a who?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: Who plays the mayor in the 2008 film Horton Hears a Who?\n"
    "Who is the mayor in the 2008 film Horton Hears a Who?\n"
    "\n" +
    "(f) Needs Elaboration: The answer needs to be elaborated to fully answer the question\n"
    "Query: Where did \“you can’t have your cake and eat it too\” come from?\n"
    "Multifaceted QA Pairs:\n"
    "Rewritten Query: Where was the early recording of the phrase found?\n"
    "Who sent the letter?\n"
    "To whom was the letter sent to?\n"
    "How was it phrased in the letter:\n"
)
INPUT_TEMPLATE_METHOD_5 = (
    "Query to rewrite: {query}\n"
    "Type: \n"
    "Rewritten Query:"
)