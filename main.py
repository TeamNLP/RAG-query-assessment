import os

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from openai import OpenAI
import dotenv
dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
nltk.download('punkt')

class Retriever:
    def __init__(self, documents, method='bm25', top_n=5):
        self.documents = documents
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
        self.method = method        
        if method == 'bm25':
            self.index = BM25Okapi(self.tokenized_corpus)
        self.top_n = top_n
        
    def retrieve_documents(self, query):
        tokenized = word_tokenize(query.lower())
        doc_scores = self.index.get_scores(tokenized)
        top_n_indices = doc_scores.argsort()[-self.top_n:][::-1]
        return [self.documents[i] for i in top_n_indices]

class Generator:
    def __init__(self, model='gpt-3.5-turbo', prompt_template=None):
        self.model = model
        if prompt_template is None:
            self.prompt_template = "# Question: {query}\n # Context: {context}\n # Answer:"
        else:
            self.prompt_template = prompt_template

    def make_rag_prompt(self, query, context):
        return self.prompt_template.format(query=query, context=context)

    def generate_response(self, query, context=None):
        if self.model in ['gpt-3.5-turbo', 'gpt-4-turbo']:
            prompt = self.make_rag_prompt(query, context)
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                        ],
                    temperature=0,
                )
                output = response.choices[0].message.content.strip()
                return output

            except Exception as e:
                print(e)
                return None
            

class RAGFramework:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query, debug=False):
        
        retrieved_context = self.retriever.retrieve_documents(query)
        if debug: print("검색: ", retrieved_context)
        
        prompt = self.generator.make_rag_prompt(query, retrieved_context)
        if debug: print("프롬프트: ", prompt)
        
        # 생성
        response = self.generator.generate_response(
            query=query, 
            context=retrieved_context)
        
        if debug: print("생성: ", response)
        
        return response

def make_rag(documents, method='bm25', model='gpt-3.5-turbo', top_n=5):
    retriever = Retriever(documents, method=method, top_n=top_n)
    generator = Generator(model=model)
    
    # RAG framework 
    rag = RAGFramework(retriever, generator)
    return rag

def main():
    
    # documents = # list 타입
    documents = ["text1", "ai is apple intelligence", "searched", "notbook", "deep learning"]
    rag = make_rag(documents, method='bm25', model='gpt-3.5-turbo', top_n=5)
    
    response = rag.run("what is the ai?", debug=True)
    print(response)

if __name__ == "__main__":
    main()