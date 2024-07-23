import argparse
import os

import dotenv
from openai import OpenAI
from prompt_templates import method1 as prompt
from typing import List, Tuple, Dict, Optional


dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)


parser = argparse.ArgumentParser()
parser.add_argument('--rewriter_model_name', type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI `model_name` for Rewriter")
parser.add_argument('--rewriter_max_new_tokens', type=int, default=70, help="`max_new_tokens` for Rewriter.")
parser.add_argument('--rewrite_method', type=str, default="method1", help="`rewrite_method` for Rewriter.")

args = parser.parse_args()


class Rewriter:
    def __init__(self, model_name, prompt, max_new_tokens=max_new_tokens):
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def rewrite_query(
        self, 
        query: str
    ) -> Optional[str]:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": self.prompt(query)}
                ],
                model=self.model_name,
                max_tokens=self.max_new_tokens
            )
            output = response.choices[0].message.content.strip()
            return output

        except Exception as e:
            print(e)
            return None


if __name__=="__main__":
    if args.rewrite_method == "method1":
        from prompt_templates import method1 as prompt
    elif args.rewrite_method == "method2":
        from prompt_templates import method2 as prompt
    elif args.rewrite_method == "method3":
        from prompt_templates import method3 as prompt
    elif args.rewrite_method == "method4":
        from prompt_templates import method4 as prompt
    elif args.rewrite_method == "method5":
        from prompt_templates import method5 as prompt

    rewriter = Rewriter(
        model_name=args.rewriter_model_name,
        prompt=prompt,
        max_new_tokens=args.rewriter_max_new_tokens,
    )

    rewritten_query = rewriter.rewrite_query("what is nlp stand for")
    print(rewritten_query)