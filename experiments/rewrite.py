import argparse
import os

import dotenv
from openai import OpenAI
from prompt_templates import rewrite_method1 as prompt
from typing import List, Tuple, Dict, Optional


dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)


class Rewriter:
    def __init__(self, model_name, prompt, max_new_tokens=70):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewriter_model_name', type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI `model_name` for Rewriter")
    parser.add_argument('--rewriter_max_new_tokens', type=int, default=70, help="`max_new_tokens` for Rewriter.")
    parser.add_argument('--rewrite_method', type=str, default="method1", help="`rewrite_method` for Rewriter.")
    parser.add_argument('--query', type=str, default="what is nlp stand for", help="`query` to rewrite.")
    parser.add_argument('--debug', action="store_true", help="whether use debug mode")

    args = parser.parse_args()

    if args.rewrite_method == "method1":
        from prompt_templates import rewrite_method1 as prompt
    elif args.rewrite_method == "method2":
        from prompt_templates import rewrite_method2 as prompt
    elif args.rewrite_method == "method3":
        from prompt_templates import rewrite_method3 as prompt
    elif args.rewrite_method == "method4":
        from prompt_templates import rewrite_method4 as prompt
    elif args.rewrite_method == "method5":
        from prompt_templates import rewrite_method5 as prompt

    rewriter = Rewriter(
        model_name=args.rewriter_model_name,
        prompt=prompt,
        max_new_tokens=args.rewriter_max_new_tokens,
    )

    rewritten_query = rewriter.rewrite_query(args.query)

    if args.debug:
        print(f"Original Query: {args.query}")
        print(f"Rewritten Query ({args.rewrite_method}): {rewritten_query}")
        print()


if __name__=="__main__":
    main()

    """
    Original Query: when was the first robot used in surgery
    """

    """
    Rewritten Query (method1): What year was the first robot utilized in surgical procedures?
    """
    """
    Rewritten Query (method2): What was the date of the first robot used in surgery?  
    What type of surgery was the first robot used for?  
    Who developed the first surgical robot?  
    What impact did the introduction of robots have on surgical practices?
    """
    """
    Rewritten Query (method3): When did the first robot make its debut in surgical procedures?  
    What year marked the introduction of robots in surgery?  
    In which year was the first surgical robot utilized?  
    Can you tell me when robots were first employed in surgical operations?
    """
    """
    Rewritten Query (method4): Type: Time Dependent

    Rewritten query: When was the first robot used in surgery on human patients?
    """
    """
    Rewritten Query (method5): Type: Time Dependent

    Rewritten Query: When was the first robot used in surgery in the United States?  
    Rewritten Query: When was the first robot used in surgery in Europe?  
    Rewritten Query: When was the first robot used in surgery for a specific type of procedure (e.g., prostate surgery)?
    """