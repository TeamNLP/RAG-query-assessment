import argparse
import os

import dotenv
from lib import read_jsonl, write_jsonl
from openai import OpenAI
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)


parser = argparse.ArgumentParser()
parser.add_argument('--rewriter_model_name', type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI `model_name` for Rewriter")
parser.add_argument('--rewriter_max_new_tokens', type=int, default=200, help="`max_new_tokens` for Rewriter.")
parser.add_argument('--rewrite_method', type=str, default="method1", help="`rewrite_method` for Rewriter.")
parser.add_argument('--query', type=str, default=None, help="`query` string to rewrite.")
parser.add_argument("--dataset", type=str, default=None, choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad'), help="")
parser.add_argument("--dataset_type", type=str, default=None, choices=("train", "dev", "test_subsampled", "dev_500_subsampled"), help="")
parser.add_argument('--output_directory', type=str, default="rewritten_data", help="`output_directory` to store the rewritten data")
parser.add_argument('--do_test', action="store_true", help="whether use test mode")
parser.add_argument('--debug', action="store_true", help="whether use debug mode")

args = parser.parse_args()


if args.rewrite_method == "method1":
    from prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_1, INPUT_TEMPLATE_METHOD_1
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_1
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_1
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method2":
    from prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_2, INPUT_TEMPLATE_METHOD_2
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_2
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_2
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method3":
    from prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_3, INPUT_TEMPLATE_METHOD_3
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_3
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_3
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method4":
    from prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_4, INPUT_TEMPLATE_METHOD_4
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_4
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_4
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method5":
    from prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_5, INPUT_TEMPLATE_METHOD_5
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_5
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_5
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT

class Rewriter:
    def __init__(self, model_name, rewrite_prompt, input_prompt, max_new_tokens=70):
        self.model_name = model_name
        self.rewrite_prompt = rewrite_prompt
        self.input_prompt = input_prompt
        self.max_new_tokens = max_new_tokens

    def make_rewrite_prompt(
        self,
        query: str
    ) -> Tuple[str, str]:
        rewrite_prompt = self.rewrite_prompt
        input_prompt = self.input_prompt.format(query=query)
        return rewrite_prompt, input_prompt

    def rewrite_query(
        self, 
        query: str
    ) -> Optional[str]:
        rewrite_prompt, input_prompt = self.make_rewrite_prompt(query)
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": rewrite_prompt},
                    {"role": "user", "content": input_prompt}
                ],
                model=self.model_name,
                max_tokens=self.max_new_tokens
            )

            output = response.choices[0].message.content.strip()
            return output

        except Exception as e:
            print(e)
            return None


def get_rewritten_query(rewriter, query):
    response = rewriter.rewrite_query(query)

    if args.rewrite_method == "method2":
        response = response.split("Prioritize the sub-questions in a logical sequence:")[-1]
    rewritten_query = response.split("Rewritten Query:")[-1].strip()

    return rewritten_query


def test(args):
    rewriter = Rewriter(
        model_name=args.rewriter_model_name,
        prompt_template=prompt_template,
        max_new_tokens=args.rewriter_max_new_tokens,
    )

    rewritten_query = get_rewritten_query(rewriter, args.query)

    if args.debug:
        print(f"Original Query: {args.query}")
        print(f"Rewritten Query ({args.rewrite_method}): {rewritten_query}")
        print()


def main(args):

    rewriter = Rewriter(
        model_name=args.rewriter_model_name,
        rewrite_prompt=REWRITE_PROMPT,
        input_prompt=INPUT_PROMPT,
        max_new_tokens=args.rewriter_max_new_tokens,
    )

    input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data", args.dataset)
    input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")
    input_instance = read_jsonl(input_filepath)

    output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filepath = os.path.join(output_directory, f"{args.dataset_type}.jsonl")
    output_instances = []
    
    for datum in tqdm(input_instance, desc=f"Run Rewriter on {args.dataset_type}.jsonl of {args.dataset}"):
        question_id = datum["question_id"]
        question_text = datum["question_text"]

        rewritten_question_text = get_rewritten_query(rewriter, question_text)
        datum["question_text"] = rewritten_question_text

        output_instances.append(datum)

    write_jsonl(output_instances, output_filepath)


if __name__=="__main__":
    if args.do_test:
        test(args)
    else:
        main(args)