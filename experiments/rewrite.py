import argparse
import os

import dotenv
from datasets import load_dataset, Dataset
from lib import read_jsonl, write_jsonl, hf_stratified_sampling, stratified_sampling, stratified_sampling_for_ambig_qa
from model.rewriter import Rewriter
from openai import OpenAI
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=13370)
parser.add_argument('--rewriter_model_name', type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI `model_name` for Rewriter")
parser.add_argument('--rewriter_max_new_tokens', type=int, default=200, help="`max_new_tokens` for Rewriter.")
parser.add_argument('--rewrite_method', type=str, default="method1", help="`rewrite_method` for Rewriter.")
parser.add_argument('--query', type=str, default=None, help="`query` string to rewrite.")
parser.add_argument("--dataset", type=str, default=None, choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad', 'sewon/ambig_qa', 'ambig_qa', 'microsoft/ms_marco', 'ms_marco'), help="")
parser.add_argument("--dataset_type", type=str, default=None, choices=("train", "dev", "test_subsampled", "dev_500_subsampled"), help="")
parser.add_argument('--output_directory', type=str, default="rewritten_data", help="`output_directory` to store the rewritten data")
parser.add_argument('--do_test', action="store_true", help="whether use test mode")
parser.add_argument('--debug', action="store_true", help="whether use debug mode")

args = parser.parse_args()


if args.rewrite_method == "method1":
    from model.prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_1, INPUT_TEMPLATE_METHOD_1
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_1
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_1
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method2":
    from model.prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_2, INPUT_TEMPLATE_METHOD_2
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_2
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_2
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method3":
    from model.prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_3, INPUT_TEMPLATE_METHOD_3
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_3
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_3
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method4":
    from model.prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_4, INPUT_TEMPLATE_METHOD_4
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_4
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_4
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT
elif args.rewrite_method == "method5":
    from model.prompt_templates import REWRITER_GENERAL_SYSTEM_PROMPT, REWRITE_PROMPT_METHOD_5, INPUT_TEMPLATE_METHOD_5
    REWRITE_PROMPT = REWRITE_PROMPT_METHOD_5
    INPUT_PROMPT = INPUT_TEMPLATE_METHOD_5
    SYSTEM_PROMPT = REWRITER_GENERAL_SYSTEM_PROMPT


def get_rewritten_query(rewriter, query):
    response = rewriter.rewrite_query(query)

    if args.rewrite_method == "method2":
        response = response.split("Prioritize the sub-questions in a logical sequence:")[-1]
    rewritten_query = response.split("Rewritten Query:")[-1].strip()

    return rewritten_query


def test(args):
    rewriter = Rewriter(
        model_name=args.rewriter_model_name,
        rewrite_prompt=REWRITE_PROMPT,
        input_prompt=INPUT_PROMPT,
        max_new_tokens=args.rewriter_max_new_tokens,
    )

    rewritten_query = get_rewritten_query(rewriter, args.query)

    if args.debug:
        print(f"Original Query: {args.query}")
        print(f"Rewritten Query ({args.rewrite_method}): {rewritten_query}")
        print()


def main(args):
    openai_client = OpenAI(api_key=api_key)

    rewriter = Rewriter(
        model_name=args.rewriter_model_name,
        rewrite_prompt=REWRITE_PROMPT,
        input_prompt=INPUT_PROMPT,
        max_new_tokens=args.rewriter_max_new_tokens,
        openai_client=openai_client
    )

    if args.dataset in ['sewon/ambig_qa', 'ambig_qa', 'microsoft/ms_marco', 'ms_marco']:
        sample_size = 500

        if args.dataset in ['microsoft/ms_marco', 'ms_marco']:
            dataset = load_dataset(args.dataset, "v2.1", split='train')
            question_id_column = "query_id"
            question_text_column = "query"
            answer_list_column = "answers"
            criterion = "query_type"
            
            sampled_dataset = hf_stratified_sampling(dataset, criterion, num_sample=sample_size, seed=args.seed, verbose=True)
            sampled_list = [dict(record) for record in sampled_dataset]
            assert len(sampled_list) == sample_size

        elif args.dataset in ['sewon/ambig_qa', 'ambig_qa']:
            dataset = load_dataset(args.dataset, "full", split='train')
            question_id_column = "id"
            question_text_column = "question"
            answer_list_column = "nq_answer"
            criterion = "query_type"

            list_of_data = [dict(record) for record in dataset]
            sampled_list = stratified_sampling_for_ambig_qa(list_of_data, num_sample=sample_size, seed=args.seed, verbose=True)
            assert len(sampled_list) == sample_size

        output_file_name = "rewritten_train_subsampled"
        output_instances = []
        for datum in tqdm(sampled_list, desc=f"Run Rewriter on {output_file_name}.jsonl of {args.dataset}"):
            question_text = datum[question_text_column]
            rewritten_question_text = get_rewritten_query(rewriter, question_text)

            output_instance = {}
            output_instance["question_id"] = datum[question_id_column]
            output_instance["question_text"] = rewritten_question_text
            output_instance["original_question"] = datum[question_text_column]
            output_instance["answers_objects"] = [{"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": datum[answer_list_column]}]
            output_instances.append(output_instance)

    else:
        input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data", args.dataset)

        input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")
        input_instance = read_jsonl(input_filepath)

        output_file_name = args.dataset_type
        output_instances = []
        for datum in tqdm(input_instance, desc=f"Run Rewriter on {args.dataset_type}.jsonl of {args.dataset}"):
            question_id = datum["question_id"]
            question_text = datum["question_text"]

            rewritten_question_text = get_rewritten_query(rewriter, question_text)
            datum["question_text"] = rewritten_question_text

            output_instances.append(datum)

    output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filepath = os.path.join(output_directory, f"{output_file_name}.jsonl")

    write_jsonl(output_instances, output_filepath)


if __name__=="__main__":
    if args.do_test:
        test(args)
    else:
        main(args)