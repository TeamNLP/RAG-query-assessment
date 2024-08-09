import argparse
import os

import dotenv
from lib import read_jsonl, write_jsonl
from model.classifier import Classifier
from tqdm import tqdm


dotenv.load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')


parser = argparse.ArgumentParser()
parser.add_argument('--classifier_model_name', type=str, default=None, help="`model_name` for Classifier")
parser.add_argument("--dataset", type=str, default=None, choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad', 'sewon/ambig_qa', 'ambig_qa', 'microsoft/ms_marco', 'ms_marco'), help="")
parser.add_argument("--dataset_type", type=str, default=None, choices=("train", "dev", "test_subsampled", "dev_500_subsampled"), help="")
parser.add_argument('--rewrite_method', type=str, default="method3", help="`rewrite_method` for Rewriter.")
parser.add_argument('--output_directory', type=str, default="rewritten_data/classifier_model", help="`output_directory` to store the rewritten data")

args = parser.parse_args()


def get_rewritten_query(rewriter, query):
    response = rewriter.rewrite_query(query)

    if args.rewrite_method == "method2":
        response = response.split("Prioritize the sub-questions in a logical sequence:")[-1]
    rewritten_query = response.split("Rewritten Query:")[-1].strip()

    return rewritten_query


def main(args):
    classifier = Classifier(
        model_name=args.classifier_model_name,
    )

    input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data", args.dataset)
    input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")
    input_instance = read_jsonl(input_filepath)

    rewritten_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rewritten_data", args.rewrite_method, args.dataset)
    rewritten_filepath = os.path.join(rewritten_directory, f"{args.dataset_type}.jsonl")
    rewritten_instance = read_jsonl(rewritten_filepath)

    output_file_name = args.dataset_type
    output_instances = []
    
    for idx, datum in enumerate(tqdm(rewritten_instance, desc=f"Run Classifier on {args.dataset_type}.jsonl of {args.dataset}")):
        question_id = datum["question_id"]
        question_text = datum["question_text"]

        label, score = classifier.classify(question_text)
        datum["rewrite"] = label

        if label:
            assert rewritten_instance[idx]["question_id"] == question_id
            rewritten_question_text = rewritten_instance[idx]["question_text"]
            datum["question_text"] = rewritten_question_text
        else:
            pass

        output_instances.append(datum)

    output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filepath = os.path.join(output_directory, f"{output_file_name}.jsonl")

    write_jsonl(output_instances, output_filepath)


if __name__=="__main__":
    main(args)