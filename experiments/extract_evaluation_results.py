import argparse
import os
import json
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--base_directory', type=str, default=".", help="base_directory where prediction results are located")
parser.add_argument('--output_directory_prefix', type=str, default="", help="prefix of output_directory where prediction results are located")

args = parser.parse_args()

def extract_metrics(directories):
    # Define the subdirectories to look for the JSON files
    subdirs = ["nq", "squad", "trivia", "2wikimultihopqa", "hotpotqa", "musique"]
    
    # Prepare the CSV file
    output_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.base_directory, f'{args.output_directory_prefix}evaluation_results.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        header = ['models']
        for subdir in subdirs:
            header.append(f'{subdir}_EM')
            header.append(f'{subdir}_F1')
        csvwriter.writerow(header)
        
        # Iterate through each directory
        for directory in directories:
            row = [directory]
            for subdir in subdirs:
                json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), directory, subdir, 'eval_metic_result_acc.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as json_file:
                        data = json.load(json_file)
                        em = data.get("em", None)
                        f1 = data.get("f1", None)
                        row.append(em)
                        row.append(f1)
                else:
                    row.append(None)
                    row.append(None)
            csvwriter.writerow(row)

    print(f"Saved evaluation result CSV file at {output_csv_path}")

# Example usage
directories = [
    f'{args.base_directory}/{args.output_directory_prefix}gpt-4o-mini-2024-07-18',
    f'{args.base_directory}/{args.output_directory_prefix}Meta-Llama-3-8B',
    f'{args.base_directory}/{args.output_directory_prefix}Meta-Llama-3-8B-Instruct',
    f'{args.base_directory}/{args.output_directory_prefix}Meta-Llama-3.1-8B',
    f'{args.base_directory}/{args.output_directory_prefix}Meta-Llama-3.1-8B-Instruct',
    f'{args.base_directory}/{args.output_directory_prefix}Qwen2-7B',
    f'{args.base_directory}/{args.output_directory_prefix}Qwen2-7B-Instruct',
]

extract_metrics(directories)
