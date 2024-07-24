import os
import json
import csv

def extract_metrics(directories):
    # Define the subdirectories to look for the JSON files
    subdirs = ["nq", "squad", "trivia", "2wikimultihopqa", "hotpotqa", "musique"]
    
    # Prepare the CSV file
    output_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'evaluation_results.csv')
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
    'predictions_Meta-Llama-3-8B',
    'predictions_Meta-Llama-3-8B-Instruct',
    'predictions_Meta-Llama-3.1-8B',
    'predictions_Meta-Llama-3.1-8B-Instruct',
]

extract_metrics(directories)
