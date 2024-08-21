import os

from lib import read_jsonl, write_jsonl


def main():

    set_names = ["train", "dev"]

    input_directory = os.path.join("raw_data", "ms_marco")
    output_directory = os.path.join("processed_data", "ms_marco")
    os.makedirs(output_directory, exist_ok=True)

    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        input_filepath = os.path.join(input_directory, f"{set_name}.jsonl")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        raw_instances = read_jsonl(input_filepath)

        for raw_instance in raw_instances:

            question_id = raw_instance["query_id"]
            question_text = raw_instance["query"]
            raw_contexts = raw_instance["passages"]["passage_text"]
            supporting_labels = raw_instance["passages"]["is_selected"]
            question_type = raw_instance["query_type"]

            processed_contexts = []
            for index in range(len(raw_contexts)):
                paragraph_text = raw_contexts[index].strip()
                is_supporting = bool(int(supporting_labels[index]))
                processed_contexts.append(
                    {
                        "idx": index,
                        "paragraph_text": paragraph_text,
                        "is_supporting": is_supporting,
                    }
                )

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": raw_instance["answers"],
            }
            answers_objects = [answers_object]

            processed_instance = {
                "question_id": question_id,
                "question_text": question_text,
                "answers_objects": answers_objects,
                "contexts": processed_contexts,
                "question_type": question_type,
            }

            processed_instances.append(processed_instance)

        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
