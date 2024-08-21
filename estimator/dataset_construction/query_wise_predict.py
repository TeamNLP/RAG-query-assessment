import argparse
import json
import os

import dotenv
from lib import make_request_dict, write_json, read_jsonl, write_jsonl, CORPUS_NAME_DICT
from model.LLM import GeneratorLLM
from model.RAG import BM25OkapiRetriever, make_rag_framework
from openai import OpenAI
from tqdm import tqdm


dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', type=str, default="source_dataset", help="`input_directory` to predict results")
parser.add_argument('--output_directory', type=str, default="predictions", help="`output_directory` to store the prediction results")
parser.add_argument('--retrieval_corpus_name', type=str, default=None, required=False, help="`corpus_name` for ElasticSearch Retriever")
parser.add_argument('--retriever_api_url', type=str, default=None, help="`api_url` for ElasticSearch Retriever")
parser.add_argument('--retrieval_top_n', type=int, default=5, help="A number for how many results to retrieve")
parser.add_argument('--generator_model_name', type=str, default=None, help="`model_name` for Generator. Please refer to https://docs.vllm.ai/en/latest/models/supported_models.html.")
parser.add_argument('--generator_max_new_tokens', type=int, default=100, help="`max_new_tokens` for generator.")
parser.add_argument("--dataset", type=str, default=None, choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad', "ms_marco"), help="")
parser.add_argument("--dataset_type", type=str, default="train_18000_subsampled", choices=("train", "train_18000_subsampled", "dev", "test_subsampled", "dev_500_subsampled", "dev_7_subsampled"), help="")
parser.add_argument('--do_sample', action="store_true", help="whether use sampling while generate responses")
parser.add_argument('--temperature', type=float, default=1.0, help="")
parser.add_argument('--top_k', type=int, default=50, help="")
parser.add_argument('--top_p', type=float, default=1.0, help="")
parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1, help="TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.")
parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9, help="TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.")
parser.add_argument('--vllm_dtype', type=str, default="auto", help="Data type for model weights and activations. Possible choices: auto, half, float16, bfloat16, float, float32")
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--openai_batch_api', type=str, default=None, choices=("upload", "check", "analysis"), help="OpenAI Batch API")
parser.add_argument('--openai_batch_id', type=str, default=None, help="ID of OpenAI Batch object")
parser.add_argument('--openai_file_id', type=str, default=None, help="ID of OpenAI Files API")
parser.add_argument('--get_retriever_result', action="store_true")
parser.add_argument('--use_retriever_result', action="store_true")
parser.add_argument('--use_chat_template', action="store_true", help="whether use chat template")
parser.add_argument('--use_template_wo_instruction', action="store_true", help="whether use prompt template without instruction")

args = parser.parse_args()


def retrieve(rag_framework, query):
    retrieval_results, retrieval_time = rag_framework.retriever.retrieve(query)
    passage_list = rag_framework.retriever.get_passages_as_list(retrieval_results)
    return retrieval_results, passage_list


def generate_LLM(generator_LLM, query):
    if generator_LLM.use_hf:
        return generator_LLM.generate_hf_response(query)
    else:
        return generator_LLM.generate_gpt_response(query)


def generate_RAG(rag_framework, query, passage_list):
    passages = "\n".join(passage_list)
    return rag_framework.generator.generate(query, passages)


def prepare_openai_batch_file(generator, input_filepath, retriever_output_directory=None):
    batch_instances = []
    tok_cnt = 0

    if retriever_output_directory is None:
        input_instance = read_jsonl(input_filepath)
        
        for datum in tqdm(input_instance, desc=f"Making an OpenAI Batch File on {input_filepath}"):
            question_id = datum["question_id"]
            question_text = datum["question_text"]
            
            batch_instance, tok_len = make_request_dict(question_id, question_text, generator, passages=None)
            batch_instances.append(batch_instance)
            tok_cnt += tok_len

    else:
        input_instance = read_jsonl(input_filepath)
        retriever_instance = read_jsonl(retriever_output_directory)
        assert len(input_instance) == len(retriever_instance)

        for idx in tqdm(range(len(input_instance)), desc=f"Making an OpenAI Batch File on {input_filepath}"):
            datum = input_instance[idx]
            question_id = datum["question_id"]
            question_text = datum["question_text"]
            passages = retriever_instance[idx]
            
            batch_instance, tok_len = make_request_dict(question_id, question_text, generator, passages)
            batch_instances.append(batch_instance)
            tok_cnt += tok_len

    print(f"Token Length: {tok_cnt} @ {input_filepath}")
    return batch_instances



def main(args):
    openai_client = None
    if "gpt-3.5" in args.generator_model_name.lower() or "gpt-4" in args.generator_model_name.lower():
        api_key = os.environ.get('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)

    if args.openai_batch_api == "check":
        if args.openai_batch_id is not None:
            batch_obj = openai_client.batches.retrieve(args.openai_batch_id)
            status = batch_obj.status
            description = batch_obj.metadata["description"]
            print(f"{args.generator_model_name} - {description} ({args.openai_batch_id}): {status}")

            if status == "completed":
                print(f"Result of {description} ({args.openai_batch_id}) will be analyzed now :)")
                args.openai_file_id = batch_obj.output_file_id
                args.dataset = description[10:]
                print(f"Output File ID of {description} ({args.openai_batch_id}): {args.openai_file_id}")
                args.openai_batch_api = "analysis"
            else:
                return
        else:
            raise Exception(f"args.openai_batch_id is None!")

    if args.retriever_api_url is None:
        try:
            args.retriever_api_url = os.environ.get('RETRIEVER_API_URL')
        except KeyError:
            raise KeyError("`retriever_api_url` required!")
        
    if args.retrieval_corpus_name is None:
        args.retrieval_corpus_name = CORPUS_NAME_DICT[args.dataset]  

    rag_framework = make_rag_framework(args)

    generator_LLM = GeneratorLLM(
        model_name=args.generator_model_name, 
        generator_config={
            "use_chat_template":args.use_chat_template,
            "use_template_wo_instruction":args.use_template_wo_instruction,
            "max_new_tokens":args.generator_max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_dtype": args.vllm_dtype,
            "batch_size": args.batch_size
        },
        openai_client=openai_client
    )

    input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.input_directory, args.dataset)
    input_filepath = os.path.join(input_directory, f"{args.dataset_type}.jsonl")

    retriever_output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "retriever", args.dataset)
    if not os.path.exists(retriever_output_directory):
        os.makedirs(retriever_output_directory)
    retriever_output_filepath = os.path.join(retriever_output_directory, f"prediction_{args.dataset_type}.jsonl")
    
    generator_output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "generator", args.dataset)
    if not os.path.exists(generator_output_directory):
        os.makedirs(generator_output_directory)
    generator_output_filepath = os.path.join(generator_output_directory, f"prediction_{args.dataset_type}.jsonl")

    rag_output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "rag", args.dataset)
    if not os.path.exists(rag_output_directory):
        os.makedirs(rag_output_directory)
    rag_output_filepath = os.path.join(rag_output_directory, f"prediction_{args.dataset_type}.jsonl")

    input_instance = read_jsonl(input_filepath)

    if args.openai_batch_api == "upload":
        # Preparing Batch Files
        batch_file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "BatchAPI", args.dataset)
        if not os.path.exists(batch_file_directory):
            os.makedirs(batch_file_directory)

        rag_batch_input_filepath = os.path.join(batch_file_directory, f"RAG_batch_input_{args.dataset_type}.jsonl")
        llm_batch_input_filepath = os.path.join(batch_file_directory, f"LLM_batch_input_{args.dataset_type}.jsonl")

        rag_batch_instances = prepare_openai_batch_file(rag_framework.generator, input_filepath, retriever_output_filepath)
        write_jsonl(rag_batch_instances, rag_batch_input_filepath)

        llm_batch_instances = prepare_openai_batch_file(generator_LLM, input_filepath, None)
        write_jsonl(llm_batch_instances, llm_batch_input_filepath)

        # Uploading Batch Input Files
        # rag_batch_input_file = openai_client.files.create(
        #     file=open(rag_batch_input_filepath, "rb"),
        #     purpose="batch"
        # )

        llm_batch_input_file = openai_client.files.create(
            file=open(llm_batch_input_filepath, "rb"),
            purpose="batch"
        )

        # Creating the Batch
        # rag_batch_input_file_id = rag_batch_input_file.id
        llm_batch_input_file_id = llm_batch_input_file.id

        # rag_batch_obj = openai_client.batches.create(
        #     input_file_id=rag_batch_input_file_id,
        #     endpoint="/v1/chat/completions",
        #     completion_window="24h",
        #     metadata={
        #         "description": f"rag_batch_{args.dataset}"
        #     }
        # )
        # print(f"rag_batch_obj: {rag_batch_obj}")

        llm_batch_obj = openai_client.batches.create(
            input_file_id=llm_batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"llm_batch_{args.dataset}"
            }
        )
        print(f"llm_batch_obj: {llm_batch_obj}")

        # rag_batch_id = rag_batch_obj.id
        llm_batch_id = llm_batch_obj.id

        # print(f"rag_batch_obj.id: {rag_batch_id}")
        print(f"llm_batch_obj.id: {llm_batch_id}")

        # Checking the Status of a Batch
        # print(f"Checking the Status of a RAG Batch: {openai_client.batches.retrieve(rag_batch_id)}")
        # print(f"Checking the Status of a LLM Batch: {openai_client.batches.retrieve(llm_batch_id)}")

        return
    elif args.openai_batch_api == "analysis":
        if args.openai_file_id is not None:
            # Preparing Batch File Results
            file_response = openai_client.files.content(args.openai_file_id)
            batch_str_list = file_response.text.split("\n")[:-1]
            
            batch_output_instances = []
            for batch_str in tqdm(batch_str_list):
                batch_output_instance = json.loads(batch_str)
                batch_output_instances.append(batch_output_instance)

            batch_file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_directory, "BatchAPI", args.dataset)
            if not os.path.exists(batch_file_directory):
                os.makedirs(batch_file_directory)

            if "LLM" in batch_output_instance["custom_id"][0:3]:
                batch_output_filepath = os.path.join(batch_file_directory, f"LLM_batch_output_{args.dataset_type}.jsonl")
                generation_output_filepath = generator_output_filepath
            elif "RAG" in batch_output_instance["custom_id"][0:3]:
                batch_output_filepath = os.path.join(batch_file_directory, f"RAG_batch_output_{args.dataset_type}.jsonl")
                generation_output_filepath = rag_output_filepath
            else:
                raise Exception
            write_jsonl(batch_output_instances, batch_output_filepath)
            
            generation_output_instance = []
            for idx, batch_output_instance in enumerate(batch_output_instances):
                datum = input_instance[idx]
                question_id = datum["question_id"]
                question_text = datum["question_text"]
                answers_objects = datum["answers_objects"]
                generation_result = batch_output_instance["response"]["body"]["choices"][0]["message"]["content"]
                
                generation_output_dict = {}
                generation_output_dict["question_id"] = question_id
                generation_output_dict["question_text"] = question_text
                generation_output_dict["answers_objects"] = answers_objects
                generation_output_dict["result"] = generation_result
                generation_output_instance.append(generation_output_dict)
            write_jsonl(generation_output_instance, generation_output_filepath)
            
            return
        else:
            raise Exception(f"args.openai_batch_id is None!")

    if args.use_retriever_result == False:
        generator_output_instance = []
        retriever_output_instance = []
        rag_output_instance = []

        for datum in tqdm(input_instance, desc=f"Generating predictions on {input_filepath}"):
            generator_output_dict = {}
            retriever_output_dict = {}
            rag_output_dict = {}

            question_id = datum["question_id"]
            question_text = datum["question_text"]
            answers_objects = datum["answers_objects"]

            generator_output_dict["question_id"] = question_id
            retriever_output_dict["question_id"] = question_id
            rag_output_dict["question_id"] = question_id

            generator_output_dict["question_text"] = question_text
            retriever_output_dict["question_text"] = question_text
            rag_output_dict["question_text"] = question_text

            generator_output_dict["answers_objects"] = answers_objects
            rag_output_dict["answers_objects"] = answers_objects

            generator_result = generate_LLM(generator_LLM, question_text)
            generator_output_dict["result"] = generator_result

            contexts = datum["contexts"] 
            total_document_list = [context_dict["paragraph_text"] for context_dict in contexts]
            total_support_list = [context_dict["is_supporting"] for context_dict in contexts]

            assert len(total_document_list) == len(total_support_list)
            doc_support_dict= {}
            for i in range(len(total_document_list)):
                doc_support_dict[total_document_list[i]] = total_support_list[i]

            if args.dataset == "ms_marco":
                assert args.retrieval_top_n == 3
                retriever = BM25OkapiRetriever(total_document_list, args.retrieval_top_n)
                passage_list = retriever.retrieve_documents(question_text)
            else:
                retrieval_results, passage_list = retrieve(rag_framework, question_text)

            is_support_list = []
            for passage in passage_list:
                if passage.strip() in total_document_list:
                    is_support_list.append(doc_support_dict[passage.strip()])
                else:
                    is_support_list.append(False)

            retriever_output_dict["result"] = {"passage_list": passage_list, "is_support_list": is_support_list, "total_supported_doc": sum(total_support_list)}

            rag_result = generate_RAG(rag_framework, question_text, passage_list)
            rag_output_dict["result"] = rag_result

            generator_output_instance.append(generator_output_dict)
            retriever_output_instance.append(retriever_output_dict)
            rag_output_instance.append(rag_output_dict)

        write_jsonl(generator_output_instance, generator_output_filepath)
        write_jsonl(retriever_output_instance, retriever_output_filepath)
        write_jsonl(rag_output_instance, rag_output_filepath)

    else:
        generator_output_instance = []
        rag_output_instance = []
        
        retriever_output_instance = read_jsonl(retriever_output_filepath)

        for idx in tqdm(range(len(input_instance)), desc=f"Generating predictions on {input_filepath}"):
            datum = input_instance[idx]
            
            # generator_output_dict = {}
            rag_output_dict = {}

            question_id = datum["question_id"]
            question_text = datum["question_text"]
            answers_objects = datum["answers_objects"]

            # generator_output_dict["question_id"] = question_id
            rag_output_dict["question_id"] = question_id

            # generator_output_dict["question_text"] = question_text
            rag_output_dict["question_text"] = question_text

            # generator_output_dict["answers_objects"] = answers_objects
            rag_output_dict["answers_objects"] = answers_objects

            # generator_result = generate_LLM(generator_LLM, question_text)
            generator_output_dict["result"] = generator_result

            passage_list = [context_dict["paragraph_text"] for context_dict in retriever_output_instance[idx]["contexts"]]

            rag_result = generate_RAG(rag_framework, question_text, passage_list)
            rag_output_dict["result"] = rag_result

            # generator_output_instance.append(generator_output_dict)
            rag_output_instance.append(rag_output_dict)

        # write_jsonl(generator_output_instance, generator_output_filepath)
        write_jsonl(rag_output_instance, rag_output_filepath)


if __name__ == "__main__":
    main(args)