import argparse

parser = argparse.ArgumentParser(description="Finetune a seq2seq based model")

parser.add_argument(
    "--use_fp16", 
    action="store_true", 
    help="Whether to use fp16 (mixed) precision instead of 32-bit."
)

args = parser.parse_args()

print(args.use_fp16)
print(not args.use_fp16)
