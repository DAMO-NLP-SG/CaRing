import os, time
import argparse
import json, jsonlines
from tqdm import tqdm
import logging

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import transformers
import torch
from vllm import LLM, SamplingParams

from data import load_dataset
from build_prompt import build_demo, build_prompt


# Configure the logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s:%(lineno)s - %(levelname)s - %(message)s',
    # filename='app.log',  # Uncomment this if you want to log to a file
    # filemode='w',  # Overwrites the log file every time
) 

logger = logging.getLogger(__name__)


def main(args):

    # Load Data
    dataset_name = args.input_path.split("/")[-2]
    input_data = load_dataset(args.input_path, dataset_name=dataset_name)

    if args.debug:
        logger.info("Debug mode. Only process the first two examples. ")
        input_data = input_data[:2]
    else:
        if args.max_samples > 0:
            logger.info(f'"--max_samples" is set. Only process the first {args.max_samples} examples. ')
            input_data = input_data[:args.max_samples]
        else:
            logger.info(f"Process all {len(input_data)} examples. ")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop="Problem No.{}".format(len(args.demo_indices)+2),
        # n=5,
        # use_beam_search=True,
        )
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.num_gpus, max_num_batched_tokens=args.max_tokens_total)
    
    # 
    output_path = args.output_path
    
    output_path = output_path.rstrip(".json")
    output_path = output_path + ".demo-{}.json".format("_".join([str(_) for _ in args.demo_indices]))

    all_responses = []
    if os.path.exists(output_path) and not args.debug and not args.overwrite_output:
        with open(output_path, "r") as f:
            all_responses = [json.loads(line) for line in f.readlines() if line.strip()]
        logger.info(f"Continue from {len(all_responses)}-th example.")
        logger.info(f"Number of examples to be processed: {len(input_data)-len(all_responses)}")
    else:
        logger.info(f"Start from the beginning.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    prompt_demo = build_demo(
        demo_indices=args.demo_indices,
        demonstration_path=args.demonstration_path,
        dataset_name=dataset_name,
        mode=args.mode,
    )

    if args.mode == "cot":
        system_message = "You are a helpful, pattern-following assistant that helps people solve problems. "
    elif args.mode == "prolog":
        system_message = "You are a helpful, pattern-following assistant that helps people solve problems using Prolog. "
    
    prompt_list = []
    for example in tqdm(input_data[len(all_responses):], total=len(input_data[len(all_responses):])):
        prompt = build_prompt(
            example=example,
            prompt_demo=prompt_demo,
            dataset_name=dataset_name,
            mode=args.mode,
            instruct=args.instruct,
        )
        # import pdb; pdb.set_trace()
        # prompt = f"<s>[INST] <<SYS>>\\n{system_message}\\n<</SYS>>\\n\\n{prompt}[/INST]"
        prompt_list.append(prompt)

    if args.debug:
        # for i, _ in enumerate(all_responses):
        #     logger.info(f"****** Input-{i+1} ****** \n")
        #     print(json.dumps(input_data[i]))
        #     logger.info(f"***** Response-{i+1} ***** \n")
        #     print(json.dumps(_))
        pass
    else:

        outputs = llm.generate(prompt_list, sampling_params)
        # import pdb; pdb.set_trace()

        # Print the outputs.
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        all_responses = [
            {
                "prompt": output.prompt,
                "response": [_.text for _ in output.outputs]
            }
            for output in outputs
        ]

        with open(output_path, "w") as f:
            f.writelines([json.dumps(_)+"\n" for _ in all_responses])
        logger.info("*"*20)
        logger.info(f"Finished. Output saved to {output_path}. ")
        logger.info("*"*20)
        
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, required=True)
    argparser.add_argument('--demonstration_path', type=str, required=True)
    argparser.add_argument('--output_path', type=str, required=True)
    argparser.add_argument('--model_name_or_path', type=str, required=True)
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--demo_indices', nargs='+', type=int, required=True)
    argparser.add_argument('--max_tokens_total', type=int, default=16384, help="Max number of total tokens (prompt + generated). Default is Code-LLaMA max-len = 16384 ")
    argparser.add_argument('--max_tokens', type=int, default=2048, help="Max number of generated tokens. Default is 2048. ")
    argparser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use. ")
    argparser.add_argument('--self_debug', action='store_true', help="Adopt self-debugging mode, which execute the GPT's response and prompt GPT with error messages if the execution fails. ")
    argparser.add_argument('--self_debug_limit', type=int, default=3, help="The maximum number of self-debugging trials. Default is 3. ")
    argparser.add_argument('--max_samples', type=int, default=-1, help="The maximum number of samples to be processed. Default is -1. ")
    argparser.add_argument('--mode', type=str, required=True, choices=['cot', 'prolog', 'direct'])
    argparser.add_argument('--overwrite_output', action='store_true', help="Whether to overwrite the output file, or to read the output file and continue from the break point. ")
    argparser.add_argument('--instruct', action='store_true', help="Whether to use instruction-style prompt. If False, use ICL-style prompt. ")
    args = argparser.parse_args()

    if args.debug:
        logger.info("Debug mode. ")
    main(args)