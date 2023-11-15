#

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_response_path", type=str, required=True, help="")
    parser.add_argument('--gold_path', type=str, default=None, help="")
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--mode', type=str, default=None, choices=[None, 'prolog', 'cot'])
    parser.add_argument('--eval_num', type=int, default=-1, help="Number of examples to evaluate. ")
    # Below: boolean arguments
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode. ")
    parser.add_argument('--use_multiprocessing', action='store_true', help="Whether to use multiprocessing. ")
    parser.add_argument('--evaluate_proof', action='store_true', help="Whether to evaluate the correctness of the generated proofs. ")

    args = parser.parse_args()

    if args.dataset_name == 'gsm8k':
        from tasks.gsm8k.evaluate import main
        if args.gold_path is None:
            args.gold_path = '../data/gsm8k/reasoning_annotated_test.jsonl'
    elif args.dataset_name == 'proofwriter':
        from tasks.proofwriter.evaluate import main
        if args.gold_path is None:
            args.gold_path = '../data/proofwriter/meta-test.shuffled.json'
    elif args.dataset_name == 'prontoqa':
        from tasks.prontoqa.evaluate import main
        if args.gold_path is None:
            args.gold_path = '../data/prontoqa/dev.json'
    else:
        raise ValueError(f"--dataset_name {args.dataset_name} is not properly set in src/evaluate.py")
    
    if args.mode is None:
        gpt_response_file_name = os.path.split(args.gpt_response_path)[-1]
        if gpt_response_file_name.startswith('cot') or gpt_response_file_name.startswith('direct'):
            args.mode = 'cot'
        elif gpt_response_file_name.startswith('prolog'):
            args.mode = 'prolog'
        else:
            raise ValueError(f'LLM response file "{gpt_response_file_name}" does not have a correct prefix.')
    if args.mode == 'direct':
        assert not args.evaluate_proof, "Cannot evaluate proofs with direct prompting. "

    main(args)