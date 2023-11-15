
from typing import List, Dict, Any, Tuple, Union
import json
from tqdm import tqdm
import argparse
import os
import multiprocessing
from collections import defaultdict
import jsonlines
import networkx as nx
import random
import re
from interruptingcow import timeout
import tempfile


from tasks.proofwriter.call_swipl import consult_prolog
import tasks.proofwriter.parse_proof_gold as parse_gold
import tasks.proofwriter.parse_proof_pred as parse_pred

random.seed(42)

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s:%(lineno)s - %(levelname)s - %(message)s',
    # filename='app.log',  # Uncomment this if you want to log to a file
    # filemode='w',  # Overwrites the log file every time
) 

logger = logging.getLogger(__name__)

"""
prolog_meta_interpreters = {
        "raw": "{}",
        "raw_all_ans": prolog_output_all_answers.format("{}"),
        "with_proof": prolog_output_all_answers.format("mi_tree(g({}), Proof)"),  # With proof generation. One argument: Goal
        "iter_deep_with_proof": prolog_output_all_answers.format("mi_id_limit(g({}), Proof, {})"),  # Iterative deepening search, with proof generation. Two arguments: Goal, MaxDepth
        "iter_deep_no_proof": prolog_output_all_answers.format("mi_id_limit_no_proof(g({}), {})"),  # Iterative deepening search. Two arguments: Goal, MaxDepth
    }
"""

def divide_list(lst, n, randomize=False):
    """Divide a list into $n$ chunks."""

    if randomize:
        random.shuffle(lst)
    # calculate the size of each chunk
    chunk_size = len(lst) // n
    # calculate the remainder
    remainder = len(lst) % n
    chunks = []
    i = 0
    # create the chunks
    for _ in range(n):
        # if remainder > 0, add 1 to the chunk size
        if remainder > 0:
            chunks.append(lst[i:i+chunk_size+1])
            i += chunk_size + 1
            remainder -= 1
        else:
            chunks.append(lst[i:i+chunk_size])
            i += chunk_size
    return chunks

# Pre-process Code & Labels
def extract_gold(gold_data):
    """Extract gold labels and proofs from gold data."""
    gold_labels, num_hops, proofs = [], [], []
    for i, example in enumerate(gold_data):
        gold_labels.extend([str(_['answer']).strip() for _ in example['questions'].values()])
        num_hops.extend([_['QDep'] for _ in example['questions'].values()])
        # proofs.extend([_['proofsWithIntermediates'][0]['representation'] for _ in example['questions'].values() if str(_['answer']).strip() != 'Unknown'])
        for _ in example['questions'].values():
            if str(_['answer']).strip() != 'Unknown':
                proofs.append(_['proofsWithIntermediates'][0]['representation'])
            else:
                proofs.append("None")
        # for j, _ in enumerate(example['questions'].values()):
            # assert "proofsWithIntermediates" in _, f"'proofsWithIntermediates' not in {j+1}-th question: {_}"
    return [
        {
            "gold_label": gold_labels[i],
            "num_hops": num_hops[i],
            "proof": proofs[i],
        }
        for i in range(len(gold_labels))
    ]

def negate_query(query_string: str) -> str:
    """
    negate a query string
    For example:
        Example #1. 
            input: 
                cold(charlie)
            output:
                not(cold(charlie))
        Example #2.
            input:
                not(cold(charlie))
            output:
                cold(charlie)
    """
    if query_string.strip() == "":
        return query_string
    query_string = query_string.strip()
    if not query_string.endswith("."):
        query_string = query_string + '.'
    if query_string.strip().startswith("not("):
        query_string = query_string.strip()[4:].replace(").", ".").strip()
    else:
        query_string = "not(" + query_string.strip().replace(".", "").strip() + ")."
    # query_string = "\+ " + query_string 
    return query_string[:-1]

def remove_prolog_comments(prolog_code):
    prolog_code = re.sub(r"%.*", "", prolog_code)
    prolog_code = re.sub(r'/\*.*?\*/', "", prolog_code, flags=re.DOTALL)
    return prolog_code

def preprocess_response_prolog(raw_response):
    if raw_response.strip().lower() in ('error', 'error.'):
        return ""
    
    # Extract all lines between "```prolog" and "```"
    if "```" in raw_response:
        if "```prolog" in raw_response:
            pattern = r'```prolog\n(.*?)\n```'
            num_code_blocks = raw_response.count("```prolog")
        elif "```" in raw_response:
            pattern = r'```\n(.*?)\n```'
            num_code_blocks = int(raw_response.count("```") / 2)
        matches = re.findall(pattern, raw_response, re.DOTALL)
        # merged_matches = ['\n'.join(match.splitlines()) for match in matches]
        # response = '\n\n'.join(merged_matches)
        response = matches[0]
    else:
        logger.info("Response containing no '```':")
        logger.info(raw_response)
        return None, None

    # transfrom "/* Questions */" / "/* Queries */" / "/* Query */" to "/* Question */"
    response = re.sub(r"/\* (Questions|Queries|Query) \*/", "/* Question */", response)

    # split using "/* Question */"
    response = response.split("/* Question */")
    assert len(response) == 2, f"Response should be split into two parts: knowledge_base_code and query_code. \nResponse: \n{response}"
    kb_code, query_code = response

    # Process Knowledge Base Code
    kb_code = remove_prolog_comments(kb_code)
    kb_code = '\n'.join([_ for _ in kb_code.splitlines() if _.strip() != ''])

    # Process Query Code
    query_code = '\n'.join([_ for _ in query_code.splitlines() if _.strip() != ''])
    query_code = remove_prolog_comments(query_code).strip()
    if "?-" in query_code:
        query_code = query_code.split("?-")[1].strip()
    if query_code.endswith("."):
        query_code = query_code[:-1].strip()
    return kb_code, query_code

def preprocess_response_cot(raw_response: str):
    """
    Example Input:
        Here is the reasoning chain:\ntriple-1 & rule-1 -> int-1: Charlie is white.; int-1 & rule-2 -> int-2: Charlie is young.; int-2 & rule-4 -> int-3: Charlie is not cold.\n#### False
    """
    raw_response = raw_response.strip()

    if "```" in raw_response:
        raw_response = raw_response.split("```")[1].strip()
        temp = raw_response.split('####')
        # assert len(temp) == 2, f"CoT response is not properly splitted into two parts by '####'. Current response: {raw_response}"
        assert len(temp) == 2, f"CoT response is not properly splitted into two parts by '####'. "
        response, answer = temp[0].strip(), temp[1].strip()
        answer = 'Unknown' if answer.lower() == "uncertain" else answer
    elif 'Here is the reasoning chain:' in raw_response:
        lines = []
        for line in raw_response.split('\n'):
            if '#### ' in line:
                answer = line.split('####')[-1].strip()
                break
            lines.append(line)
        response = '\n'.join(lines)
        response = response.split("Here is the reasoning chain:")[-1].strip()

    edges = []
    if response:
        raw_lines = [_.strip() for _ in response.splitlines() if _.strip() and '->' in _]
        # assert len(raw_lines) == 1, f"Current raw_lines = {raw_lines}"
        if len(raw_lines) > 1:
            for i in range(1, len(raw_lines)):
                raw_lines[0] += ';' + raw_lines[i]
        proofs = [_.strip().split(':')[0].split('->') for _ in raw_lines[0].split(";") if _.strip() and _.strip() != ';']
        for proof in proofs:
            consequent = proof[1]
            antecedents = [_.strip() for _ in proof[0].split('&') if _.strip()]
            for antecedent in antecedents:
                edges.append((antecedent, consequent))
        edges = list(set(edges))
    return answer, edges

def cal_proof_similarity(proof_pred: [str, List[Tuple]], proof_gold: str, target: str, debug=False, original_proof_pred=None) -> float:
    original_proofs = [proof_pred, proof_gold]


    if proof_gold.startswith("["):
        proof_gold = proof_gold[1:-1]

    # parse proof_pred
    G_pred = nx.DiGraph()
    if type(proof_pred) == str: # Prolog
        transformed_expression_pred = parse_pred.transform_string(proof_pred)
        e_list_pred = parse_pred.parse_string_into_edges(transformed_expression_pred)
    elif isinstance(proof_pred, List): # CoT
        e_list_pred = proof_pred
    else:
        raise ValueError
    # logger.info(f"e_list_pred: {e_list_pred}")
    G_pred.add_edges_from(e_list_pred)

    # parse proof_gold
    transformed_expression_gold = parse_gold.remove_brackets(parse_gold.transform_string(proof_gold))
    e_list_gold = parse_gold.parse_string_into_edges(transformed_expression_gold)
    G_gold = nx.DiGraph()
    G_gold.add_edges_from(e_list_gold)

    # edit_distance = next(nx.optimize_graph_edit_distance(G_pred, G_gold))  # Approximated Calculation
    edit_distance = nx.graph_edit_distance(G_pred, G_gold, timeout=10) # Exact Calculation

    num_nodes_edges = max(len(G_pred.nodes) + len(G_pred.edges), len(G_gold.nodes) + len(G_gold.edges))
    if num_nodes_edges == 0:
        return 1.0
    else:
        similarity =  1 - edit_distance / num_nodes_edges
    
    if debug:
        import pdb; pdb.set_trace()
    
    return similarity

def evaluate_single_query(
        prolog_code: str,
        query_code: str,
        meta_interpreter: str,
        max_depth: int,
        debug: bool,
):
    # import pdb; pdb.set_trace()
    curr_depth = 10
    while curr_depth <= max_depth:
        exec_result = consult_prolog(
            prolog_string=prolog_code,
            query_string=query_code.strip(),
            meta_interpreter=meta_interpreter,
            max_depth=curr_depth,
            debug=debug,
        )
        neg_exec_result = consult_prolog(
            prolog_string=prolog_code,
            query_string=negate_query(query_code).strip(),
            meta_interpreter=meta_interpreter,
            max_depth=curr_depth,
            debug=debug,
        )
        if exec_result['proofs'] or neg_exec_result['proofs']:
            break
        curr_depth += 5

    if exec_result['answer'] == neg_exec_result['answer']:
        pred_answer = "Unknown"
    elif exec_result['answer'] == True:
        pred_answer = "True"
    elif neg_exec_result['answer'] == True:
        pred_answer = "False"
    else:
        pred_answer = "Error"
        
    return {
        "positive_answer": exec_result['answer'],
        "negative_answer": neg_exec_result['answer'],
        "positive_proofs": list(set(exec_result['proofs'])) if exec_result['proofs'] is not None else [""],
        "negative_proofs": list(set(neg_exec_result['proofs'])) if neg_exec_result['proofs'] is not None else [""],
        "pred_answer": pred_answer,
    }

def evaluate_batch(batch_input):
    """
    batch_input: Dict[str]
        {
            "instances": List[Dict[str, Any]],
            "max_depth": int,
            "debug": bool,
            "evaluate_proof": bool,
            "use_multiprocessing": bool,
            "mode": str,
        }
    """
    instances = batch_input["instances"]
    max_depth = batch_input["max_depth"]
    debug = batch_input["debug"]
    evaluate_proof = batch_input["evaluate_proof"]
    use_multiprocessing = batch_input["use_multiprocessing"]
    mode = batch_input["mode"]

    num_correct = 0
    num_total = 0
    num_total_without_error = 0
    num_correct_hops = [0] * 10
    num_total_hops = [0] * 10
    num_total_hops_without_error = [0] * 10
    error_records = defaultdict(int)
    similarity_list = []
    similarity_list_hops = [[] for i in range(10)]
    similarity_list_correct = []
    similarity_list_hops_correct = [[] for i in range(10)]

    if not use_multiprocessing:
        pbar = tqdm(total=len(instances), smoothing=50/len(instances))
    for i, instance in enumerate(instances):
        if not use_multiprocessing:
            pbar.update()
        curr_correct = False
        
        try:
            with timeout(99999.0, RuntimeError):
                num_total += 1
                # if instance['gold_label'] != "Unknown":
                #     num_total_hops[instance['num_hops']] += 1
                num_total_hops[instance['num_hops']] += 1
                
                pred_answer, similarity = None, None

                if mode == 'prolog':
                    kb_code, query_code = preprocess_response_prolog(instance['gpt_response'])
                    if not kb_code or not query_code:
                        pred_answer = "Error"
                        similarity = 0.0
                    else:
                        try:
                            curr_output = evaluate_single_query(
                                prolog_code=kb_code,
                                query_code=query_code,
                                meta_interpreter="iter_deep_with_proof",
                                max_depth=max_depth,
                                debug=debug,
                            )
                            num_total_without_error += 1
                            # if instance['gold_label'] != "Unknown":
                                # num_total_hops_without_error[instance['num_hops']] += 1
                            num_total_hops_without_error[instance['num_hops']] += 1
                            pred_answer = curr_output['pred_answer']
                        except Exception as e:
                            pred_answer = "Error"
                            similarity = 0.0
                elif mode == 'cot':
                    query_code = None
                    try:
                        original_proof_pred = instance['gpt_response']
                        pred_answer, proof_pred = preprocess_response_cot(instance['gpt_response'])
                        num_total_without_error += 1
                        # if instance['gold_label'] != "Unknown":
                            # num_total_hops_without_error[instance['num_hops']] += 1
                        num_total_hops_without_error[instance['num_hops']] += 1
                    except Exception as e:
                        # logger.info(f"Error: {e}")
                        pred_answer = 'Error'
                        similarity = 0.0
                    # pred_answer, proof_pred = preprocess_response_cot(instance['gpt_response'])
                    # num_total_without_error += 1
                    # num_total_hops_without_error[instance['num_hops']] += 1
                
                if pred_answer != "Error":
                    debug = False
                    if pred_answer == instance['gold_label']:
                        num_correct += 1
                        curr_correct = True
                        # if instance['gold_label'] != "Unknown":
                            # num_correct_hops[instance['num_hops']] += 1
                        num_correct_hops[instance['num_hops']] += 1
                        debug = False
                    else:
                        error_records[f"Gold: {instance['gold_label']} != Pred: {pred_answer}"] += 1
                    if evaluate_proof and instance['gold_label'] != "Unknown":
                        if mode == "prolog":
                            positive_proofs = curr_output['positive_proofs'] if curr_output['positive_proofs'] else ['']
                            negative_proofs = curr_output['negative_proofs'] if curr_output['negative_proofs'] else ['']
                            proofs_pred = positive_proofs + negative_proofs
                            proofs_pred = [_.strip() for _ in proofs_pred if _.strip() != '' and 'true' in _]
                            if len(proofs_pred) == 0:
                                proof_pred = ""
                            else:
                                proof_pred = min(proofs_pred, key=len)
                            original_proof_pred = None

                        # try:
                        #     similarity = cal_proof_similarity(
                        #         proof_pred=proof_pred,
                        #         proof_gold=instance['proof'],
                        #         target=query_code,
                        #         debug=debug,
                        #         original_proof_pred=original_proof_pred,
                        #     )
                        # except Exception as e:
                        #     # logger.warning(f"Error when calculating proof similarity: {e}")
                        #     similarity = 0.0
                        similarity = cal_proof_similarity(
                            proof_pred=proof_pred,
                            proof_gold=instance['proof'],
                            target=query_code,
                            debug=debug,
                            original_proof_pred=original_proof_pred,
                        )

                if evaluate_proof \
                    and instance['gold_label'] != "Unknown" \
                    and instance['num_hops'] != 0:
                    # exclude all "Unknown" instances and "0-hop" instances when calculating similarity
                    if similarity is None or curr_correct is False:
                        similarity = 0.0
                    # if instance['num_hops'] == 0:
                    #     similarity = 1.0
                    similarity_list.append(similarity)
                    similarity_list_hops[instance['num_hops']].append(similarity)
                    if pred_answer == instance['gold_label']:
                        similarity_list_correct.append(similarity)
                        similarity_list_hops_correct[instance['num_hops']].append(similarity)
        except RuntimeError:
            logger.info(f"Skipping the {i+1}-th instance due to time-limit. ")
            continue
    if not use_multiprocessing:
        pbar.close()
    return {
        "num_correct": num_correct,
        "num_total": num_total,
        "num_total_without_error": num_total_without_error,
        "num_correct_hops": num_correct_hops,
        "num_total_hops": num_total_hops,
        "num_total_hops_without_error": num_total_hops_without_error,
        "error_records": error_records,
        "similarity_list": similarity_list,
        "similarity_list_hops": similarity_list_hops,
        "similarity_list_correct": similarity_list_correct,
        "similarity_list_hops_correct": similarity_list_hops_correct,
    }


def evaluate(
    gpt_responses: List[str],
    gold_data: List[Dict[str, Any]],
    evaluate_proof: bool,
    meta_interpreter: str,
    max_depth: int,
    debug: bool,
    use_multiprocessing: bool,
    eval_num: int,
    mode: str,
    ) -> Dict[str, Any]:

    if eval_num > 0:
        if len(gpt_responses) < eval_num:
            logger.warning(f"Number of GPT-responses={len(gpt_responses) } is less than the --eval_num={eval_num} ")
        else:
            logger.info(f"--eval_num is set. Only evaluating the first {eval_num} instances.")
            gpt_responses = gpt_responses[:eval_num]

    gold_answers = extract_gold(gold_data)

    try:
        assert len(gpt_responses) == len(gold_answers)
    except:
        logger.warning(f"Length of GPT responses ({len(gpt_responses)}) does not match length of gold answers ({len(gold_answers)}). Truncating to the shorter one.")
        min_len = min(len(gpt_responses), len(gold_answers))
        gpt_responses = gpt_responses[:min_len]
        gold_answers = gold_answers[:min_len]
    
    num_correct = 0
    num_total = 0
    num_total_without_error = 0
    num_correct_hops = [0] * 10
    num_total_hops = [0] * 10
    num_total_hops_without_error = [0] * 10
    error_records = defaultdict(int)
    similarity_list = []
    similarity_list_hops = [[] for i in range(10)]
    similarity_list_correct = []
    similarity_list_hops_correct = [[] for i in range(10)]

    all_instances = gold_answers
    for i in range(len(all_instances)):
        all_instances[i]['gpt_response'] = gpt_responses[i]
        all_instances[i]['id'] = i
    # all_instances[i] = {'id', 'gold_label', 'num_hops', 'proof', 'gpt_response'}

    if use_multiprocessing:
        num_processes = int(multiprocessing.cpu_count() * 4 / 5)
        # num_processes = min(multiprocessing.cpu_count(), 32)

        batch_instances = divide_list(all_instances, num_processes, randomize=True)
        batch_input_multiprocessing = [
            {
                "instances": batch_instances[i],
                "meta_interpreter": meta_interpreter,
                "max_depth": max_depth,
                "debug": debug,
                "evaluate_proof": evaluate_proof,
                "use_multiprocessing": use_multiprocessing,
                "mode": mode,
            }
            for i in range(num_processes)
        ]
        pool = multiprocessing.Pool(processes=num_processes)
        # imap_unordered function maps the input array to the target function and returns an iterator
        results = list(
            tqdm(
                pool.imap_unordered(evaluate_batch, batch_input_multiprocessing), 
                total=len(batch_input_multiprocessing), 
                desc=f"Using Multi-Processing. The instances are splitted into {num_processes} chunks. "
            )
        )
        # close the pool and wait for the worker processes to finish
        pool.close()
        pool.join()
    else:
        batch_input = {
            "instances": all_instances,
            "meta_interpreter": meta_interpreter,
            "max_depth": max_depth,
            "debug": debug,
            "evaluate_proof": evaluate_proof,
            "use_multiprocessing": use_multiprocessing,
            "mode": mode,
        }
        results = [evaluate_batch(batch_input)]
    
    for r in results:
        num_correct += r['num_correct']
        num_total += r['num_total']
        num_total_without_error += r['num_total_without_error']
        num_correct_hops = [num_correct_hops[i] + r['num_correct_hops'][i] for i in range(len(num_correct_hops))]
        num_total_hops = [num_total_hops[i] + r['num_total_hops'][i] for i in range(len(num_total_hops))]
        num_total_hops_without_error = [num_total_hops_without_error[i] + r['num_total_hops_without_error'][i] for i in range(len(num_total_hops_without_error))]
        for k, v in r['error_records'].items():
            error_records[k] += v
        if evaluate_proof:
            similarity_list.extend(r['similarity_list'])
            similarity_list_correct.extend(r['similarity_list_correct'])
        for i in range(len(similarity_list_hops)):
            similarity_list_hops[i].extend(r['similarity_list_hops'][i])
            similarity_list_hops_correct[i].extend(r['similarity_list_hops_correct'][i])
    
    print()
    logger.info("******************** Evaluation Results ********************")
    logger.info(f"Accuracy: {num_correct / num_total} = {num_correct} / {num_total}")
    logger.info(f"Accuracy on no-error instances: {num_correct / num_total_without_error}")
    for num_hop in range(len(num_correct_hops)):
        if num_total_hops[num_hop]:
            logger.info(f"Accuracy on {num_hop}-hop instances: {num_correct_hops[num_hop] / num_total_hops[num_hop]} = {num_correct_hops[num_hop]} / {num_total_hops[num_hop]}")
        if num_total_hops_without_error[num_hop]:
            logger.info(f"Accuracy on no-error {num_hop}-hop instances: {num_correct_hops[num_hop] / num_total_hops_without_error[num_hop]} = {num_correct_hops[num_hop]} / {num_total_hops_without_error[num_hop]}")
    print()
    logger.info("*** Error Records ***")
    for key, value in error_records.items():
        logger.info(f"{key}: {value}")
    if evaluate_proof:
        print()
        logger.info("***** Evaluating Proofs *****")
        logger.info(f"Average proof similarity: {sum(similarity_list) / len(similarity_list)} on {len(similarity_list)} instances")
        logger.info(f"Proof EM: {similarity_list.count(1.0) / len(similarity_list)}")
        for num_hop in range(len(similarity_list_hops)):
            if len(similarity_list_hops[num_hop]) > 0:
                print()
                logger.info(f"Average proof similarity on {num_hop}-hop instances: {sum(similarity_list_hops[num_hop]) / len(similarity_list_hops[num_hop])} on {len(similarity_list_hops[num_hop])} instances")
                logger.info(f"Proof EM on {num_hop}-hop instances: {similarity_list_hops[num_hop].count(1.0) / len(similarity_list_hops[num_hop])}")
        print()
        logger.info(f"Average proof similarity on correctly-predicted instances: {sum(similarity_list_correct) / len(similarity_list_correct)} on {len(similarity_list_correct)} instances")
        logger.info(f"Proof EM on correctly-predicted instances: {similarity_list_correct.count(1.0) / len(similarity_list_correct)}")
        for num_hop in range(len(similarity_list_hops_correct)):
            if len(similarity_list_hops_correct[num_hop]) > 0:
                print()
                logger.info(f"Average proof similarity on correctly-predicted {num_hop}-hop instances: {sum(similarity_list_hops_correct[num_hop]) / len(similarity_list_hops_correct[num_hop])} on {len(similarity_list_hops_correct[num_hop])} instances")
                logger.info(f"Proof EM on correctly-predicted {num_hop}-hop instances: {similarity_list_hops_correct[num_hop].count(1.0) / len(similarity_list_hops_correct[num_hop])}")
    print()
    return {
        "num_correct": num_correct,
        "num_total": num_total,
        "num_total_without_error": num_total_without_error,
        "num_correct_hops": num_correct_hops,
        "num_total_hops": num_total_hops,
        "num_total_hops_without_error": num_total_hops_without_error,
        "similarity_list": similarity_list,
        "similarity_list_hops": similarity_list_hops,
        "similarity_list_correct": similarity_list_correct,
        "similarity_list_hops_correct": similarity_list_hops_correct,
    }

def main(args):

    print()
    logger.info("**************************************")
    logger.info("********** Evaluation Start **********")
    logger.info("**************************************")
    print()
    logger.info("******* Arguments *******")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    print()

    # Load GPT responses and Gold Data
    with open(args.gpt_response_path, 'r', encoding='utf-8') as f:
        gpt_responses = [json.loads(line)['response'][0] for line in f.readlines() if line.strip() != '']
    with open(args.gold_path, 'r', encoding='utf-8') as f:
        gold_data = [json.loads(line) for line in f.readlines() if line.strip() != '']

    # Evaluation
    eval_results = evaluate(
        gpt_responses=gpt_responses,
        gold_data=gold_data,
        evaluate_proof=args.evaluate_proof,
        meta_interpreter="iter_deep_with_proof",
        max_depth=20,
        debug=args.debug,
        use_multiprocessing=args.use_multiprocessing,
        eval_num=args.eval_num,
        mode=args.mode,
    )
    
    if args.output_path is None:
        if args.evaluate_proof:
            output_path = os.path.join(os.path.split(args.gpt_response_path)[0], f"eval_results_with_proof.{os.path.split(args.gpt_response_path)[1]}")
        else:
            output_path = os.path.join(os.path.split(args.gpt_response_path)[0], f"eval_results.{os.path.split(args.gpt_response_path)[1]}")
    else:
        output_path = args.output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f)