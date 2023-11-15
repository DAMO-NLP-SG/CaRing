
import os, json
from typing import List, Dict, Any, Tuple, Union

from llama_utils import construct_prompt, Message, Dialog


##############################################
#                   Main                     #
##############################################

def build_demo(
    demo_indices: List[int],
    demonstration_path: str,
    dataset_name: str,
    mode: str,
):
    ########## Add new dataset here ##########
    dataset_to_build_demo = {
        "proofwriter": build_demo_proofwriter,
        "gsm8k": build_demo_gsm8k,
        "prontoqa": build_demo_prontoqa,
    }
    ###########################################
    prompt_demo = dataset_to_build_demo[dataset_name](
        demo_indices=demo_indices,
        demonstration_path=demonstration_path,
        mode=mode,
    )
    return prompt_demo
        
def build_prompt(
    example: Dict[str, Any],
    prompt_demo: str,
    dataset_name: str,
    mode: str,
    instruct: bool = True,
):
    ########## Add new dataset here ##########
    dataset_to_build_prompt = {
        "proofwriter": build_prompt_proofwriter,
        "gsm8k": build_prompt_gsm8k,
        "prontoqa": build_prompt_prontoqa,
    }
    ###########################################
    prompt_text = dataset_to_build_prompt[dataset_name](
        example=example,
        prompt_demo=prompt_demo,
        mode=mode,
        instruct=instruct,
    )
    return prompt_text


##############################################
#              ProofWriter                   #
##############################################

def build_demo_proofwriter(
    demo_indices: List[int],
    demonstration_path: str,
    mode: str,  
):
    # load demonstrations
    demos = []
    for i, index in enumerate(demo_indices):
        if not os.path.exists(os.path.join(demonstration_path, f"problem_{index}.txt")):
            if not os.path.exists(demonstration_path):
                raise FileNotFoundError(f"Cannot find {demonstration_path}")
            else:
                raise FileNotFoundError(f"Cannot find problem_{index}.txt in {demonstration_path}")
        curr = dict()
        with open(os.path.join(demonstration_path, f"problem_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['text'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"prolog_{index}.pl"), 'r', encoding='utf-8') as f:
            curr['prolog'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"cot_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['cot'] = f.read().strip()
            curr['direct'] = curr['cot'].split('####')[-1].strip()
        # if mode == 'prolog':
        #     demo_text += template.DEMO_TEMPLATE_PROLOG.format(i+1, curr['text'], curr['code'], i+1)
        # elif mode == 'cot':
        #     demo_text += template.DEMO_TEMPLATE_COT.format(i+1, curr['text'], curr['cot'], i+1)
        # else:
        #     raise ValueError(f"Unknown mode: {mode}")
        demos.append(curr)
    return demos

def build_prompt_proofwriter(
    example: Dict[str, Any],
    prompt_demo: List[Dict],
    mode: str,
    instruct: bool = True,
):
    import llm_prompts_related.prompt_templates.template_proofwriter as template
    if mode == 'prolog':
        NEW_PROBLEM_TEMPLATE = template.NEW_PROBLEM_TEMPLATE_PROLOG
        INSTRUCTION = template.INSTRUCTION_PROLOG
        PLACE_HOLDER = "Sure! I am happy to help you write Prolog code about this reasoning problem. Here is the Prolog code:\n```prolog\n{}\n```\n"
        system_message = "You are a helpful assistant who **produces Prolog code** to solve problems.\n"
    elif mode == 'cot':
        NEW_PROBLEM_TEMPLATE = template.NEW_PROBLEM_TEMPLATE_COT
        INSTRUCTION = template.INSTRUCTION_COT
        PLACE_HOLDER = "Sure! I am happy to help you solve this reasoning problem. Here is the reasoning chain:\n{}\n"
        system_message = "You are a helpful assistant that helps people solve problems.\n"
    elif mode == 'direct':
        NEW_PROBLEM_TEMPLATE = template.NEW_PROBLEM_TEMPLATE_COT
        INSTRUCTION = template.INSTRUCTION_DIRECT
        PLACE_HOLDER = "Sure! I am happy to help you solve this reasoning problem. Here is the answer:\n{}\n"
        system_message = "You are a helpful assistant that helps people solve problems.\n"

    triples = [_['text'] for _ in example['triples'].values()]
    rules = [_['text'] for _ in example['rules'].values()]
    question = example['question']
    problem_text = NEW_PROBLEM_TEMPLATE.format(
        "\n".join(["triple-{}: {}".format(i+1, _) for i, _ in enumerate(triples)]),
        "\n".join(["rule-{}: {}".format(i+1, _) for i, _ in enumerate(rules)]),
        "statement: {}".format(question),
    )

    if instruct:
        if mode == 'direct':
            raise NotImplementedError
        dialog = []
        dialog.append({"role": "system", "content": system_message})
        dialog.append({
            "role": "user", "content": f"{INSTRUCTION}\n\nHere is the problem:\n\n{prompt_demo[0]['text']}\n"
        })
        dialog.append({
            "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[0][mode])
        })
        for i in range(1, len(prompt_demo)):
            dialog.append({
                "role": 'user', 
                "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{prompt_demo[i]['text']}\n"
            })
            dialog.append({
                "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[i][mode])
            })
        dialog.append({
            'role': "user", "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{problem_text}\n"
        })
        # dialog = Dialog(dialog)

        dialog_texts = construct_prompt(dialog)
        output_prompt = '\n'.join(dialog_texts)
    else:
        if mode == 'prolog':
            output_prompt = "Below are a few examples on how to write Prolog code to solve reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nProlog Code:\n\n```prolog\n{}\n```\n"
        elif mode == 'cot':
            output_prompt = "Below are a few examples on how to think step-by-step to solve reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nReasoning Chain:\n\n```\n{}\n```\n"
        elif mode == 'direct':
            output_prompt = "Below are a few examples and answers on logic reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nAnswer:\n\n```\n#### {}\n```\n"
        for i in range(len(prompt_demo)):
            output_prompt += icl_template.format(i+1, prompt_demo[i]['text'], prompt_demo[i][mode])
        # add the new problem
        output_prompt += "\n### Problem No.{}\n\n```{}```\n".format(len(prompt_demo) + 1, problem_text)

    return output_prompt


##############################################
#                GSM8k                       #
##############################################

def build_demo_gsm8k(
    demo_indices: List[int],
    demonstration_path: str,
    mode: str,
):
    import llm_prompts_related.prompt_templates.template_gsm8k as template
    # load demonstrations
    demos = []
    for i, index in enumerate(demo_indices):
        if not os.path.exists(os.path.join(demonstration_path, f"problem_{index}.txt")):
            if not os.path.exists(demonstration_path):
                raise FileNotFoundError(f"Cannot find {demonstration_path}")
            else:
                raise FileNotFoundError(f"Cannot find problem_{index}.txt in {demonstration_path}")
        curr = dict()
        with open(os.path.join(demonstration_path, f"problem_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['text'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"prolog_{index}.pl"), 'r', encoding='utf-8') as f:
            curr['prolog'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"cot_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['cot'] = f.read().strip()
            curr['direct'] = curr['cot'].split('####')[-1].strip()
        with open(os.path.join(demonstration_path, f"prolog_cot_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['prolog_cot'] = f.read().strip()
        # if mode == "prolog":
        #     demo_text += template.DEMO_TEMPLATE_PROLOG.format(i+1, curr['text'], curr['prolog_cot'], curr['code'], i+1)
        # elif mode == "cot":
        #     demo_text += template.DEMO_TEMPLATE_COT.format(i+1, curr['text'], curr['cot'])
        demos.append(curr)
    return demos

def build_prompt_gsm8k(
    example: Dict[str, Any],
    prompt_demo: str,
    mode: str,
    instruct: bool,
):
    import llm_prompts_related.prompt_templates.template_gsm8k as template
    if mode == 'prolog':
        # NEW_PROBLEM_TEMPLATE = template.NEW_PROBLEM_TEMPLATE_PROLOG
        INSTRUCTION = template.INSTRUCTION_PROLOG
        PLACE_HOLDER = "Sure! I am happy to help you write Prolog code to solve this arithmetic reasoning problem. Here is the Prolog code:\n```prolog\n{}\n```\n"
        system_message = "You are a helpful assistant who **produces Prolog code** to solve problems.\n"
    elif mode == 'cot':
        # NEW_PROBLEM_TEMPLATE = template.NEW_PROBLEM_TEMPLATE_COT
        INSTRUCTION = template.INSTRUCTION_COT
        PLACE_HOLDER = "Sure! I am happy to help you solve this arithmetic reasoning problem. Here is the reasoning chain:\n{}\n"
        system_message = "You are a helpful assistant that helps people solve problems.\n"
    problem_text = example['question']

    if instruct:
        dialog = []
        dialog.append({"role": "system", "content": system_message})
        dialog.append({
            "role": "user", "content": f"{INSTRUCTION}\n\nHere is the problem:\n\n{prompt_demo[0]['text']}\n"
        })
        dialog.append({
            "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[0][mode])
        })
        for i in range(1, len(prompt_demo)):
            dialog.append({
                "role": 'user', 
                "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{prompt_demo[i]['text']}\n"
            })
            dialog.append({
                "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[i][mode])
            })
        dialog.append({
            'role': "user", "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{problem_text}\n"
        })
        
        dialog_texts = construct_prompt(dialog)
        output_prompt = '\n'.join(dialog_texts)
    else:
        if mode == 'prolog':
            output_prompt = "Below are a few examples on how to write Prolog code to solve arithmetic reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nReasoning Process: {}\n\nProlog Code:\n\n```prolog\n{}\n```\n"
            # icl_template = "\n### Problem No.{}\n\n```{}```\n\nProlog Code:\n\n```prolog\n{}\n```\n"
        elif mode == 'cot':
            output_prompt = "Below are a few examples on how to think step-by-step to solve arithmetic reasoning problems.\n\n"
            # icl_template = "\n### Problem No.{}\n\n```{}```\n\nReasoning Process: {}\n\nReasoning Chain:\n\n```\n{}\n```\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nReasoning Chain:\n\n```\n{}\n```\n"
        elif mode == 'direct':
            output_prompt = "Below are a few examples and answers about arithmetic reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nAnswer:\n\n```\n#### {}\n```\n"
        for i in range(len(prompt_demo)):
            if mode == 'prolog':
                output_prompt += icl_template.format(i+1, prompt_demo[i]['text'], prompt_demo[i]['prolog_cot'], prompt_demo[i][mode])
            elif mode == 'cot' or mode == 'direct':
                output_prompt += icl_template.format(i+1, prompt_demo[i]['text'], prompt_demo[i][mode])
            else:
                raise ValueError()
        # add the new problem
        output_prompt += "\n### Problem No.{}\n\n```{}```\n".format(len(prompt_demo) + 1, problem_text)

    return output_prompt

##############################################
#                Game24                      #
##############################################

def build_demo_game24():
    raise NotImplementedError

def build_prompt_game24():
    raise NotImplementedError

##############################################
#                PrOntoQA                    #
##############################################

def build_demo_prontoqa(
    demo_indices: List[int],
    demonstration_path: str,
    mode: str,  
):
    # load demonstrations
    demos = []
    for i, index in enumerate(demo_indices):
        if not os.path.exists(os.path.join(demonstration_path, f"problem_{index}.txt")):
            if not os.path.exists(demonstration_path):
                raise FileNotFoundError(f"Cannot find {demonstration_path}")
            else:
                raise FileNotFoundError(f"Cannot find problem_{index}.txt in {demonstration_path}")
        curr = dict()
        with open(os.path.join(demonstration_path, f"problem_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['text'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"prolog_{index}.pl"), 'r', encoding='utf-8') as f:
            curr['prolog'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"cot_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['cot'] = f.read().strip()
            curr['direct'] = curr['cot'].split('####')[-1].strip()

        # if mode == 'prolog':
        #     demo_text += template.DEMO_TEMPLATE_PROLOG.format(i+1, curr['text'], curr['code'], i+1)
        # elif mode == 'cot':
        #     demo_text += template.DEMO_TEMPLATE_COT.format(i+1, curr['text'], curr['cot'], i+1)
        # else:
        #     raise ValueError(f"Unknown mode: {mode}")
        demos.append(curr)
    return demos

def build_prompt_prontoqa(
    example: Dict[str, Any],
    prompt_demo: List[Dict],
    mode: str,
    instruct: bool = True,
):

    
    import llm_prompts_related.prompt_templates.template_prontoqa as template
    NEW_PROBLEM_TEMPLATE = template.NEW_PROBLEM_TEMPLATE
    if mode == 'prolog':
        INSTRUCTION = template.INSTRUCTION_PROLOG
        PLACE_HOLDER = "Sure! I am happy to help you write Prolog code about this reasoning problem. Here is the Prolog code:\n```prolog\n{}\n```\n"
        system_message = "You are a helpful assistant who **produces Prolog code** to solve problems.\n"
    elif mode == 'cot':
        INSTRUCTION = template.INSTRUCTION_COT
        PLACE_HOLDER = "Sure! I am happy to help you solve this reasoning problem. Here is the reasoning chain:\n{}\n"
        system_message = "You are a helpful assistant that helps people solve problems.\n"
    elif mode == 'direct':
        INSTRUCTION = template.INSTRUCTION_DIRECT
        PLACE_HOLDER = "Sure! I am happy to help you solve this reasoning problem. Here is the answer:\n{}\n"
        system_message = "You are a helpful assistant that helps people solve problems.\n"
    context = example['context']
    question = example['question']
    problem_text = template.NEW_PROBLEM_TEMPLATE.format(
        "\n".join(["statement-{}: {}.".format(i+1, _.strip()) for i, _ in enumerate(context.split('.')) if _.strip()]),
        question,
    )
    
    if instruct:
        if mode == 'direct':
            raise NotImplementedError
        dialog = []
        dialog.append({"role": "system", "content": system_message})
        dialog.append({
            "role": "user", "content": f"{INSTRUCTION}\n\nHere is the problem:\n\n{prompt_demo[0]['text']}\n"
        })
        dialog.append({
            "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[0][mode])
        })
        for i in range(1, len(prompt_demo)):
            dialog.append({
                "role": 'user', 
                "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{prompt_demo[i]['text']}\n"
            })
            dialog.append({
                "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[i][mode])
            })
        dialog.append({
            'role': "user", "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{problem_text}\n"
        })
        # dialog = Dialog(dialog)

        dialog_texts = construct_prompt(dialog)
        output_prompt = '\n'.join(dialog_texts)
    else:
        
        if mode == 'prolog':
            output_prompt = "Below are a few examples on how to write Prolog code to solve reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nProlog Code:\n\n```prolog\n{}\n```\n"
        elif mode == 'cot':
            output_prompt = "Below are a few examples on how to think step-by-step to solve reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nReasoning Process:\n\n```\n{}\n```\n"
        elif mode == 'direct':
            output_prompt = "Below are a few examples and answers on logic reasoning problems.\n\n"
            icl_template = "\n### Problem No.{}\n\n```{}```\n\nAnswer:\n\n```\n#### {}\n```\n"
        for i in range(len(prompt_demo)):
            output_prompt += icl_template.format(i+1, prompt_demo[i]['text'], prompt_demo[i][mode])

        # add the new problem
        output_prompt += "\n### Problem No.{}\n\n```{}```\n".format(len(prompt_demo) + 1, problem_text)

    return output_prompt