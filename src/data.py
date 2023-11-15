import json
import jsonlines
import os
import time
from typing import List, Dict, Any, Tuple, Union
import copy


##############################################
#              Main                          #
##############################################
def load_dataset(file_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    dataset_to_function = {
        "proofwriter": open_proofwriter,
        "gsm8k": open_gsm8k,
        'prontoqa': open_prontoqa,
    }
    data = dataset_to_function[dataset_name](file_path)
    return data

##############################################
#              ProofWriter                   #
##############################################
def open_proofwriter(file_path):
    if file_path.endswith(".jsonl"):
        with jsonlines.open(file_path) as reader:
            data = [obj for obj in reader]
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f.readlines() if line.strip()]
    else:
        file_extension = os.path.splitext(file_path)[1]
        raise ValueError(f'Invalid extension "{file_extension}" for file "{file_path}"')
    outputs = []
    for d in data:
        for i in range(100):
        # for k, v in d['questions'].items():
            if f'Q{i+1}' not in d['questions']:
                break
            v = copy.deepcopy(d['questions'][f'Q{i+1}'])
            outputs.append(copy.deepcopy(d))
            outputs[-1]['questions'] = None
            for kk, vv in v.items():
                outputs[-1][kk] = vv
    return outputs

##############################################
#              GSM8K                         #
##############################################
def open_gsm8k(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines() if line.strip()]
    return data

##############################################
#              PrOntoQA                      #
##############################################
def open_prontoqa(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines() if line.strip()]
    return data