import tempfile
import os
import re
import subprocess
import json

curr_dir = os.path.dirname(os.path.abspath(__file__))

def extract_clauses_from_code(prolog_code: str):
    lines = prolog_code.split("\n")
    clauses = []

    continue_signal = False
    for i, line in enumerate(lines):
        if line.startswith("%") or line.startswith("/*") or line.strip() == '':
            continue_signal = False
            continue
        
        if "%" in line:
            line = line.split("%")[0].strip()
        if "/*" in line:
            line = line.split("/*")[0].strip()

        if continue_signal and line.startswith(" "):
            clauses[-1] += ' ' + line.strip()
        else:
            clauses.append(line)
            continue_signal = True
    clauses = [_.strip().rstrip('.') for _ in clauses]
    
    predicates = []
    for clause in clauses:
        if clause.startswith(":-"):
            continue
        if ':-' in clause:
            head, body = clause.split(":-")
            predicates.extend(
                [_.strip() for _ in head.split("(")[:-1]]
            )
    return clauses, set(predicates)

##############################################
#              Main Function                 #
#            To call SWI-Prolog              #
##############################################
def consult_prolog(
        prolog_string,
        query_string,
        meta_interpreter="raw",
        max_depth=5,
        debug=False,
        dataset_name="vanilla",
):
    
    """
    Args:
        prolog_string:
            string, the string of Prolog knwoledge base to be consulted
        query_string:
            string, the string of Prolog query to be executed
        consult_raw_query:
            bool, whether to consult the raw query, i.e., **NO** special meta-interpreter is used.
        generate_proof_tree:
            bool, whether to generate the proof tree for the query
        max_depth:
            int, the maximum depth of the iterative deepening search
        debug:
            bool, whether to print all the inputs and outputs when interacting with SWI-Prolog
        dataset_name:
            string, the name of the dataset, determines which meta-interpreter_*.pl to use
    """


    prolog_meta_interpreters = {
        "raw": "{}",
        "with_proof": "mi_tree(g({}), Proof)",  # With proof generation. One argument: Goal
        "iter_deep_with_proof": "mi_id_limit(g({}), Proof, {})",   # Iterative deepening search, with proof generation. Two arguments: Goal, MaxDepth
        # "iter_deep_no_proof": prolog_output_all_answers.format("mi_id_limit_no_proof(g({}), {})"),  # Iterative deepening search. Two arguments: Goal, MaxDepth
        "iter_deep_no_proof": "mi_id_limit_no_proof(g({}), {})",  # Iterative deepening search. Two arguments: Goal, MaxDepth
    }

    ########################################
    clauses, predicates = extract_clauses_from_code(prolog_string)
    ########################################

    if query_string.endswith('.'):
        query_string = query_string[:-1].strip()
    # which types of meta-interpreters to use for querying Prolog
    if "iter_deep" not in meta_interpreter:
        user_query = prolog_meta_interpreters[meta_interpreter].format(query_string)
    else:
        user_query = prolog_meta_interpreters[meta_interpreter].format(query_string, max_depth)

    # import pdb; pdb.set_trace()

    # Write the Prolog knowledge base to a temporary file.
    tmp_clause_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    with open(tmp_clause_file.name, 'w') as f:
        f.writelines(
            [clause.strip() + '\n' for clause in clauses] + [user_query + '\n']
        )
    tmp_output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    mi_path = os.path.join(file_path, "meta_interpreter.pl")
    tmp_clause_path = os.path.abspath(tmp_clause_file.name)
    tmp_output_path = os.path.abspath(tmp_output_file.name)

    ###### Execute Prolog ######
    command = [
            "python",
            f"{curr_dir.split('/src/')[0]}/src/individual_prologging.py",
            "--assert_path",
            tmp_clause_path,
            "--mi_path",
            mi_path,
            "--output_path",
            tmp_output_path,
        ]
    response= subprocess.run(
        command,
    )

    if response.returncode == 0:
        with open(tmp_output_file.name, 'r', encoding='utf-8') as f:
            results = [json.loads(_) for _ in f.readlines() if _.strip()]
    else:
        results = []

    output = {
        'answer': None,
        'proofs': None,
    }

    num_results = 0
    for r in results:
        num_results += 1
        if "Proof" in r:
            if output["proofs"] is None:
                output["proofs"] = [r['Proof']]
            else:
                output["proofs"].append(r['Proof'])
    if num_results > 0:
        output['answer'] = True

    os.remove(tmp_clause_file.name)
    os.remove(tmp_output_file.name)
    
    # import pdb; pdb.set_trace()
    
    return output