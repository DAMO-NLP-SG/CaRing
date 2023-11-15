import networkx as nx
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

def transform_string(input_str):
    # Pattern matching for '(ruleX % intY)'
    pattern1 = re.compile(r'\(([^)]+)%([^)]+)\)')
    
    # Replacement function for '%'
    def replacer1(match):
        return f"({match.group(1)}) > ({match.group(2)})"
    
    # Transform '%'
    intermediate_str = pattern1.sub(replacer1, input_str)
    
    # Replace '->' with '&'
    transformed_str = intermediate_str.replace('->', ',')
    
    return transformed_str.replace(" ", "")

def remove_brackets(input_string):
    # The pattern r'\((\w+)\)' captures any alphanumeric word (\w+) enclosed in brackets.
    # \1 in the replacement string refers to the first captured group.
    return re.sub(r'\((\w+)\)', r'\1', input_string)

def check_internal_balance_point(input_str):
    bracket_num = 0
    for i, char in enumerate(input_str):
        if char == '(':
            bracket_num += 1
        elif char == ')':
            bracket_num -= 1
            if bracket_num == 0 and i+1 < len(input_str) and input_str[i+1] not in (',', ')', '>'):
                return True
    return False

def remove_brackets_from_both_sides(input_str):
    while input_str.startswith('(') and input_str.endswith(')'):
        have_internal_balance_point = check_internal_balance_point(input_str)
        new_str = input_str[1:-1]
        if not have_internal_balance_point:
            input_str = new_str
        else:
            if check_internal_balance_point(new_str) != have_internal_balance_point:
                break
            else:
                input_str = new_str

    return input_str

def parse_string_into_edges(proof_string: str, debug=False) -> List[Tuple[str, str]]:
    # proof_string: (((triple1,rule1>int2)triple1),rule7>int1)
    # tag_set: {}
    # return: [('triple1', ''int2'), ('rule1', 'int2'), ('triple1', 'int1'), ('rule7', 'int1'), ('int2', 'int1')]
    stack = [proof_string]
    rhs_stack = ['Root']
    edges = []
    deduce_symbol = '>'
    and_symbol = ','
    while stack:
        current = stack.pop()
        rhs = rhs_stack.pop()
        current = remove_brackets_from_both_sides(current)
        bracket_num = 0
        next_loop = False
        for i, char in enumerate(current):
            if char == '(':
                bracket_num += 1
            elif char == ')':
                bracket_num -= 1
            elif char == deduce_symbol and bracket_num == 0:
                stack.append(current[:i])
                rhs_stack.append(current[i+1:])
                stack.append(current[i+1:])
                rhs_stack.append(rhs)
                next_loop = True
                break
        if not next_loop: # if not found any ">" symbol, we try "," symbol
            for i, char in enumerate(current):
                if char == '(':
                    bracket_num += 1
                elif char == ')':
                    bracket_num -= 1
                    if bracket_num == 0 and i+1 < len(current) and current[i+1] not in (',', ')', '>'):
                        stack.append(current[:i+1])
                        stack.append(current[i+1:])
                        rhs_stack.append(rhs)
                        rhs_stack.append(rhs)
                        next_loop = True
                        break
                elif char == and_symbol and bracket_num == 0:
                    stack.append(current[:i])
                    stack.append(current[i+1:])
                    rhs_stack.append(rhs)
                    rhs_stack.append(rhs)
                    next_loop = True
                    break
            if not next_loop and (deduce_symbol in current or and_symbol in current):
                for i, char in enumerate(current):
                    if char == '(':
                        stack.append(current[:i])
                        stack.append(current[i:])
                        rhs_stack.append(rhs)
                        rhs_stack.append(rhs)
                        next_loop = True
                        break
        if debug:
            print(f"current: {current}")
            print(f"rhs: {rhs}")
        if not next_loop:
            edges.append((current, rhs))
            if debug:
                print(f"New edge: {current} -> {rhs}")
        if debug:
            print()
    edges = [_ for _ in edges if _[1] != 'Root' and not _[0].startswith('rule')]
    edges = list(set(edges))
    return edges


if __name__ == "__main__":
    # expression = "((triple1 ((triple1 ((((((triple1) -> (rule1 % int5)) triple1) -> (rule7 % int4))) -> (rule9 % int3))) -> (rule3 % int2))) -> (rule5 % int1))"
    expression = "((((triple1 ((triple1) -> (rule3 % int5))) -> (rule6 % int4)) ((((((triple1 ((triple1) -> (rule3 % int5))) -> (rule6 % int4))) -> (rule1 % int3)) ((triple1) -> (rule3 % int5))) -> (rule7 % int2))) -> (rule2 % int1))"
    # expression = "((((triple1) -> (rule1 % int2)) triple1) -> (rule7 % int1))"
    graph = nx.DiGraph()

    transformed_expression = remove_brackets(transform_string(expression))

    print()
    print("Original expression:")
    print(transformed_expression)
    print()

    edges = parse_string_into_edges(transformed_expression, debug=False)
    print("Edges:")
    print(edges)
    print()

    graph.add_edges_from(edges)

    # Drawing the graph
    # nx.draw(graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=18)
    # plt.show()


    # ProofWriter-Gold: ((((triple1) -> (rule1 % int2)) triple1) -> (rule7 % int1))
    # ProofWriter-Pred: =>(=>(,(=>(true, round(harry)), ,(=>(=>(true, big(harry)), tall(harry)), =>(true, kind(harry)))), wizard(harry)), smart(harry))
    # GSM8K-Gold: 
    # GSM8K-Pred: (builtin((g(records(200)),g(800 is 4*200)))=>sammy_offer(800),(builtin((g(records(200)),g(600 is 6*(200/2))))=>bryan_interested_offer(600),builtin((g(records(200)),g(100 is 1*(200/2))))=>bryan_not_interested_offer(100),builtin(700 is 600+100)=>700 is 600+100)=>bryan_total_offer(700),builtin(100 is 800-700)=>100 is 800-700)=>profit_difference(100)