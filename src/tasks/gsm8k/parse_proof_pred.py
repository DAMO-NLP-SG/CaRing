import networkx as nx
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

def transform_string(input_str):
    # replace "=>" with ">"
    transformed_str = input_str.replace("=>", ">")

    # replace " " with ""
    transformed_str = transformed_str.replace(" ", "")

    # # replace "is" with "="
    # transformed_str = replace_is_with_equal(transformed_str)

    return transformed_str

def replace_is_with_equal(input_str: str) -> str:
    return re.sub(r'\bis\b', '=', input_str)

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
    """
    proof_string:
        >(,(>(,(>(,(>(builtin(true),wage(18.0)),>(builtin(is(144.0,*(8,18.0))),is(144.0,*(8,18.0)))),regular_earning_for_day(144.0)),,(>(,(>(>(builtin(is(2,-(10,8))),is(2,-(10,8))),overtime_hours(2)),,(>(,(>(builtin(true),wage(18.0)),>(builtin(is(27.0,*(1.5,18.0))),is(27.0,*(1.5,18.0)))),overtime_wage(27.0)),>(builtin(is(54.0,*(2,27.0))),is(54.0,*(2,27.0))))),overtime_earning_for_day(54.0)),>(builtin(is(198.0,+(144.0,54.0))),is(198.0,+(144.0,54.0))))),total_earning_for_day(198.0)),>(builtin(is(990.0,*(5,198.0))),is(990.0,*(5,198.0)))),total_earning_for_5_days(990.0))
    return: 

    """
    stack = [proof_string]
    rhs_stack = ['Root']
    edges = []
    deduce_symbol = '>'
    and_symbol = ','
    while stack:
        current = stack.pop()
        rhs = rhs_stack.pop()
        bracket_num = 0
        next_loop = False
        lowest_level = False

        if debug:
            print(f"current: {current}")
            print(f"rhs: {rhs}")
        
        current = remove_brackets_from_both_sides(current)

        ############ Special Conditions with GSM8K ############
        # handle "builtin": directly ignore "builtin(X)"
        if current.startswith('builtin('):
            # continue
            assert current.endswith(')')
            current = current[len('builtin'): -1]
        
        # # handle "is"
        # if current.startswith('is('):
        #     lowest_level = True
        #     # continue

        if ',' in current and not lowest_level:
            if current.startswith("is("):
                current_symbol = deduce_symbol
                current = remove_brackets_from_both_sides(current[2:])
            elif current[0] in {'+', '-', '*', '/'}:
                current_symbol = and_symbol
                current = remove_brackets_from_both_sides(current[1:])
            else:
                current_symbol = current.strip()[0]
                current = remove_brackets_from_both_sides(current[1:])

            for i, char in enumerate(current):
                if char == '(':
                    bracket_num += 1
                elif char == ')':
                    bracket_num -= 1
                elif char == ',' and bracket_num == 0 and i != 0:
                    # ">"
                    if current_symbol == deduce_symbol:
                        stack.append(current[:i])
                        rhs_stack.append(current[i+1:])
                        stack.append(current[i+1:])
                        rhs_stack.append(rhs)
                    # "," AND
                    elif current_symbol == and_symbol:
                        stack.append(current[:i])
                        rhs_stack.append(rhs)
                        stack.append(current[i+1:])
                        rhs_stack.append(rhs)
                    else:
                        raise ValueError("Unknown symbol: {}".format(current_symbol))
                    next_loop = True
                    break

        if lowest_level or not next_loop:
            edges.append((current, rhs))
            if debug:
                print(f"New edge: {current} -> {rhs}")
        if debug:
            print()
    edges = [_ for _ in edges if _[1] != 'Root' and _[0] != 'true']
    edges = list(set(edges))
    return edges


if __name__ == "__main__":
    expression = "=>(,(=>(,(=>(,(=>(builtin(true), wage(18.0)), =>(builtin(is(144.0, *(8, 18.0))), is(144.0, *(8, 18.0)))), regular_earning_for_day(144.0)), ,(=>(,(=>(=>(builtin(is(2, -(10, 8))), is(2, -(10, 8))), overtime_hours(2)), ,(=>(,(=>(builtin(true), wage(18.0)), =>(builtin(is(27.0, *(1.5, 18.0))), is(27.0, *(1.5, 18.0)))), overtime_wage(27.0)), =>(builtin(is(54.0, *(2, 27.0))), is(54.0, *(2, 27.0))))), overtime_earning_for_day(54.0)), =>(builtin(is(198.0, +(144.0, 54.0))), is(198.0, +(144.0, 54.0))))), total_earning_for_day(198.0)), =>(builtin(is(990.0, *(5, 198.0))), is(990.0, *(5, 198.0)))), total_earning_for_5_days(990.0))"
    # expression = "((((triple1) -> (rule1 % int2)) triple1) -> (rule7 % int1))"
    graph = nx.DiGraph()

    transformed_expression = transform_string(expression)

    print()
    print("Original expression:")
    print(transformed_expression)
    print()

    # raise KeyboardInterrupt

    edges = parse_string_into_edges(transformed_expression, debug=False)
    print("Edges:")
    print(edges)
    print()

    graph.add_edges_from(edges)

    # Drawing the graph
    # nx.draw(graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=18)
    # plt.show()


    # ProofWriter-Gold: ((((triple1) -> (rule1 % int2)) triple1) -> (rule7 % int1))
    # ((((triple1 ((triple1) -> (rule3 % int5))) -> (rule6 % int4)) ((((((triple1 ((triple1) -> (rule3 % int5))) -> (rule6 % int4))) -> (rule1 % int3)) ((triple1) -> (rule3 % int5))) -> (rule7 % int2))) -> (rule2 % int1))
    # ProofWriter-Pred: =>(=>(,(=>(true, round(harry)), ,(=>(=>(true, big(harry)), tall(harry)), =>(true, kind(harry)))), wizard(harry)), smart(harry))
    # =>(,(=>(,(=>(true, nice(bob)), =>(=>(true, nice(bob)), big(bob))), rough(bob)), =>(,(=>(=>(,(=>(true, nice(bob)), =>(=>(true, nice(bob)), big(bob))), rough(bob)), furry(bob)), =>(=>(true, nice(bob)), big(bob))), cold(bob))), white(bob))
    # GSM8K-Gold: 
    # GSM8K-Pred: =>(,(=>(,(=>(,(=>(builtin(true), wage(18.0)), =>(builtin(is(144.0, *(8, 18.0))), is(144.0, *(8, 18.0)))), regular_earning_for_day(144.0)), ,(=>(,(=>(=>(builtin(is(2, -(10, 8))), is(2, -(10, 8))), overtime_hours(2)), ,(=>(,(=>(builtin(true), wage(18.0)), =>(builtin(is(27.0, *(1.5, 18.0))), is(27.0, *(1.5, 18.0)))), overtime_wage(27.0)), =>(builtin(is(54.0, *(2, 27.0))), is(54.0, *(2, 27.0))))), overtime_earning_for_day(54.0)), =>(builtin(is(198.0, +(144.0, 54.0))), is(198.0, +(144.0, 54.0))))), total_earning_for_day(198.0)), =>(builtin(is(990.0, *(5, 198.0))), is(990.0, *(5, 198.0)))), total_earning_for_5_days(990.0))

    # >(,(>(true,round(harry)),,(>(>(true,big(harry)),tall(harry)),>(true,kind(harry)))),wizard(harry)),smart(harry)

    # >(,(>(,(>(,(>(builtin(true),wage(18.0)),>(builtin(=(144.0,*(8,18.0))),=(144.0,*(8,18.0)))),regular_earning_for_day(144.0)),,(>(,(>(>(builtin(=(2,-(10,8))),=(2,-(10,8))),overtime_hours(2)),,(>(,(>(builtin(true),wage(18.0)),>(builtin(=(27.0,*(1.5,18.0))),=(27.0,*(1.5,18.0)))),overtime_wage(27.0)),>(builtin(=(54.0,*(2,27.0))),=(54.0,*(2,27.0))))),overtime_earning_for_day(54.0)),>(builtin(=(198.0,+(144.0,54.0))),=(198.0,+(144.0,54.0))))),total_earning_for_day(198.0)),>(builtin(=(990.0,*(5,198.0))),=(990.0,*(5,198.0)))),total_earning_for_5_days(990.0))