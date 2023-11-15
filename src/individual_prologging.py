import pyswip
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert_path", type=str, required=True, help="")
    parser.add_argument("--mi_path", type=str, required=True, help="")
    parser.add_argument('--output_path', type=str, required=True)
    # parser.add_argument('--query', type=str, required=True)
    parser.add_argument("--max_result", type=int, default=20)

    args = parser.parse_args()

    prolog = pyswip.Prolog()
    
    with open(args.assert_path, 'r', encoding='utf-8') as f:
        _clauses = [_.strip() for _ in f.readlines() if _.strip()]
    query = _clauses[-1]
    clauses = _clauses[:-1]

    query = query.rstrip('.')
    
    try:
        prolog.consult(args.mi_path)
        for clause in clauses:
            clause = clause.rstrip('.')
            prolog.assertz(clause)
            
        results = prolog.query(query, maxresult=args.max_result)

        with open(args.output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r).strip() + '\n')
    except pyswip.prolog.PrologError as e:
        pass
