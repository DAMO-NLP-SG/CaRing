% Prolog

:- use_module(library(clpfd)).

% set_prolog_flag(answer_write_options, [max_depth(30)]).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  To write all answers in one query          %
%                  So that we do not need to hit ";".         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Output all answers at once, without hitting ";"
writeall(Q) :- forall(Q,writeln(Q)).


% to limit the number of solutions to be written out
% more advanced than "writeall\1"

% Main predicate
write_max_solutions(Q, Max) :-
    write_solutions_up_to(Q, Max, 1).

% Recursive predicate to stop after Max solutions
write_solutions_up_to(_, Max, Counter) :-
    Counter > Max, 
    !.  % Cut to ensure it doesn't backtrack into infinite solutions

write_solutions_up_to(Q, Max, Counter) :-
    once(Q),  % Retrieves only one solution without backtracking for more
    writeln(Q),
    NewCounter is Counter + 1,
    write_solutions_up_to(Q, Max, NewCounter).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      meta-interpreters                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g(G) :- call(G).

% Clause lookup with defaulty betterment
mi_clause(G, G) :-
    predicate_property(G, built_in), !.
mi_clause(G, Body) :-
    clause(G, B),
    defaulty_better(B, Body).


defaulty_better(true, true).
defaulty_better((A,B), (BA,BB)) :-
        defaulty_better(A, BA),
        defaulty_better(B, BB).
defaulty_better(G, g(G)) :-
        G \= true,
        G \= (_,_).

% Define the operator for proofs
:- op(750, xfy, =>).

% Proof tree generation
mi_tree(true, true).
mi_tree((A,B), (TA,TB)) :-
        mi_tree(A, TA),
        mi_tree(B, TB).
mi_tree(G, builtin(G)) :- predicate_property(G, built_in), !, call(G).
mi_tree(g(G), TBody => G) :-
        mi_clause(G, Body),
        mi_tree(Body, TBody).

% Depth-limited meta-interpreter with proof tree generation
mi_limit(true, true, N, N).
mi_limit((A,B), (TA,TB), N0, N) :-
        mi_limit(A, TA, N0, N1),
        mi_limit(B, TB, N1, N).
% mi_limit(G, builtin(G), N, N) :- predicate_property(G, built_in), !, call(G). % **This line seems to make iterative-deepening search not work.**
mi_limit(g(G), TBody => G, N0, N) :-
        N0 #> 0,
        N1 #= N0 - 1,
        mi_clause(G, Body),
        mi_limit(Body, TBody, N1, N).

% Iterative deepening with proof tree generation
mi_id(Goal, Proof) :-
        length(_, N),
        mi_limit(Goal, Proof, N, _).

% Iterative deepening with maximum depth with proof tree generation
mi_id_limit(Goal, Proof, MaxDepth) :-
        between(1, MaxDepth, N),
        mi_limit(Goal, Proof, N, _).

% Sample Usage
% mi_id(some_goal, Proof)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mi_clause(G, Body) :-
%         clause(G, B),
%         defaulty_better(B, Body).

% defaulty_better(true, true).
% defaulty_better((A,B), (BA,BB)) :-
%         defaulty_better(A, BA),
%         defaulty_better(B, BB).
% defaulty_better(G, g(G)) :-
%         G \= true,
%         G \= (_,_).


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :- op(750, xfy, =>).

% mi_tree(true, true).
% mi_tree((A,B), (TA,TB)) :-
%         mi_tree(A, TA),
%         mi_tree(B, TB).
% mi_tree(g(G), TBody => G) :-
%         mi_clause(G, Body),
%         mi_tree(Body, TBody).


% Below Code limits the depth of the search tree. From https://www.metalevel.at/acomip/
mi_limit_no_proof(Goal, Max) :-
        mi_limit_no_proof(Goal, Max, _).

mi_limit_no_proof(true, N, N).
mi_limit_no_proof((A,B), N0, N) :-
        mi_limit_no_proof(A, N0, N1),
        mi_limit_no_proof(B, N1, N).
mi_limit_no_proof(g(G), N0, N) :-
        N0 #> 0,
        N1 #= N0 - 1,
        mi_clause(G, Body),
        mi_limit_no_proof(Body, N1, N).

% Below is iterative deepening search, no proof tree generation
mi_id_limit_no_proof(Goal, MaxDepth) :-
        between(1, MaxDepth, N),
        mi_limit_no_proof(Goal, N).

% How to use this?
% mi_tree(g(Goal), T)
% mi_limit_no_proof(g(Goal), 3).
% mi_id(g(Goal)).