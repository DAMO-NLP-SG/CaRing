/* Triples */

eats(cow, squirrel).
eats(cow, tiger).
cold(cow).
not(need(cow, mouse)).
needs(cow, squirrel).
chases(mouse, cow).
chases(mouse, squirrel).
not(chase(mouse, tiger)).
eats(mouse, tiger).
not(big(mouse)).
eats(squirrel, cow).
cold(squirrel).
kind(squirrel).
not(need(squirrel, mouse)).
chases(tiger, mouse).
not(need(tiger, mouse)).

/* Rules */

% rule-1: If something eats the cow then it is cold.
eats(X, cow) :- cold(X).

% rule-2: All red things are big.
big(X) :- red(X).

% rule-3: If something needs the tiger and it chases the mouse then it chases the cow.
chases(X, cow) :- needs(X, tiger), chases(X, mouse).

% rule-4: If something is big then it eats the cow.
eats(X, cow) :- big(X).

% rule-5: If something eats the squirrel then the squirrel needs the tiger.
needs(squirrel, tiger) :- eats(_, squirrel).

% rule-6: If something needs the tiger then the tiger is red.
red(tiger) :- needs(_, tiger).

% rule-7: If the cow is cold then the cow eats the squirrel.
eats(cow, squirrel) :- cold(cow).

% rule-8: If something needs the cow and it is not cold then it does not eat the tiger.
not(eats(X, tiger)) :- needs(X, cow), not(cold(X)).

% rule-9: If something chases the tiger then it eats the tiger and the mouse needs the tiger.
eats(X, tiger) :- chases(X, tiger).
needs(mouse, tiger) :- chases(_, tiger).

/* Question */

not(kind(mouse)). % statement: The mouse is not kind.
