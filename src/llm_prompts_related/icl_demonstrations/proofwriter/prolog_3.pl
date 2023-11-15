/* Triples */

kind(anne).
not(big(charlie)).
not(green(charlie)).
white(charlie).
big(erin).
green(erin).
white(erin).
green(fiona).
kind(fiona).
quiet(fiona).
red(fiona).
white(fiona).

/* Rules */

% rule-1: If Erin is big and Erin is red then Erin is kind.
kind(erin) :- big(erin), red(erin).

% rule-2: All rough things are green.
green(X) :- rough(X).

% rule-3: If something is kind then it is green.
green(X) :- kind(X).

% rule-4: Quiet, green things are big.
big(X) :- quiet(X), green(X).

% rule-5: If something is rough and green then it is red.
red(X) :- rough(X), green(X).

% rule-6: If something is green then it is rough.
rough(X) :- green(X).

% rule-7: If Erin is red then Erin is green.
green(erin) :- red(erin).

% rule-8: All red, rough things are quiet.
quiet(X) :- red(X), rough(X).

% rule-9: If something is quiet and not red then it is not white.
not(white(X)) :- quiet(X), not(red(X)).

/* Question */

not(rough(erin)). % statement: Erin is not rough.
