/* Context */

% statement-1: Tumpuses are kind.
kind(X) :- tumpus(X).

% statement-2: Every tumpus is an impus.
impus(X) :- tumpus(X).

% statement-3: Impuses are not dull.
not(dull(X)) :- impus(X).

% statement-4: Impuses are jompuses.
jompus(X) :- impus(X).

% statement-5: Jompuses are not large.
not(large(X)) :- jompus(X).

% statement-6: Jompuses are zumpuses.
zumpus(X) :- jompus(X).

% statement-7: Every zumpus is happy.
happy(X) :- zumpus(X).

% statement-8: Zumpuses are wumpuses.
wumpus(X) :- zumpus(X).

% statement-9: Every dumpus is not fruity.
not(fruity(X)) :- dumpus(X).

% statement-10: Each wumpus is sweet.
sweet(X) :- wumpus(X).

% statement-11: Wumpuses are yumpuses.
yumpus(X) :- wumpus(X).

% statement-12: Yumpuses are orange.
orange(X) :- yumpus(X).

% statement-13: Every yumpus is a numpus.
numpus(X) :- yumpus(X).

% statement-14: Numpuses are transparent.
transparent(X) :- numpus(X).

% statement-15: Each numpus is a vumpus.
vumpus(X) :- numpus(X).

% statement-16: Vumpuses are fruity.
fruity(X) :- vumpus(X).

% statement-17: Every vumpus is a rompus.
rompus(X) :- vumpus(X).

% statement-18: Fae is a zumpus.
zumpus(fae).

/* Question */

% Fae is not fruity.
not(fruity(fae)).