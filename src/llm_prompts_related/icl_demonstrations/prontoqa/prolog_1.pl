/* Context */

% statement-1: Wumpuses are large.
large(X) :- wumpus(X).

% statement-2: Wumpuses are numpuses.
numpus(X) :- wumpus(X).

% statement-3: Every numpus is metallic.
metallic(X) :- numpus(X).

% statement-4: Numpuses are yumpuses.
yumpus(X) :- numpus(X).

% statement-5: Every yumpus is bright.
bright(X) :- yumpus(X).

% statement-6: Every yumpus is a jompus.
jompus(X) :- yumpus(X).

% statement-7: Jompuses are not bitter.
not(bitter(X)) :- jompus(X).

% statement-8: Jompuses are zumpuses.
zumpus(X) :- jompus(X).

% statement-9: Every zumpus is transparent.
transparent(X) :- zumpus(X).

% statement-10: Zumpuses are rompuses.
rompus(X) :- zumpus(X).

% statement-11: Each rompus is earthy.
earthy(X) :- rompus(X).

% statement-12: Rompuses are impuses.
impus(X) :- rompus(X).

% statement-13: Each impus is kind.
kind(X) :- impus(X).

% statement-14: Every impus is a dumpus.
dumpus(X) :- impus(X).

% statement-15: Dumpuses are not hot.
not(hot(X)) :- dumpus(X).

% statement-16: Dumpuses are vumpuses.
vumpus(X) :- dumpus(X).

% statement-17: Each tumpus is not transparent.
not(transparent(X)) :- tumpus(X).

% statement-18: Wren is a wumpus.
wumpus(wren).

/* Question */

% Wren is transparent.
transparent(wren).