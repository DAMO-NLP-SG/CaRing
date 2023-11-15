/* Context */

% facts
total_score(30).
triple_word_multiplier(3).
first_and_third_letter_value(1).

% calculate the value of the middle letter
middle_letter_value(Value) :-
    triple_word_multiplier(Multiplier),
    first_and_third_letter_value(FirstThirdValue),
    total_score(Total),
    Value is (Total / Multiplier) - 2 * FirstThirdValue.

/* Query */
solve(MiddleValue) :- middle_letter_value(MiddleValue).