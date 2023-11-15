/* Context */

% facts
first_hour(15).
next_two_hours_each(35).
fourth_hour_collected(50).
fourth_hour_given(15).

% calculate total coins after fourth hour
total_coins_after_fourth_hour(Total) :-
    first_hour(First),
    next_two_hours_each(NextTwo),
    fourth_hour_collected(FourthCollected),
    fourth_hour_given(FourthGiven),
    Total is First + 2 * NextTwo + FourthCollected - FourthGiven.

/* Query */
solve(TotalCoins) :- total_coins_after_fourth_hour(TotalCoins).