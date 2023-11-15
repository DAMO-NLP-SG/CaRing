/* Context */

% facts
goal_amount(150).
amount_from_three_families_each(10).
amount_from_fifteen_families_each(5).
number_of_three_families(3).
number_of_fifteen_families(15).

% calculate the total amount they have collected so far
amount_collected(Total) :-
    amount_from_three_families_each(ThreeFamiliesEach),
    amount_from_fifteen_families_each(FifteenFamiliesEach),
    number_of_three_families(NoThreeFamilies),
    number_of_fifteen_families(NoFifteenFamilies),
    Total is ThreeFamiliesEach * NoThreeFamilies + FifteenFamiliesEach * NoFifteenFamilies.

% calculate how much more they need to reach their goal
amount_needed(Needed) :-
    goal_amount(Goal),
    amount_collected(Collected),
    Needed is Goal - Collected.

/* Query */
solve(AmountNeeded) :- amount_needed(AmountNeeded).