/* Context */
% facts
total_distance(30).
jesse_first_three_days_avg(2/3).
jesse_day_four(10).
mia_first_four_days_avg(3).

% calculate the total distance Jesse ran over the first four days
jesse_first_four_days_total(Distance) :-
    jesse_first_three_days_avg(DayAvg),
    jesse_day_four(DayFour),
    Distance is 3 * DayAvg + DayFour.

% calculate the total distance Mia ran over the first four days
mia_first_four_days_total(Distance) :-
    mia_first_four_days_avg(DayAvg),
    Distance is 4 * DayAvg.

% calculate the average miles they have to run over the final three days
remaining_avg(Person, Avg) :-
    (Person = jesse -> jesse_first_four_days_total(Distance);
    Person = mia -> mia_first_four_days_total(Distance)),
    total_distance(Total),
    Remaining is Total - Distance,
    Avg is Remaining / 3.

% determine the average of their averages over the final three days
average_of_averages(Result) :-
    remaining_avg(jesse, JesseAvg),
    remaining_avg(mia, MiaAvg),
    Result is (JesseAvg + MiaAvg) / 2.

/* Query */
solve(Average) :- average_of_averages(Average).