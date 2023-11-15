/* Context */

% sent-1: Tina makes $18.00 an hour.
wage(18.00).

% sent-2: If she works more than 8 hours per shift,
% sent-3: she is eligible for overtime,
% sent-4: which is paid by your hourly wage + 1/2 your hourly wage.
overtime_wage(W) :- 
    wage(W1), 
    W is 1.5 * W1.

% earnings without overtime for 1 day
regular_earning_for_day(E) :- 
    wage(W),
    E is 8 * W.

% sent-5: If she works 10 hours every day for 5 days,
overtime_hours(H) :-
    H is 10 - 8.

% overtime earnings for 1 day
overtime_earning_for_day(E) :- 
    overtime_hours(H),
    overtime_wage(W),
    E is H * W.

% total earnings for 1 day
total_earning_for_day(Total) :-
    regular_earning_for_day(Regular),
    overtime_earning_for_day(Overtime),
    Total is Regular + Overtime.

% total earnings for 5 days
total_earning_for_5_days(Total) :-
    total_earning_for_day(OneDay),
    Total is 5 * OneDay.

/* Query */
solve(Total) :- total_earning_for_5_days(Total).