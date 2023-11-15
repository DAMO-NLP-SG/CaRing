/* Context */

% facts
batch_size_gallons(1.5).
ounces_per_gallon(128).
consumption_rate(96).
days_for_consumption_rate(2).
time_to_make_batch(20).

% convert batch size from gallons to ounces
batch_size_ounces(Size) :-
    batch_size_gallons(Gallons),
    ounces_per_gallon(OuncePerGallon),
    Size is Gallons * OuncePerGallon.

% calculate the amount of coffee consumed over 24 days
consumption_24_days(TotalOunces) :-
    consumption_rate(Per2Days),
    TotalOunces is (24 / 2) * Per2Days.

% calculate how many batches Jack needs to make over 24 days
batches_needed(Batches) :-
    consumption_24_days(Consumed),
    batch_size_ounces(BatchSize),
    Batches is ceil(Consumed / BatchSize).

% calculate how long Jack spends making coffee over 24 days
time_spent_making_coffee(Time) :-
    batches_needed(Batches),
    time_to_make_batch(PerBatch),
    Time is Batches * PerBatch.

/* Query */
solve(TimeSpent) :- time_spent_making_coffee(TimeSpent).