/* Context */

% facts
venue_cost(10000).
cost_per_guest(500).
johns_guests(50).
johns_wife_more_percentage(0.6).

% calculate the total wedding cost
total_wedding_cost(Total) :-
    venue_cost(Venue),
    cost_per_guest(PerGuest),
    johns_guests(JohnsGuests),
    johns_wife_more_percentage(Percentage),
    WifeGuests is JohnsGuests + Percentage * JohnsGuests,
    Total is Venue + WifeGuests * PerGuest.

/* Query */
solve(TotalCost) :- total_wedding_cost(TotalCost).