/* Context */

% facts
carwash_earnings(100).
carwash_percentage(0.9).
bake_sale_earnings(80).
bake_sale_percentage(0.75).
lawn_earnings(50).
lawn_percentage(1).

% calculate the total donation by Hank
total_donation(Total) :-
    carwash_earnings(Carwash),
    carwash_percentage(CarwashPerc),
    bake_sale_earnings(BakeSale),
    bake_sale_percentage(BakeSalePerc),
    lawn_earnings(Lawn),
    lawn_percentage(LawnPerc),
    Total is Carwash * CarwashPerc + BakeSale * BakeSalePerc + Lawn * LawnPerc.

/* Query */
solve(TotalDonated) :- total_donation(TotalDonated).