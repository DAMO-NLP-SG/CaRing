/* Context */

% facts
pieces_per_mushroom(4).
kenny_pieces(38).
karla_pieces(42).
remaining_pieces(8).

% calculate total mushrooms cut at the beginning
total_mushrooms(Mushrooms) :-
    pieces_per_mushroom(PiecesPerMushroom),
    kenny_pieces(Kenny),
    karla_pieces(Karla),
    remaining_pieces(Remaining),
    TotalPieces is Kenny + Karla + Remaining,
    Mushrooms is TotalPieces / PiecesPerMushroom.

/* Query */
solve(MushroomsCut) :- total_mushrooms(MushroomsCut).