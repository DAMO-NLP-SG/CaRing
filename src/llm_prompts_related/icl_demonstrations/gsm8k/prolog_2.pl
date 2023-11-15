/* Context */

% facts
sentences_per_minute(6).
typing_sessions([20, 15, 18]).
erased_sentences(40).
total_end_sentences(536).

% calculate the number of sentences typed in a session
sentences_typed(SessionMinutes, Typed) :-
    sentences_per_minute(SPM),
    Typed is SPM * SessionMinutes.

% calculate the total number of sentences typed across all sessions today
total_sentences_typed_today(Total) :-
    typing_sessions(Sessions),
    maplist(sentences_typed, Sessions, TypedPerSession),
    sum_list(TypedPerSession, TotalTyped),
    erased_sentences(Erased),
    Total is TotalTyped - Erased.

% calculate how many sentences she started with today
start_sentences(TodayStart) :-
    total_end_sentences(EndToday),
    total_sentences_typed_today(TodayTyped),
    TodayStart is EndToday - TodayTyped.

/* Query */
solve(StartSentences) :- start_sentences(StartSentences).