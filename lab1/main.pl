% Определение симптомов
symptom(engine_overheating).
symptom(low_oil_level).
symptom(engine_misfiring).
symptom(old_spark_plugs).
symptom(engine_not_starting).
symptom(fuel_pump_not_working).
symptom(engine_losing_power).
symptom(clogged_air_filter).
symptom(engine_running_rich).
symptom(faulty_oxygen_sensor).
symptom(low_coolant_level).
symptom(engine_knocking).
symptom(low_fuel_octane).
symptom(engine_stalling).
symptom(faulty_idle_air_control_valve).
symptom(engine_emitting_black_smoke).
symptom(clogged_fuel_injectors).
symptom(engine_vibrating_excessively).
symptom(worn_engine_mounts).

% Определение проблем и их симптомов
problem(low_oil_level, [engine_overheating, low_oil_level]).
problem(faulty_spark_plugs, [engine_misfiring, old_spark_plugs]).
problem(faulty_fuel_pump, [engine_not_starting, fuel_pump_not_working]).
problem(clogged_air_filter, [engine_losing_power, clogged_air_filter]).
problem(faulty_oxygen_sensor, [engine_running_rich, faulty_oxygen_sensor]).
problem(low_coolant_level, [engine_overheating, low_coolant_level]).
problem(low_fuel_octane, [engine_knocking, low_fuel_octane]).
problem(faulty_idle_air_control_valve, [engine_stalling, faulty_idle_air_control_valve]).
problem(clogged_fuel_injectors, [engine_emitting_black_smoke, clogged_fuel_injectors]).
problem(worn_engine_mounts, [engine_vibrating_excessively, worn_engine_mounts]).

% Симптомы у разных автомобилей
car(car1, [engine_overheating, low_oil_level]).
car(car2, [engine_misfiring, old_spark_plugs]).
car(car3, [engine_not_starting, fuel_pump_not_working]).
car(car4, [engine_losing_power, clogged_air_filter]).
car(car5, [engine_running_rich, faulty_oxygen_sensor]).
car(car6, [engine_overheating, low_coolant_level]).
car(car7, [engine_knocking, low_fuel_octane]).
car(car8, [engine_stalling, faulty_idle_air_control_valve]).
car(car9, [engine_emitting_black_smoke, clogged_fuel_injectors]).
car(car10, [engine_vibrating_excessively, worn_engine_mounts]).

% Диагностика проблемы автомобиля
diagnose_car(Car, Problem) :-
    car(Car, Symptoms),
    problem(Problem, ProblemSymptoms),
    forall(member(Symptom, ProblemSymptoms), member(Symptom, Symptoms)).

% Рекомендация проблемы по симптомам
recommend_problem(Symptoms, Problem) :-
    problem(Problem, ProblemSymptoms),
    forall(member(Symptom, ProblemSymptoms), member(Symptom, Symptoms)).

% Диагностика по симптомам
diagnose_by_symptoms(Symptoms, Problem) :-
    problem(Problem, ProblemSymptoms),
    forall(member(Symptom, ProblemSymptoms), member(Symptom, Symptoms)).

% Автомобиль с определенным симптомом
car_with_symptom(Symptom, Car) :-
    car(Car, Symptoms),
    member(Symptom, Symptoms).

% Проблема с определенным симптомом
problem_with_symptom(Symptom, Problem) :-
    problem(Problem, Symptoms),
    member(Symptom, Symptoms).

% Симптомы для определенной проблемы
problem_symptoms(Problem, Symptoms) :-
    problem(Problem, Symptoms).

% Проверка, есть ли у автомобиля определенная проблема
has_problem(Car, Problem) :-
    diagnose_car(Car, Problem).

% Частичное совпадение симптомов автомобиля с проблемой
problem_partially_match(Car, Problem) :-
    car(Car, Symptoms),
    problem(Problem, ProblemSymptoms),
    subset(Symptoms, ProblemSymptoms).

% Проблемы с определенным симптомом
problems_with_symptom(Symptom, Problem) :-
    problem(Problem, Symptoms),
    member(Symptom, Symptoms).

% Автомобили с определенными симптомами
cars_with_symptoms(Symptoms, Car) :-
    car(Car, CarSymptoms),
    subset(Symptoms, CarSymptoms).

% Симптомы для определенной проблемы
symptoms_for_problem(Problem, Symptoms) :-
    problem(Problem, Symptoms).

% Частичная диагностика автомобиля
diagnose_car_partial(Car, Problem) :-
    car(Car, Symptoms),
    problem(Problem, ProblemSymptoms),
    intersection(Symptoms, ProblemSymptoms, CommonSymptoms),
    length(CommonSymptoms, L),
    L > 0.

% Примеры запросов
% diagnose_car(car1, Problem).
% car_with_symptom(engine_overheating, Car).
% problem_with_symptom(engine_overheating, Problem).
% problem_symptoms(low_oil_level, Symptoms).
% has_problem(car1, low_oil_level).
% problem_partially_match(car1, Problem).
% problems_with_symptom(engine_overheating, Problem).
% cars_with_symptoms([engine_overheating, low_oil_level], Car).
% symptoms_for_problem(low_oil_level, Symptoms).
% diagnose_car_partial(car1, low_oil_level).