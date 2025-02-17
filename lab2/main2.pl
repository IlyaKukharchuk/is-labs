% Part-Whole (Состав)
part_of(engine, car).
part_of(transmission, car).
part_of(brake_system, car).
part_of(suspension, car).
part_of(electrical_system, car).
part_of(battery, electrical_system).
part_of(alternator, electrical_system).
part_of(fuel_system, car).
part_of(cooling_system, car).
part_of(exhaust_system, car).
part_of(sensors, car).
part_of(fuel_system, engine).  
part_of(cooling_system, engine).  

% Functional Dependency (Функциональная зависимость)
depends_on(engine, fuel_system).
depends_on(engine, cooling_system).
depends_on(electrical_system, battery).
depends_on(electrical_system, alternator).
depends_on(fuel_system, sensors).

% Cause-Effect (Причина-следствие)
causes(battery_failure, starting_issues).
causes(alternator_failure, battery_drain).
causes(sensor_failure, unstable_engine_operation).
causes(cooling_system_leak, engine_overheating).
causes(fuel_system_clog, poor_starting).

% Queries (Запросы)
% 1. Какие компоненты входят в состав двигателя?
engine_components(X) :- part_of(X, engine).

% 2. Какие неисправности могут быть вызваны неисправностью аккумулятора?
battery_failure_issues(X) :- causes(battery_failure, X).

% 3. Какие датчики влияют на работу топливной системы?
fuel_system_sensors(X) :- depends_on(fuel_system, X).

% 4. Какие системы зависят от электрической системы?
systems_dependent_on_electrical(X) :- depends_on(X, electrical_system).

% 5. Какие компоненты могут вызвать перегрев двигателя?
causes_of_overheating(X) :- causes(X, engine_overheating).
