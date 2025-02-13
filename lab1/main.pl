% Facts
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

% Rules
problem(low_oil_level) :- symptom(engine_overheating), symptom(low_oil_level).
problem(faulty_spark_plugs) :- symptom(engine_misfiring), symptom(old_spark_plugs).
problem(faulty_fuel_pump) :- symptom(engine_not_starting), symptom(fuel_pump_not_working).
problem(clogged_air_filter) :- symptom(engine_losing_power), symptom(clogged_air_filter).
problem(faulty_oxygen_sensor) :- symptom(engine_running_rich), symptom(faulty_oxygen_sensor).
problem(low_coolant_level) :- symptom(engine_overheating), symptom(low_coolant_level).
problem(low_fuel_octane) :- symptom(engine_knocking), symptom(low_fuel_octane).
problem(faulty_idle_air_control_valve) :- symptom(engine_stalling), symptom(faulty_idle_air_control_valve).
problem(clogged_fuel_injectors) :- symptom(engine_emitting_black_smoke), symptom(clogged_fuel_injectors).
problem(worn_engine_mounts) :- symptom(engine_vibrating_excessively), symptom(worn_engine_mounts).