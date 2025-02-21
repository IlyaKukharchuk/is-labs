% Frame structure
frame(component, [
    name: '',
    type: '',
    status: '',
    diagnosis: diagnosis,
    check_status: check_status,
    repair: repair
]).

% Middle level frames
frame(engine, [
    name: 'Engine',
    type: 'component',
    status: 'unknown',
    diagnosis: engine_diagnosis,
    check_status: engine_check_status,
    repair: engine_repair
]).

frame(electrical_system, [
    name: 'Electrical System',
    type: 'component',
    status: 'unknown',
    diagnosis: electrical_diagnosis,
    check_status: electrical_check_status,
    repair: electrical_repair
]).

frame(fuel_system, [
    name: 'Fuel System',
    type: 'component',
    status: 'unknown',
    diagnosis: fuel_diagnosis,
    check_status: fuel_check_status,
    repair: fuel_repair
]).

% Lower level frames with multiple inheritance
frame(battery, [
    name: 'Battery',
    type: 'electrical_system',
    status: 'unknown',
    diagnosis: battery_diagnosis,
    check_status: battery_check_status,
    repair: battery_repair
]).

frame(alternator, [
    name: 'Alternator',
    type: 'electrical_system',
    status: 'unknown',
    diagnosis: alternator_diagnosis,
    check_status: alternator_check_status,
    repair: alternator_repair
]).

frame(fuel_pump, [
    name: 'Fuel Pump',
    type: 'fuel_system',
    status: 'unknown',
    diagnosis: fuel_pump_diagnosis,
    check_status: fuel_pump_check_status,
    repair: fuel_pump_repair
]).

frame(spark_plug, [
    name: 'Spark Plug',
    type: 'fuel_system',
    status: 'unknown',
    diagnosis: spark_plug_diagnosis,
    check_status: spark_plug_check_status,
    repair: spark_plug_repair
]).

frame(cooling_system, [
    name: 'Cooling System',
    type: 'engine',
    status: 'unknown',
    diagnosis: cooling_system_diagnosis,
    check_status: cooling_system_check_status,
    repair: cooling_system_repair
]).

% Diagnosis procedures
diagnosis(engine, Issue) :- Issue = 'Possible overheating'.
diagnosis(electrical_system, Issue) :- Issue = 'Possible battery drain'.
diagnosis(fuel_system, Issue) :- Issue = 'Possible fuel leak'.

% Check status procedures
check_status(engine, Status) :- Status = 'Engine is running'.
check_status(electrical_system, Status) :- Status = 'Electrical system is operational'.
check_status(fuel_system, Status) :- Status = 'Fuel system is operational'.

% Repair procedures
repair(engine, Action) :- Action = 'Check coolant level'.
repair(electrical_system, Action) :- Action = 'Check battery connections'.
repair(fuel_system, Action) :- Action = 'Check for fuel leaks'.

% Example queries
% 1. Diagnose engine issues
% ?- diagnosis(engine, Issue).

% 2. Check the status of the electrical system
% ?- check_status(electrical_system, Status).

% 3. Get repair steps for the fuel system
% ?- repair(fuel_system, Action).

% 4. Diagnose battery issues
% ?- diagnosis(battery, Issue).

% 5. Check the status of the cooling system
% ?- check_status(cooling_system, Status).
