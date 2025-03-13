import networkx as nx
import matplotlib.pyplot as plt

# Создаем направленный граф
G = nx.DiGraph()

# Добавляем узлы и ребра для части "Part-Whole"
edges_part_of = [
    ('engine', 'car'),
    ('transmission', 'car'),
    ('brake_system', 'car'),
    ('suspension', 'car'),
    ('electrical_system', 'car'),
    ('battery', 'electrical_system'),
    ('alternator', 'electrical_system'),
    ('fuel_system', 'car'),
    ('cooling_system', 'car'),
    ('exhaust_system', 'car'),
    ('sensors', 'car'),
    ('fuel_system', 'engine'),
    ('cooling_system', 'engine')
]

# Добавляем узлы и ребра для части "Functional Dependency"
edges_depends_on = [
    ('engine', 'fuel_system'),
    ('engine', 'cooling_system'),
    ('electrical_system', 'battery'),
    ('electrical_system', 'alternator'),
    ('fuel_system', 'sensors')
]

# Добавляем узлы и ребра для части "Cause-Effect"
edges_causes = [
    ('battery_failure', 'starting_issues'),
    ('alternator_failure', 'battery_drain'),
    ('sensor_failure', 'unstable_engine_operation'),
    ('cooling_system_leak', 'engine_overheating'),
    ('fuel_system_clog', 'poor_starting')
]

# Добавляем ребра в граф
G.add_edges_from(edges_part_of, label='part_of')
G.add_edges_from(edges_depends_on, label='depends_on')
G.add_edges_from(edges_causes, label='causes')

# Используем алгоритм расположения узлов
pos = nx.kamada_kawai_layout(G)  # Используем алгоритм Камады-Каваи для улучшения расположения

# Визуализация графа
plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8, font_weight='bold', arrowsize=15)

# Добавляем метки ребер с небольшим смещением
edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
for (u, v), label in edge_labels.items():
    x_mid = (pos[u][0] + pos[v][0]) / 2
    y_mid = (pos[u][1] + pos[v][1]) / 2
    plt.text(x_mid, y_mid + 0.1, label, fontsize=8, color='red', ha='center', va='center')

plt.title("Semantic Network for Car Components and Relationships")
plt.show()
