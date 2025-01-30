# Код реализует симуляцию игры "Охота на оленя" (Stag Hunt) на языке Python.
# Автор: Кухарчук Илья Николаевич, группа АС-63, 2025 год.

# Постановка задачи:
# Игра "Охота на оленя" — это классическая задача теории игр, в которой два игрока могут выбрать либо сотрудничать (охотиться на оленя),
# либо предать (охотиться на зайца). Охота на оленя приносит максимальный выигрыш, но только если оба игрока выбирают сотрудничество.
# Если один из игроков выбирает охоту на зайца, он получает меньший, но гарантированный выигрыш, а другой игрок, выбравший охоту на оленя,
# получает нулевой выигрыш. Если оба игрока выбирают охоту на зайца, они получают одинаковый, но меньший выигрыш.

# Цель задачи:
# Реализовать симуляцию игры "Охота на оленя" с двумя стратегиями:
# 1. Игрок 1 всегда выбирает охоту на оленя.
# 2. Игрок 2 выбирает действие случайным образом.
# Провести симуляцию на 1000 раундов и вывести итоговые результаты.

# Использованные источники:
# 1. Википедия: "Список игр теории игр" — https://ru.wikipedia.org/wiki/Список_игр_теории_игр
# 2. Разбор решения игры "Камень-ножницы-бумага" на Python — https://habr.com/ru/articles/713120/
# 3. Теория игр и её применение в жизни — https://habr.com/ru/articles/502384/

import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Определим возможные действия игроков
ACTIONS = ['Олень', 'Заяц']

# Определим матрицу выигрышей
PAYOFF_MATRIX = {
    ('Олень', 'Олень'): (4, 4),
    ('Олень', 'Заяц'): (0, 3),
    ('Заяц', 'Олень'): (3, 0),
    ('Заяц', 'Заяц'): (3, 3),
}

# Стратегия игрока 1: случайный выбор
def player1_strategy():
    return random.choice(ACTIONS)

# Нейронная сеть для игрока 2
class Player2Net(nn.Module):
    def __init__(self):
        super(Player2Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Стратегия игрока 2: использование нейронной сети
def player2_strategy(model, optimizer, criterion, action1):
    action1_tensor = torch.tensor([ACTIONS.index(action1)], dtype=torch.float32)
    output = model(action1_tensor)
    action2_index = torch.argmax(output).item()
    action2 = ACTIONS[action2_index]

    # Обучение нейронной сети
    target = torch.tensor([ACTIONS.index(action2)], dtype=torch.float32)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return action2

# Функция для симуляции одного раунда игры
def play_round(player1_strategy, player2_strategy, model, optimizer, criterion):
    action1 = player1_strategy()
    action2 = player2_strategy(model, optimizer, criterion, action1)
    payoff = PAYOFF_MATRIX[(action1, action2)]
    return action1, action2, payoff

# Функция для симуляции нескольких раундов игры
def simulate_game(player1_strategy, player2_strategy, model, optimizer, criterion, rounds=1000, epochs=15):
    total_payoff1 = 0
    total_payoff2 = 0
    win_counts = {'Игрок 1 (рандом)': 0, 'Игрок 2 (машинное обучение)': 0, 'Ничья': 0}

    for epoch in range(epochs):
        for _ in range(rounds):
            action1, action2, payoff = play_round(player1_strategy, player2_strategy, model, optimizer, criterion)
            total_payoff1 += payoff[0]
            total_payoff2 += payoff[1]

            if payoff[0] > payoff[1]:
                win_counts['Игрок 1 (рандом)'] += 1
            elif payoff[0] < payoff[1]:
                win_counts['Игрок 2 (машинное обучение)'] += 1
            else:
                win_counts['Ничья'] += 1

    return total_payoff1, total_payoff2, win_counts

# Инициализация нейронной сети
model = Player2Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Запуск симуляции
total_payoff1, total_payoff2, win_counts = simulate_game(player1_strategy, player2_strategy, model, optimizer, criterion, rounds=1000, epochs=20)
print(f"Игрок 1 (рандом): {total_payoff1} очков")
print(f"Игрок 2 (машинное обучение): {total_payoff2} очков")

# График статистики побед
labels = win_counts.keys()
sizes = win_counts.values()

fig, ax = plt.subplots()
ax.bar(labels, sizes)
ax.set_title('Статистика побед в раундах')
ax.set_xlabel('Игрок')
ax.set_ylabel('Количество побед')
plt.show()


# Инструкция по компиляции и запуску кода:
# 1. Убедитесь, что на вашем компьютере установлен Python (версия 3.6 или выше).
# 2. Скопируйте код в файл с расширением .py, например, stag_hunt.py.
# 3. Откройте терминал или командную строку.
# 4. Перейдите в директорию, где находится файл stag_hunt.py.
# 5. Запустите код командой: python stag_hunt.py
# 6. Результаты симуляции будут выведены в терминал.