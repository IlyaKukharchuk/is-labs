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

# Импорт необходимых библиотек
import random  # Для генерации случайных чисел (используется в стратегии игрока 1)
import torch  # Основная библиотека для работы с нейронными сетями
import torch.nn as nn  # Модуль для создания нейронных сетей
import torch.optim as optim  # Модуль для оптимизации нейронных сетей
import matplotlib.pyplot as plt  # Для визуализации данных (построения графиков)

# Определим возможные действия игроков
ACTIONS = ['Олень', 'Заяц']  # Два возможных действия, которые могут выбрать игроки

# Функция загрузки матрицы выигрышей
def read_payoff_matrix(file_path):
    payoff_matrix = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                key_part, value_part = line.split(':')
                actions = tuple(key_part.split(','))
                values = tuple(map(int, value_part.split(',')))
                payoff_matrix[actions] = values
            except (ValueError, IndexError) as e:
                print(f"Ошибка в строке '{line}': {e}")
    return payoff_matrix

# Загрузка матрицы
PAYOFF_MATRIX = read_payoff_matrix('payoff_matrix.txt')

# Стратегия игрока 1: случайный выбор
def player1_strategy():
    return random.choice(ACTIONS)  # Игрок 1 случайным образом выбирает одно из действий ("Олень" или "Заяц")

# Нейронная сеть для игрока 2
class Player2Net(nn.Module):
    def __init__(self):
        super(Player2Net, self).__init__()  # Инициализация родительского класса nn.Module
        self.fc1 = nn.Linear(1, 10)  # Первый полносвязный слой: входной размер 1, выходной размер 10
        self.fc2 = nn.Linear(10, 2)  # Второй полносвязный слой: входной размер 10, выходной размер 2

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применяем функцию активации ReLU к выходу первого слоя  если результат меньше нуля, то он становится нулём. Формула: ReLU(x) = max(0, x).
        x = self.fc2(x)  # Пропускаем данные через второй слой
        return x  # Возвращаем выход нейронной сети

# Стратегия игрока 2: использование нейронной сети
def player2_strategy(model, optimizer, criterion, action1):
    # Преобразуем действие игрока 1 в тензор (число, представляющее индекс действия)
    action1_tensor = torch.tensor([ACTIONS.index(action1)], dtype=torch.float32)
    
    # Получаем выход нейронной сети (предсказание)
    output = model(action1_tensor)
    
    # Выбираем действие с максимальным значением (предсказанное действие)
    action2_index = torch.argmax(output).item()
    action2 = ACTIONS[action2_index]  # Преобразуем индекс обратно в действие

    # Обучение нейронной сети
    target = torch.tensor([ACTIONS.index(action2)], dtype=torch.float32)  # Целевое значение (то же действие)
    loss = criterion(output, target)  # Вычисляем ошибку (разницу между предсказанием и целевым значением)
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()  # Вычисляем градиенты
    optimizer.step()  # Обновляем веса нейронной сети

    return action2  # Возвращаем выбранное действие

# Функция для симуляции одного раунда игры
def play_round(player1_strategy, player2_strategy, model, optimizer, criterion):
    action1 = player1_strategy()  # Игрок 1 выбирает действие случайным образом
    action2 = player2_strategy(model, optimizer, criterion, action1)  # Игрок 2 выбирает действие с помощью нейронной сети
    payoff = PAYOFF_MATRIX[(action1, action2)]  # Получаем выигрыши для обоих игроков
    return action1, action2, payoff  # Возвращаем действия и выигрыши

# Функция для симуляции нескольких раундов игры
def simulate_game(player1_strategy, player2_strategy, model, optimizer, criterion, rounds=100, epochs=10):
    total_payoff1 = 0  # Общий выигрыш игрока 1
    total_payoff2 = 0  # Общий выигрыш игрока 2
    win_counts = {'Игрок 1 (рандом)': 0, 'Игрок 2 (машинное обучение)': 0, 'Ничья': 0}  # Счетчик побед и ничьих

    for epoch in range(epochs):  # Цикл по эпохам
        for _ in range(rounds):  # Цикл по раундам в каждой эпохе
            action1, action2, payoff = play_round(player1_strategy, player2_strategy, model, optimizer, criterion)
            total_payoff1 += payoff[0]  # Добавляем выигрыш игрока 1
            total_payoff2 += payoff[1]  # Добавляем выигрыш игрока 2

            # Обновляем счетчик побед
            if payoff[0] > payoff[1]:
                win_counts['Игрок 1 (рандом)'] += 1  # Игрок 1 выиграл раунд
            elif payoff[0] < payoff[1]:
                win_counts['Игрок 2 (машинное обучение)'] += 1  # Игрок 2 выиграл раунд
            else:
                win_counts['Ничья'] += 1  # Ничья

    return total_payoff1, total_payoff2, win_counts  # Возвращаем общие выигрыши и статистику побед

# Инициализация нейронной сети
model = Player2Net()  # Создаем экземпляр нейронной сети для игрока 2
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Используем оптимизатор Adam с learning rate 0.01
criterion = nn.MSELoss()  # Используем среднеквадратичную ошибку (MSE) в качестве функции потерь

# Запуск симуляции
total_payoff1, total_payoff2, win_counts = simulate_game(player1_strategy, player2_strategy, model, optimizer, criterion, rounds=1000, epochs=20)
print(f"Игрок 1 (рандом): {total_payoff1} очков")  # Выводим общий выигрыш игрока 1
print(f"Игрок 2 (машинное обучение): {total_payoff2} очков")  # Выводим общий выигрыш игрока 2

# График статистики побед
labels = win_counts.keys()  # Названия категорий (игроки и ничья)
sizes = win_counts.values()  # Количество побед в каждой категории

fig, ax = plt.subplots()  # Создаем график
ax.bar(labels, sizes)  # Строим столбчатую диаграмму
ax.set_title('Статистика побед в раундах')  # Заголовок графика
ax.set_xlabel('Игрок')  # Подпись оси X
ax.set_ylabel('Количество побед')  # Подпись оси Y
plt.show()  # Показываем график


# Инструкция по компиляции и запуску кода:
# 1. Убедитесь, что на вашем компьютере установлен Python (версия 3.6 или выше).
# 2. Скопируйте код в файл с расширением .py, например, game_theory.py.
# 3. Откройте терминал или командную строку.
# 4. Перейдите в директорию, где находится файл game_theory.py
# 5. Запустите код командой: python game_theory.py
# 6. Результаты симуляции будут выведены в терминал.