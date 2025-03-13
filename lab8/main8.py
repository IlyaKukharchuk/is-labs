import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Парсинг и обработка данных
# Загрузка данных из CSV файла
data = pd.read_csv('Fish.csv')

# Удаление строк с отсутствующими значениями
# Это необходимо, чтобы избежать проблем с пропущенными данными при обучении модели
data = data.dropna()

# Преобразование категориальной переменной 'Species' в числовую с помощью one-hot encoding
# Это позволяет использовать категориальные данные в нейронной сети
data = pd.get_dummies(data, columns=['Species'], drop_first=True)

# Разделение данных на признаки (X) и целевую переменную (y)
# Признаки - это входные данные, а целевая переменная - это то, что мы пытаемся предсказать
X = data.drop(columns=['Weight'])
y = data['Weight']

# 2. Разделение данных на обучающую и проверочную выборки
# Это необходимо для оценки производительности модели на данных, которые она не видела во время обучения
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
# Это помогает улучшить производительность модели, приводя все признаки к одному масштабу
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Создание модели нейронной сети
# Инициализация последовательной модели
model = Sequential()

# Добавление слоев в модель
# Первый слой с 64 нейронами и функцией активации ReLU
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# Второй слой с 32 нейронами и функцией активации ReLU
model.add(Dense(32, activation='relu'))
# Третий слой с 16 нейронами и функцией активации ReLU
model.add(Dense(16, activation='relu'))
# Выходной слой с одним нейроном для регрессии
model.add(Dense(1))

# Компиляция модели
# Оптимизатор 'adam' и функция потерь 'mean_squared_error' используются для обучения модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
# Модель обучается на обучающих данных в течение 100 эпох с размером пакета 32
# 20% данных используются для проверки во время обучения
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Сохранение обученной модели
model.save('fish_weight_model.h5')

# Функция для ввода данных и предсказания веса
def predict_fish_weight(input_data):
    # Преобразование входных данных в DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Масштабирование входных данных
    input_scaled = scaler.transform(input_df)

    # Предсказание веса
    predicted_weight = model.predict(input_scaled)
    return predicted_weight[0][0]

# Пример использования функции
input_data = {
    'Length1': 23.2,
    'Length2': 25.4,
    'Length3': 30.1,
    'Height': 11.52,
    'Width': 4.02,
    'Species_Bream': 1,
    'Species_Parkki': 0,
    'Species_Perch': 0,
    'Species_Pike': 0,
    'Species_Roach': 0,
    'Species_Smelt': 0,
    'Species_Whitefish': 0
}

predicted_weight = predict_fish_weight(input_data)
print(f'Предсказанный вес рыбы: {predicted_weight}')

# 4. Визуализация результатов обучения
# Построение графика потерь для обучающей и проверочной выборок
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Предсказание на тестовых данных
# Использование обученной модели для предсказания значений на тестовых данных
y_pred = model.predict(X_test)

# Визуализация предсказаний
# Построение графика истинных значений vs предсказанных значений
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.show()
