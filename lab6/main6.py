import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Загрузка и подготовка датасета MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Визуализация метрик
def plot_metrics(history):
    # Точность (accuracy)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('Точность модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    # Потери (loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('Потери модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.show()

# Вызов функции для визуализации
plot_metrics(history)

# Функция для загрузки и подготовки изображения
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = 1 - img / 255.0  # Инверсия цветов и нормирование
    return img

# Загрузка и подготовка изображения
img_path = '9.png'  # Укажите путь к вашему изображению
img = prepare_image(img_path)

# Распознавание цифры
prediction = model.predict(img)
predicted_class = np.argmax(prediction, axis=1)
print(f'Распознанная цифра: {predicted_class[0]}')

# Отображение изображения
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f'Распознанная цифра: {predicted_class[0]}')
plt.show()