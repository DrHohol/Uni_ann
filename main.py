from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Завантаження набору даних Iris
data = load_iris()
X = data.data  # Вхідні дані (ознаки квітки)
y = data.target  # Мітки класів (види квіток)
# Розділення даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=5,
validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test)
train_loss, train_acc = model.evaluate(X_train, y_train)
print(f"Точність на тренувальних даних: {train_acc * 100:.2f}%")
print(f"Точність на тестових даних: {test_acc * 100:.2f}%")

