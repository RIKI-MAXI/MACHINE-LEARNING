import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. Datos de entrenamiento (problema XOR)
# Este problema requiere una red neuronal porque no es linealmente separable
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

print("=== RED NEURONAL CON BACKPROPAGATION ===")
print("Problema: Puerta XOR")
print("Entradas:\n", X)
print("Salidas esperadas:\n", y)

# 2. Crear red neuronal
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),  # Capa oculta con 4 neuronas
    Dense(1, activation='sigmoid')  # Capa de salida (probabilidad)
])

# 3. Compilar
# binary_crossentropy: función de pérdida para clasificación binaria
# adam: optimizador que ajusta automáticamente la tasa de aprendizaje
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\n=== ARQUITECTURA DE LA RED ===")
model.summary()

# 4. Entrenar con seguimiento del progreso
print("\n=== ENTRENAMIENTO ===")
history = model.fit(X, y, epochs=500, verbose=0)

# Visualizar el aprendizaje
plt.figure(figsize=(12, 4))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Evolución de la Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida (Binary Crossentropy)')
plt.grid(True)

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Evolución de la Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Evaluar predicciones
print("\n=== RESULTADOS FINALES ===")
predicciones = model.predict(X, verbose=0)

print("\nComparación de resultados:")
print("Entrada | Esperado | Predicción | Redondeado")
print("-" * 50)
for i in range(len(X)):
    print(f"{X[i]}  |    {y[i][0]}     |   {predicciones[i][0]:.4f}   |     {round(predicciones[i][0])}")

# Evaluar métricas finales
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nPérdida final: {loss:.4f}")
print(f"Precisión final: {accuracy:.2%}")

# Visualizar las regiones de decisión
print("\n=== ANÁLISIS DE BACKPROPAGATION ===")
print("El algoritmo ajustó los pesos mediante:")
print("1. Forward pass: calcular predicción")
print("2. Calcular error: diferencia con valor esperado")
print("3. Backward pass: propagar error hacia atrás")
print("4. Actualizar pesos: reducir el error")
print("5. Repetir 500 veces hasta convergencia")