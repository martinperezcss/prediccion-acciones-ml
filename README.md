# 📊 Predicción de Acciones con Machine Learning

Este proyecto aplica técnicas de *Machine Learning* para predecir el rendimiento futuro de acciones de **NVIDIA**, **Apple** y **Meta**. Utiliza modelos estadísticos y de aprendizaje automático sobre datos históricos y métricas fundamentales para explorar el potencial predictivo en el ámbito bursátil.

---

## 🗂️ Estructura del Proyecto

- **`main.py`**: Script principal para la carga de datos, entrenamiento de modelos y evaluación.
- **`requirements.txt`**: Lista de dependencias necesarias.
- **Informe PDF**: Análisis completo de resultados, gráficos y conclusiones (adjunto).

---

## ⚙️ Funcionamiento del Código

### 1. Carga de Datos
- Se espera un archivo `.csv` con datos históricos y métricas fundamentales.
- **Importante:** Cada usuario debe proporcionar su propio archivo CSV.
- El archivo debe incluir columnas como:
  - `Rendimiento_Apple`
  - `Rendimiento_NVIDIA`
  - `Rendimiento_Meta_Platforms`
  - Y otras variables explicativas.

### 2. Preprocesamiento
- Relleno de valores faltantes (`ffill`).
- Normalización de variables con `StandardScaler` o `MinMaxScaler`.

### 3. Modelos Implementados
- 🔹 **Regresión Lineal**
- 🔹 **Regresión Logística**
- 🔹 **Árboles de Decisión**
- 🔹 **Máquinas de Soporte Vectorial (SVM)**
- 🔹 **Redes Neuronales LSTM**

### 4. Evaluación de Modelos
- Métricas utilizadas:
  - RMSE (*Root Mean Squared Error*)
  - R² (*Coeficiente de determinación*)
- Visualizaciones:
  - Gráficos comparativos de valores reales vs. predichos.
  - Histogramas y gráficos de residuos.

---

## 📈 Resultados Destacados

- ✅ **Árboles de Decisión**: Mejor desempeño general (menor RMSE, mayor R²).
- 🍏 **Apple**: Mayor precisión predictiva (R² = 0.86).
- ⚠️ **LSTM**: Bajo rendimiento en este caso específico, con dificultades para capturar patrones temporales.

---

## 🚀 Requisitos

- Python 3.7 o superior.
- Instalar dependencias:
  ```bash
  pip install -r requirements.txt
