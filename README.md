Informe Final 
# 📊 Telecom-X - Parte 2: Modelo Predictivo de Churn

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predicción de evasión de clientes mediante Machine Learning**



## 🎯 Propósito del Proyecto

Este proyecto tiene como objetivo principal **predecir la probabilidad de churn (cancelación)** de clientes de Telecom-X utilizando técnicas de Machine Learning. 

A partir de los datos limpios de la [Parte 1 (EDA)](https://github.com/Jesy284/Telecom-X-Challenge-Parte1), se desarrolla un modelo predictivo que permite:

- 🔍 Identificar clientes con alto riesgo de cancelación antes de que ocurra
- 📊 Priorizar acciones de retención basadas en evidencia de datos
- 💰 Optimizar recursos de marketing y fidelización
- 📈 Reducir la tasa de churn y mejorar la rentabilidad



## 📁 Estructura del Proyecto

Telecom-X-Parte2/
│
├── 📁 data/
│ ├── 📄 datos_tratados.csv # Dataset limpio de la Parte 1
│ └── 📄 metricas_modelo.json # Métricas de evaluación del modelo
│
├── 📁 notebooks/
│ └── 📄 Telecom-X-Parte2-Modelo-Predictivo.ipynb # Notebook principal ⭐
│
├── 📁 models/
│ ├── 📄 modelo_churn.pkl # Modelo entrenado (Random Forest)
│ └── 📄 scaler.pkl # Escalador para nuevas predicciones
│
├── 📁 images/
│ ├── 📄 matriz_confusion.png # Matriz de confusión
│ ├── 📄 curva_roc.png # Curva ROC
│ └── 📄 importancia_variables.png # Importancia de features
│
├── 📄 requirements.txt # Dependencias del proyecto
├── 📄 README.md # Este archivo
└── 📄 LICENSE # Licencia MIT


---

## ⚙️ Preparación de los Datos

### Clasificación de Variables

| Tipo | Variables | Tratamiento Aplicado |
|------|-----------|---------------------|
| **Numéricas** | `antiguedad_meses`, `cargo_mensual`, `cargo_total` | Estandarización con StandardScaler |
| **Binarias** | `es_senior`, `tiene_pareja`, `tiene_dependientes` | Sin transformación (ya son 0/1) |
| **Categóricas** | `tipo_contrato`, `metodo_pago` | Label Encoding |
| **Target** | `cancelacion_num` | 0 = Activo, 1 = Canceló |

### Pipeline de Transformación

1. **Limpieza inicial**: Eliminación de nulos en columnas críticas
2. **Codificación**: Variables categóricas → Label Encoding
3. **Estandarización**: `(x - media) / desviación` para variables numéricas
4. **División de datos**: 
   - Train: 80% | Test: 20%
   - Estratificación por variable objetivo para mantener distribución de clases

### Justificación de Decisiones

- **Estratificación**: Evita sesgos debido al desbalance natural de clases (~27% churn)
- **Estandarización**: Necesaria para algoritmos sensibles a escala y para comparar importancia de features
- **Label Encoding vs One-Hot**: Se usa Label Encoding para mantener dimensionalidad baja con Random Forest (que maneja bien variables ordinales codificadas)
- **Random Forest**: Elegido por su robustez, interpretabilidad (importancia de variables) y buen rendimiento sin necesidad de tuning extensivo

---

## 📈 Resultados del Modelo

### Métricas de Evaluación (Conjunto de Prueba)

| Métrica | Valor | Interpretación |
|---------|-------|---------------|
| **Accuracy** | 0.XX | % de predicciones correctas totales |
| **Precision** | 0.XX | De los predichos como churn, % que realmente churnearon |
| **Recall** | 0.XX | De los que realmente churnearon, % que identificamos |
| **F1-Score** | 0.XX | Balance armónico entre precision y recall |
| **ROC-AUC** | 0.XX | Capacidad del modelo para discriminar entre clases |

> 📊 *Valores actualizados al ejecutar el notebook*

### Visualizaciones Clave

![Matriz de Confusión](https://github.com/Jesy284/Telecom-X-Parte2-Churn-Prediction/blob/main/Grafica%201.Matriz%20de%20Confusion.png?raw=true)
*Figura 1: El modelo muestra buen equilibrio entre falsos positivos y falsos negativos*

![Curva ROC](images/curva_roc.png)
*Figura 2: AUC > 0.70 indica capacidad predictiva útil para negocio*

![Importancia de Variables](images/importancia_variables.png)
*Figura 3: `tipo_contrato` y `antiguedad_meses` son los predictores más fuertes*


## 📊 Resultados del Modelo

### Métricas Obtenidas
- **Accuracy**: 76.3%
- **Precision**: 54.3%
- **Recall**: 68.2%
- **F1-Score**: 60.5%
- **ROC-AUC**: 0.830

### Hallazgos Clave
1. El **cargo mensual** es el predictor más importante del churn
2. El **tipo de contrato** tiene un impacto significativo
3. Los clientes con **menor antigüedad** tienen mayor riesgo
4. El modelo identifica correctamente el 68% de los clientes que cancelarán

### Recomendaciones de Negocio
- Ofrecer descuentos en contratos mensuales para convertirlos a anuales
- Implementar programa de retención para clientes nuevos (<12 meses)
- Revisar estructura de precios para cargos mensuales altos
---

## 🔍 Insights Estratégicos

### Factores con Mayor Impacto en Churn

1. **📋 Tipo de Contrato** (Mayor importancia)
   - Contratos "Mes a mes": Riesgo 4x mayor vs. contratos anuales
   - Recomendación: Campañas de conversión a largo plazo

2. **⏰ Antigüedad del Cliente**
   - Clientes <12 meses: 2.1x más propensos a cancelar
   - Recomendación: Programa de onboarding y retención temprana

3. **💰 Cargo Mensual**
   - Cargos >$70/mes correlacionan con mayor churn
   - Recomendación: Revisar percepción de valor en planes premium

### Segmentos de Alto Riesgo Identificados

```python
# Perfil de cliente con mayor probabilidad de churn:
{
    "tipo_contrato": "Mes a mes",
    "antiguedad_meses": "< 12",
    "cargo_mensual": "> $70",
    "metodo_pago": "Cheque electrónico",
    "probabilidad_churn_predicha": "> 70%"
}

💡 Recomendaciones de Negocio
Prioridad 1: Retención Proactiva de Clientes Nuevos
🎯 Implementar contacto proactivo en meses 1, 3, 6 y 9
🎁 Ofrecer beneficios de fidelización temprana
📊 Establecer alertas automáticas para señales de riesgo
Prioridad 2: Conversión de Contratos Flexibles
💰 Descuento del 10-15% por contrato anual
🎁 Mes gratis en contrato de 2 años
📞 Asesoría personalizada para explicar beneficios
Prioridad 3: Optimización de Experiencia de Pago
💳 Incentivar pagos automáticos con descuento del 5%
🔔 Recordatorios amigables multicanal
🛠️ Asistencia para configurar método de pago preferido
Prioridad 4: Segmentación Inteligente
🤖 Usar el modelo para score de riesgo en tiempo real
🎯 Ofertas personalizadas según perfil de riesgo
📈 Monitoreo continuo de efectividad de acciones

## 🚀 Ejecución
1. Abre el notebook en Google Colab
2. Ejecuta todas las celdas en orden
3. Los datos se cargan automáticamente desde la API


🚀 Instrucciones de Ejecución
Requisitos
Python 3.10+
Google Colab o Jupyter Notebook
Conexión a internet (para datos)
Opción A: Google Colab (Recomendado)
1. Abre https://colab.research.google.com
2. Ve a Archivo → Abrir notebook → GitHub
3. Busca: Jesy284/Telecom-X-Parte2
4. Selecciona: Telecom-X-Parte2-Modelo-Predictivo.ipynb
5. Ejecuta celda por celda con Shift+Enter

Opción B: Ejecución Local
# Clonar repositorio
git clone https://github.com/Jesy284/Telecom-X-Parte2.git
cd Telecom-X-Parte2

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook
jupyter notebook notebooks/Telecom-X-Parte2-Modelo-Predictivo.ipynb

Cargar Datos y Predecir
import pandas as pd
import joblib

# Cargar datos nuevos
nuevos_clientes = pd.read_csv('nuevos_datos.csv')

# Cargar modelo y escalador
modelo = joblib.load('models/modelo_churn.pkl')
scaler = joblib.load('models/scaler.pkl')

# Preprocesar y predecir
X_nuevos = nuevos_clientes[features_numericas].fillna(0)
X_nuevos_scaled = scaler.transform(X_nuevos)
predicciones = modelo.predict(X_nuevos_scaled)
probabilidades = modelo.predict_proba(X_nuevos_scaled)[:, 1]

# Agregar resultados al DataFrame
nuevos_clientes['prediccion_churn'] = predicciones
nuevos_clientes['probabilidad_churn'] = probabilidades.round(3)

📦 Dependencias
# requirements.txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
joblib>=1.2.0

🤝 Contribuciones
Las contribuciones son bienvenidas. Para contribuir:
Haz un fork del proyecto
Crea una rama: git checkout -b feature/nueva-mejora
Commit: git commit -m 'feat: agregar nueva funcionalidad'
Push: git push origin feature/nueva-mejora
Abre un Pull Request


 🚀 Instrucciones de Ejecución
Requisitos
Python 3.10+
Google Colab o Jupyter Notebook
Conexión a internet (para datos)
Opción A: Google Colab (Recomendado)
1. Abre https://colab.research.google.com
2. Ve a Archivo → Abrir notebook → GitHub
3. Busca: Jesy284/Telecom-X-Parte2
4. Selecciona: Telecom-X-Parte2-Modelo-Predictivo.ipynb
5. Ejecuta celda por celda con Shift+Enter
Opción B: Ejecución Local
# Clonar repositorio
git clone https://github.com/Jesy284/Telecom-X-Parte2.git
cd Telecom-X-Parte2

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook
jupyter notebook notebooks/Telecom-X-Parte2-Modelo-Predictivo.ipynb

Cargar Datos y Predecir
import pandas as pd
import joblib

# Cargar datos nuevos
nuevos_clientes = pd.read_csv('nuevos_datos.csv')

# Cargar modelo y escalador
modelo = joblib.load('models/modelo_churn.pkl')
scaler = joblib.load('models/scaler.pkl')

# Preprocesar y predecir
X_nuevos = nuevos_clientes[features_numericas].fillna(0)
X_nuevos_scaled = scaler.transform(X_nuevos)
predicciones = modelo.predict(X_nuevos_scaled)
probabilidades = modelo.predict_proba(X_nuevos_scaled)[:, 1]

# Agregar resultados al DataFrame
nuevos_clientes['prediccion_churn'] = predicciones
nuevos_clientes['probabilidad_churn'] = probabilidades.round(3)

📦 Dependencias
# requirements.txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
joblib>=1.2.0

🤝 Contribuciones
Las contribuciones son bienvenidas. Para contribuir:
Haz un fork del proyecto
Crea una rama: git checkout -b feature/nueva-mejora
Commit: git commit -m 'feat: agregar nueva funcionalidad'
Push: git push origin feature/nueva-mejora
Abre un Pull Request

📝 Licencia
Este proyecto está bajo la Licencia MIT. Ver LICENSE para detalles.

👤 Autor
Jesica Sosa G
 🐙 https://github.com/Jesy284/Telecom-X-Parte2-Churn-Prediction
      
Proyecto desarrollado como parte del programa Alura ONE - Data Science LATAM
Challenge Telecom X - Parte 2: Modelo Predictivo de Churn



