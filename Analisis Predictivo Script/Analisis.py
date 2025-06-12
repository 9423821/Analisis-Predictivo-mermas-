# IMPLEMENTACIÓN DE ANÁLISIS PREDICTIVO PARA MERMAS
# Utilizamos train.csv disponible en https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting (Es necesario 
# registrarse en la página de kaggle). Luego incorporar el archivo en el directorio de trabajo con python.
# Las librerias necesarias están en el archivo requirements.txt

# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  # Añadimos SVM como tercer modelo
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime as dt

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS PARA ANÁLISIS DE MERMAS*")

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
# Cargar el dataset (Para mermas es posible transformar a CSV o modificar script para conectarse a la base de datos. Queda a elecccion del grupo)
data = pd.read_csv('mermas_actividad_unidad_2.csv', sep=';', decimal=',')

# Convertir fechas a formato datetime con formato día/mes/año
data['fecha'] = pd.to_datetime(data['fecha'], format='%d-%m-%Y')

# Crear nuevas características para las fechas
data['mes_numero'] = data['fecha'].dt.month
data['dia_semana'] = data['fecha'].dt.dayofweek

# PASO 3: SELECCIÓN DE CARACTERÍSTICAS
# Características para predecir ventas. Estas son las que se utilizaron para el modelo. El trabajo a realizar implica testear otras variables en el caso de mermas
features = ['negocio', 'seccion', 'linea', 'categoria', 
           'abastecimiento', 'comuna', 'region', 'tienda',
           'motivo', 'ubicación_motivo', 'mes_numero', 'dia_semana']

X = data[features]
y = data['merma_unidad']  # Variable objetivo: cantidad de merma

# PASO 4: DIVISIÓN DE DATOS
# 80% entrenamiento, 20% prueba. Este porcentaje es el habitual en la literatura para este tipo de modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 5: PREPROCESAMIENTO
# Definir qué variables son categóricas y numéricas
categorical_features = ['negocio', 'seccion', 'linea', 'categoria', 
                       'abastecimiento', 'comuna', 'region', 'tienda',
                       'motivo', 'ubicación_motivo']
numeric_features = ['mes_numero', 'dia_semana']

# Crear preprocesador para manejar ambos tipos de variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# PASO 6: IMPLEMENTACIÓN DE MODELOS
# Modelo 1: Regresión Lineal. Este modelo es el habitual para este tipo de problemas debido a su simplicidad y interpretabilidad.
# En caso de mermas, es posible utilizar este modelo pero pueden explorar otros modelos mas eficientes.
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modelo 2: Random Forest
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 3. Support Vector Regression (nuevo modelo)
pipeline_svr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
])

# PASO 7: ENTRENAMIENTO DE MODELOS
# Entrenamos ambos modelos
print("Entrenando modelos...")
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_svr.fit(X_train, y_train)
print("Modelos entrenados correctamente")

# -------------------------------------------------
# EVALUACIÓN DE LOS MODELOS
# -------------------------------------------------

print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")

# PASO 8: REALIZAR PREDICCIONES CON LOS MODELOS ENTRENADOS
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_svr = pipeline_svr.predict(X_test)

# PASO 9: CALCULAR MÚLTIPLES MÉTRICAS DE EVALUACIÓN
# Error Cuadrático Medio (MSE)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_svr = mean_squared_error(y_test, y_pred_svr)

# Raíz del Error Cuadrático Medio (RMSE)
rmse_lr = np.sqrt(mse_lr)
rmse_rf = np.sqrt(mse_rf)
rmse_svr = np.sqrt(mse_svr)

# Error Absoluto Medio (MAE)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

# Coeficiente de Determinación (R²)
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
r2_svr = r2_score(y_test, y_pred_svr)

# NUEVO PASO: GUARDAR RESULTADOS DE PREDICCIÓN EN ARCHIVOS MARKDOWN
# Crear un DataFrame con las predicciones y valores reales
results_df = pd.DataFrame({
    'Valor_Real': y_test,
    'Prediccion_LR': y_pred_lr,
    'Prediccion_RF': y_pred_rf,
    'Prediccion_SVR': y_pred_svr,
    'Error_LR': y_test - y_pred_lr,
    'Error_RF': y_test - y_pred_rf,
    'Error_SVR': y_test - y_pred_svr,
    'Error_Porcentual_LR': ((y_test - y_pred_lr) / y_test) * 100,
    'Error_Porcentual_RF': ((y_test - y_pred_rf) / y_test) * 100,
    'Error_Porcentual_SVR': ((y_test - y_pred_svr) / y_test) * 100
})

# Reiniciar el índice para añadir información de las características
results_df = results_df.reset_index(drop=True)

# Añadir algunas columnas con información de las características para mayor contexto
X_test_reset = X_test.reset_index(drop=True)
for feature in X_test.columns:
    results_df[feature] = X_test_reset[feature]

# Ordenar por valor real para facilitar la comparación
results_df = results_df.sort_values('Valor_Real', ascending=False)

# Guardar resultado para Regresión Lineal
with open('prediccion_lr.md', 'w', encoding='utf-8') as f:
    f.write('# Resultados de Predicción: Regresión Lineal\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_lr:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_lr:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_lr:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Regresión Lineal explica aproximadamente el {r2_lr*100:.1f}% de la variabilidad en las mermas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_lr:.2f} unidades.\n\n')
    
    # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Ubicación |\n')
    f.write('|---|------------|------------|-------|---------|-----------|------------|\n')
    
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_LR']:.2f} | {row['Error_LR']:.2f} | "
                f"{row['Error_Porcentual_LR']:.1f}% | {row['categoria']} | {row['tienda']} |\n")
    
    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_LR"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_LR"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_LR"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_LR"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

# Guardar resultado para Random Forest
with open('prediccion_rf.md', 'w', encoding='utf-8') as f:
    f.write('# Resultados de Predicción: Random Forest\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_rf:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_rf:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_rf:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Random Forest explica aproximadamente el {r2_rf*100:.1f}% de la variabilidad en las mermas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_rf:.2f} unidades.\n\n')
    
    # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Región |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_RF']:.2f} | {row['Error_RF']:.2f} | "
                f"{row['Error_Porcentual_RF']:.1f}% | {row['categoria']} | {row['region']} |\n")

# Guardar resultado para Support Vector Regression 
with open('prediccion_svr.md', 'w', encoding='utf-8') as f:
    f.write('# Resultados de Predicción: Support Vector Regression\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_svr:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_svr:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_svr:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Support Vector Regression explica aproximadamente el {r2_svr*100:.1f}% de la variabilidad en las mermas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_svr:.2f} unidades.\n\n')
    
    # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Región |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_SVR']:.2f} | {row['Error_SVR']:.2f} | "
                f"{row['Error_Porcentual_SVR']:.1f}% | {row['categoria']} | {row['region']} |\n")
    
    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_SVR"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_SVR"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_SVR"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_SVR"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

print("Archivos de predicción generados: prediccion_lr.md, prediccion_rf.md y prediccion_svr.md")

# PASO 10: PRESENTAR RESULTADOS DE LAS MÉTRICAS EN FORMATO TABULAR
metrics_df = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Random Forest', 'Support Vector Regression'],
    'MSE': [mse_lr, mse_rf, mse_svr],
    'RMSE': [rmse_lr, rmse_rf, rmse_svr],
    'MAE': [mae_lr, mae_rf, mae_svr],
    'R²': [r2_lr, r2_rf, r2_svr]
})
print("\nComparación de métricas entre modelos:")
print(metrics_df)

# PASO 11: VISUALIZACIÓN DE PREDICCIONES VS VALORES REALES
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Mermas Reales')
plt.ylabel('Mermas Predichas')
plt.title('Random Forest: Predicciones vs Valores Reales')
plt.savefig('predicciones_vs_reales.png')
print("\nGráfico guardado: predicciones_vs_reales.png")

# PASO 12: VISUALIZACIÓN DE RESIDUOS PARA EVALUAR CALIDAD DEL MODELO
residuals = y_test - y_pred_rf
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - Random Forest')
plt.savefig('analisis_residuos.png')
print("Gráfico guardado: analisis_residuos.png")

# PASO 13: DISTRIBUCIÓN DE ERRORES
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - Random Forest')
plt.xlabel('Error')
plt.savefig('distribucion_errores.png')
print("Gráfico guardado: distribucion_errores.png")

# -------------------------------------------------
# DOCUMENTACIÓN DEL PROCESO
# -------------------------------------------------

print("\n=== DOCUMENTACIÓN DEL PROCESO ===")

# PASO 14: DOCUMENTAR LA EXPLORACIÓN INICIAL DE DATOS
print(f"Dimensiones del dataset: {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Período de tiempo analizado: de {data['fecha'].min()} a {data['fecha'].max()}")
print(f"Tipos de datos en las columnas principales:")
print(data[features + ['merma_unidad']].dtypes)

# PASO 15: DOCUMENTAR EL PREPROCESAMIENTO
print("\n--- PREPROCESAMIENTO APLICADO ---")
print(f"Variables numéricas: {numeric_features}")
print(f"Variables categóricas: {categorical_features}")
print("Transformaciones aplicadas:")
print("- Variables numéricas: Estandarización")
print("- Variables categóricas: One-Hot Encoding")

# PASO 16: DOCUMENTAR LA DIVISIÓN DE DATOS
print("\n--- DIVISIÓN DE DATOS ---")
print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/data.shape[0]:.1%} del total)")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/data.shape[0]:.1%} del total)")
print(f"Método de división: Aleatoria con random_state=42")

# PASO 17: DOCUMENTAR LOS MODELOS EVALUADOS
print("\n--- MODELOS IMPLEMENTADOS ---")
print("1. Regresión Lineal:")
print("   - Ventajas: Simple, interpretable")
print("   - Limitaciones: Asume relación lineal entre variables")

print("\n2. Random Forest Regressor:")
print("   - Hiperparámetros: n_estimators=100, random_state=42")
print("   - Ventajas: Maneja relaciones no lineales, menor riesgo de overfitting")
print("   - Limitaciones: Menos interpretable, mayor costo computacional")

print("\n3. Support Vector Regression:")
print("   - Ventajas: Efectivo en espacios de alta dimensión, versátil con diferentes funciones de kernel")
print("   - Limitaciones: Sensible a la escala de los datos, puede ser costoso computacionalmente")

# PASO 18: DOCUMENTAR LA VALIDACIÓN DEL MODELO
print("\n--- VALIDACIÓN DEL MODELO ---")
print("Método de validación: Evaluación en conjunto de prueba separado")
print("Métricas utilizadas: MSE, RMSE, MAE, R²")

# PASO 19: VISUALIZAR IMPORTANCIA DE CARACTERÍSTICAS
if hasattr(pipeline_rf['regressor'], 'feature_importances_'):
    print("\n--- IMPORTANCIA DE CARACTERÍSTICAS ---")
    # Obtener nombres de características después de one-hot encoding
    preprocessor = pipeline_rf.named_steps['preprocessor']
    cat_cols = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_cols])
    
    # Obtener importancias
    importances = pipeline_rf['regressor'].feature_importances_
    
    # Crear un DataFrame para visualización
    if len(feature_names) == len(importances):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Mostrar las 10 características más importantes
        print(feature_importance.head(10))
        
        # Visualizar
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Características Más Importantes')
        plt.savefig('importancia_caracteristicas.png')
        print("Gráfico guardado: importancia_caracteristicas.png")
    else:
        print("No se pudo visualizar la importancia de características debido a diferencias en la dimensionalidad")

# PASO 19.5: ANÁLISIS DE CORRELACIÓN CON SVR
print("\n=== ANÁLISIS DE CORRELACIÓN CON SVR ===")

# Obtener características importantes usando los coeficientes del SVR lineal
pipeline_svr_linear = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='linear', C=1.0))
])

# Entrenar el modelo SVR lineal
pipeline_svr_linear.fit(X_train, y_train)

# Crear DataFrame para análisis de correlación
correlation_data = pd.DataFrame()

# Agregar variables temporales
correlation_data['mes_numero'] = data['mes_numero']
correlation_data['dia_semana'] = data['dia_semana']
correlation_data['merma_unidad'] = data['merma_unidad']

# Agregar variables categóricas codificadas
for cat_feature in categorical_features:
    correlation_data[cat_feature] = pd.factorize(data[cat_feature])[0]

# Calcular correlaciones
correlations = correlation_data.corr()['merma_unidad'].sort_values(ascending=False)

# Visualización de correlaciones temporales
plt.figure(figsize=(12, 6))
temporal_correlations = correlations[['mes_numero', 'dia_semana']]
sns.barplot(x=temporal_correlations.values, y=temporal_correlations.index)
plt.title('Correlaciones de Variables Temporales con Mermas (SVR)')
plt.xlabel('Coeficiente de Correlación')
plt.tight_layout()
plt.savefig('correlaciones_temporales_svr.png')
print("\nGráfico de correlaciones temporales guardado: correlaciones_temporales_svr.png")

# Top 10 correlaciones
plt.figure(figsize=(12, 6))
top_correlations = correlations[1:11]
sns.barplot(x=top_correlations.values, y=top_correlations.index)
plt.title('Top 10 Correlaciones con Mermas (SVR)')
plt.xlabel('Coeficiente de Correlación')
plt.tight_layout()
plt.savefig('top_correlaciones_svr.png')
print("Gráfico de top correlaciones guardado: top_correlaciones_svr.png")

# Mapa de calor de correlaciones
plt.figure(figsize=(10, 8))
features_for_heatmap = ['merma_unidad', 'mes_numero', 'dia_semana'] + list(top_correlations.index[:3])
sns.heatmap(correlation_data[features_for_heatmap].corr(),
            annot=True,
            cmap='coolwarm',
            center=0)
plt.title('Mapa de Calor: Variables Temporales y Top Características (SVR)')
plt.tight_layout()
plt.savefig('heatmap_svr.png')
print("Mapa de calor guardado: heatmap_svr.png")

# Guardar análisis detallado en markdown
with open('analisis_correlaciones_svr.md', 'w', encoding='utf-8') as f:
    f.write('# Análisis de Correlaciones con SVR\n\n')
    
    # Variables temporales
    f.write('## Variables Temporales\n\n')
    f.write('| Variable | Correlación con Mermas | Interpretación |\n')
    f.write('|----------|----------------------|----------------|\n')
    for var in ['mes_numero', 'dia_semana']:
        corr = correlations[var]
        interp = ('Correlación fuerte' if abs(corr) > 0.5 else 
                 'Correlación moderada' if abs(corr) > 0.3 else 
                 'Correlación débil')
        f.write(f'| {var} | {corr:.3f} | {interp} |\n')
    
    # Top características
    f.write('\n## Top 5 Características más Correlacionadas\n\n')
    f.write('| Característica | Correlación | Interpretación |\n')
    f.write('|----------------|-------------|----------------|\n')
    for feature in top_correlations.head().index:
        corr = correlations[feature]
        interp = ('Correlación fuerte' if abs(corr) > 0.5 else 
                 'Correlación moderada' if abs(corr) > 0.3 else 
                 'Correlación débil')
        f.write(f'| {feature} | {corr:.3f} | {interp} |\n')

    # Análisis temporal detallado
    f.write('\n## Análisis Temporal Detallado\n\n')
    
    # Análisis por mes
    f.write('### Patrones Mensuales\n')
    monthly_stats = data.groupby('mes_numero')['merma_unidad'].agg(['mean', 'std']).round(2)
    f.write('\n```\n')
    f.write(monthly_stats.to_string())
    f.write('\n```\n')
    
    # Análisis por día de la semana
    f.write('\n### Patrones por Día de la Semana\n')
    daily_stats = data.groupby('dia_semana')['merma_unidad'].agg(['mean', 'std']).round(2)
    f.write('\n```\n')
    f.write(daily_stats.to_string())
    f.write('\n```\n')

print("Análisis detallado guardado en: analisis_correlaciones_svr.md")

# PASO 20: CONCLUSIÓN
print("\n=== CONCLUSIÓN ===")
print(f"El mejor modelo según R² es: {'Random Forest' if r2_rf > r2_lr else 'Regresión Lineal'}")
print(f"R² del mejor modelo: {max(r2_rf, r2_lr):.4f}")
print(f"RMSE del mejor modelo: {rmse_rf if r2_rf > r2_lr else rmse_lr:.2f}")

# Explicaciones adicionales para facilitar la interpretación
print("\n--- INTERPRETACIÓN DE RESULTADOS ---")
print(f"• R² (Coeficiente de determinación): Valor entre 0 y 1 que indica qué proporción de la variabilidad")
print(f"  en las mermas/ventas es explicada por el modelo. Un valor de {max(r2_rf, r2_lr):.4f} significa que")
print(f"  aproximadamente el {max(r2_rf, r2_lr)*100:.1f}% de la variación puede ser explicada por las variables utilizadas.")

print(f"\n• RMSE (Error cuadrático medio): Representa el error promedio de predicción en las mismas unidades")
print(f"  que la variable objetivo. Un RMSE de {rmse_rf if r2_rf > r2_lr else rmse_lr:.2f} significa que, en promedio,")
print(f"  las predicciones difieren de los valores reales en ±{rmse_rf if r2_rf > r2_lr else rmse_lr:.2f} unidades.")

print(f"\n• {'Random Forest' if r2_rf > r2_lr else 'Regresión Lineal'} es el mejor modelo porque:")
if r2_rf > r2_lr:
    print("  - Captura mejor las relaciones no lineales entre las variables")
    print("  - Tiene mayor capacidad predictiva (R² más alto)")
    print("  - Menor error de predicción (RMSE más bajo)")
else:
    print("  - Ofrece un buen equilibrio entre simplicidad y capacidad predictiva")
    print("  - Es más interpretable que modelos complejos")
    print("  - Presenta un mejor ajuste a los datos en este caso específico")

print("\nEl análisis predictivo ha sido completado exitosamente.")

# ANÁLISIS DE CORRELACIÓN TOP 10 VARIABLES CON MERMAS
print("\n=== ANÁLISIS DE CORRELACIÓN: TOP 10 VARIABLES CON MERMAS ===")

# Preparar datos para correlación
correlation_data = pd.DataFrame()
correlation_data['merma_unidad'] = data['merma_unidad']

# Agregar todas las variables
for num_feature in numeric_features:
    correlation_data[num_feature] = data[num_feature]
for cat_feature in categorical_features:
    correlation_data[cat_feature] = pd.factorize(data[cat_feature])[0]

# Calcular correlaciones con mermas
correlations = correlation_data.corr()['merma_unidad'].sort_values(ascending=False)
top_10_correlations = correlations[1:11]  # Excluimos la autocorrelación

# Visualización de las top 10 correlaciones
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_correlations.values, y=top_10_correlations.index)
plt.title('Top 10 Variables más Correlacionadas con Mermas')
plt.xlabel('Coeficiente de Correlación')
plt.tight_layout()
plt.savefig('top_10_correlaciones_mermas.png')

# Crear mapa de calor para las top 10 variables
plt.figure(figsize=(12, 8))
top_features = ['merma_unidad'] + list(top_10_correlations.index)
correlation_matrix = correlation_data[top_features].corr()

sns.heatmap(correlation_matrix,
            annot=True,
            cmap='RdYlBu',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .5})

plt.title('Matriz de Correlaciones - Top 10 Variables', pad=20)
plt.tight_layout()
plt.savefig('correlaciones_heatmap_tabla.png', bbox_inches='tight', dpi=300)
print("Mapa de calor guardado como: correlaciones_heatmap_tabla.png")

# Guardar análisis detallado
with open('analisis_top_10_correlaciones.md', 'w', encoding='utf-8') as f:
    f.write('# Análisis de Top 10 Variables más Correlacionadas con Mermas\n\n')
    
    # Tabla de correlaciones
    f.write('## Coeficientes de Correlación\n\n')
    f.write('| Variable | Correlación | Fuerza de la Correlación |\n')
    f.write('|----------|-------------|------------------------|\n')
    
    for var in top_10_correlations.index:
        corr = top_10_correlations[var]
        strength = ('Muy fuerte' if abs(corr) > 0.8 else
                   'Fuerte' if abs(corr) > 0.6 else
                   'Moderada' if abs(corr) > 0.4 else
                   'Débil' if abs(corr) > 0.2 else 'Muy débil')
        f.write(f'| {var} | {corr:.3f} | {strength} |\n')
    
    # Análisis estadístico por variable
    f.write('\n## Análisis Estadístico Detallado\n\n')
    for var in top_10_correlations.index:
        f.write(f'\n### {var}\n')
        f.write('#### Estadísticas Descriptivas:\n')
        if var in numeric_features:
            stats = data[var].describe()
            f.write(f'```\n{stats.to_string()}\n```\n')
        else:
            value_counts = data[var].value_counts().head()
            f.write(f'Top 5 valores más frecuentes:\n```\n{value_counts.to_string()}\n```\n')
        
        f.write(f'Correlación con mermas: {correlations[var]:.3f}\n')

print("\nAnálisis completado. Archivos generados:")
print("- top_10_correlaciones_mermas.png")
print("- heatmap_top_10_correlaciones.png")
print("- analisis_top_10_correlaciones.md")

# Crear una figura compuesta con ambas visualizaciones
plt.figure(figsize=(20, 10))

# Primera subfigura - Tabla de correlaciones
plt.subplot(1, 2, 1)
ax1 = plt.gca()
ax1.axis('tight')
ax1.axis('off')

# Preparar datos para la tabla
table_data = []
headers = ['Variable', 'Correlación', 'Fuerza', 'Significancia']

for var in top_10_correlations.index:
    corr = top_10_correlations[var]
    strength = ('Muy fuerte' if abs(corr) > 0.8 else
               'Fuerte' if abs(corr) > 0.6 else
               'Moderada' if abs(corr) > 0.4 else
               'Débil' if abs(corr) > 0.2 else 'Muy débil')
    significance = '***' if abs(corr) > 0.5 else '**' if abs(corr) > 0.3 else '*'
    table_data.append([var, f'{corr:.3f}', strength, significance])

# Crear tabla
table = ax1.table(cellText=table_data,
                 colLabels=headers,
                 loc='center',
                 cellLoc='center',
                 colColours=['#f2f2f2']*4,
                 cellColours=[['#ffffff']*4 for _ in range(len(table_data))],
                 bbox=[0, 0, 1, 1])

# Formatear tabla
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
plt.title('Top 10 Variables Correlacionadas con Mermas', pad=20, size=14)

# Segunda subfigura - Mapa de calor
plt.subplot(1, 2, 2)
top_features = ['merma_unidad'] + list(top_10_correlations.index)
correlation_matrix = correlation_data[top_features].corr()

sns.heatmap(correlation_matrix,
            annot=True,
            cmap='RdYlBu',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .5})

plt.title('Matriz de Correlaciones - Top 10 Variables', pad=20)

# Añadir leyenda general
plt.figtext(0.1, 0.02, '*** Correlación fuerte (>0.5)', fontsize=8)
plt.figtext(0.35, 0.02, '** Correlación moderada (>0.3)', fontsize=8)
plt.figtext(0.6, 0.02, '* Correlación débil (<0.3)', fontsize=8)

# Ajustar layout y guardar
plt.tight_layout()
plt.savefig('correlaciones_completo.png', bbox_inches='tight', dpi=300, 
            facecolor='white', edgecolor='none')
print("Visualización completa guardada como: correlaciones_completo.png")

# PASO 19: CONCLUSIONES
print("\n=== CONCLUSIONES DEL ANÁLISIS ===")

# Calcular métricas para todos los modelos
modelos_metricas = {
    'Regresión Lineal': {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'R²': r2_score(y_test, y_pred_lr)
    },
    'Random Forest': {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'R²': r2_score(y_test, y_pred_rf)
    },
    'SVR': {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
        'MAE': mean_absolute_error(y_test, y_pred_svr),
        'R²': r2_score(y_test, y_pred_svr)
    }
}

# Mostrar resultados directamente en consola
print("\nComparación de modelos:")
print("-" * 60)
print(f"{'Modelo':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 60)
for modelo, metricas in modelos_metricas.items():
    print(f"{modelo:<20} {metricas['RMSE']:>10.2f} {metricas['MAE']:>10.2f} {metricas['R²']:>10.3f}")
print("-" * 60)

# Conclusiones finales
mejor_modelo = max(modelos_metricas.items(), key=lambda x: x[1]['R²'])[0]
mejor_r2 = modelos_metricas[mejor_modelo]['R²']

print("\nCONCLUSIONES FINALES:")
print(f"- Mejor modelo: {mejor_modelo}")
print(f"- R² del mejor modelo: {mejor_r2:.3f}")
print(f"- El modelo explica el {mejor_r2*100:.1f}% de la variabilidad en las mermas")
print(f"- RMSE: {modelos_metricas[mejor_modelo]['RMSE']:.2f}")
print(f"- MAE: {modelos_metricas[mejor_modelo]['MAE']:.2f}")

# ANÁLISIS SVR
print("\n=== ANÁLISIS CON SUPPORT VECTOR REGRESSION ===")

# 1. Métricas de rendimiento
print("\nMétricas de rendimiento SVR:")
print("-" * 50)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.2f}")
print(f"R²: {r2_score(y_test, y_pred_svr):.3f}")

# 2. Top 10 predicciones
print("\nTop 10 predicciones vs valores reales:")
print("-" * 50)
print(f"{'Real':>10} {'Predicho':>10} {'Error':>10} {'Error %':>10}")
print("-" * 50)
predictions_df = pd.DataFrame({
    'Real': y_test,
    'Predicho': y_pred_svr,
    'Error': y_test - y_pred_svr
})
predictions_df['Error %'] = (predictions_df['Error'] / predictions_df['Real']) * 100
for _, row in predictions_df.head(10).iterrows():
    print(f"{row['Real']:10.2f} {row['Predicho']:10.2f} {row['Error']:10.2f} {row['Error %']:10.1f}%")

# 3. Análisis de correlaciones
print("\nCorrelaciones principales con mermas:")
print("-" * 50)
correlation_data = pd.DataFrame()
correlation_data['merma_unidad'] = data['merma_unidad']

for num_feature in numeric_features:
    correlation_data[num_feature] = data[num_feature]
for cat_feature in categorical_features:
    correlation_data[cat_feature] = pd.factorize(data[cat_feature])[0]

correlations = correlation_data.corr()['merma_unidad'].sort_values(ascending=False)
print("\nTop 10 variables más correlacionadas:")
for var, corr in correlations[1:11].items():
    strength = ('Muy fuerte' if abs(corr) > 0.8 else
               'Fuerte' if abs(corr) > 0.6 else
               'Moderada' if abs(corr) > 0.4 else
               'Débil' if abs(corr) > 0.2 else 'Muy débil')
    print(f"{var:<30} {corr:>8.3f} ({strength})")

# 4. Estadísticas de error
print("\nEstadísticas de error del modelo SVR:")
print("-" * 50)
errors = y_test - y_pred_svr
print(f"Error medio: {errors.mean():.2f}")
print(f"Desviación estándar del error: {errors.std():.2f}")
print(f"Error mínimo: {errors.min():.2f}")
print(f"Error máximo: {errors.max():.2f}")

# 5. Conclusiones SVR
print("\nConclusiones del modelo SVR:")
print("-" * 50)
print(f"- El modelo SVR explica el {r2_score(y_test, y_pred_svr)*100:.1f}% de la variabilidad en las mermas")
print(f"- Error promedio en las predicciones: ±{np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f} unidades")
print(f"- Precisión general del modelo: {'Alta' if r2_score(y_test, y_pred_svr) > 0.7 else 'Media' if r2_score(y_test, y_pred_svr) > 0.5 else 'Baja'}")