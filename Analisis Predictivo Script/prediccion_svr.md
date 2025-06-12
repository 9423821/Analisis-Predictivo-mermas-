# Resultados de Predicción: Support Vector Regression

## Resumen de Métricas

- **R²**: 0.0605 (Proporción de varianza explicada por el modelo)
- **RMSE**: 4.41 (Error cuadrático medio, en unidades de la variable objetivo)
- **MAE**: 0.94 (Error absoluto medio, en unidades de la variable objetivo)

## Interpretación

El modelo de Support Vector Regression explica aproximadamente el 6.1% de la variabilidad en las mermas. En promedio, las predicciones difieren de los valores reales en ±4.41 unidades.

## Muestra de Predicciones (Top 10)

| # | Valor Real | Predicción | Error | Error % | Categoría | Región |
|---|------------|------------|-------|---------|-----------|--------|
| 1500 | 12.65 | -1.55 | 14.20 | 112.3% | VACUNO ENVASADO | IX |
| 1775 | 6.29 | -1.16 | 7.45 | 118.4% | VACUNO ENVASADO | IX |
| 1530 | 1.00 | -0.79 | 1.79 | 178.9% | VIENESAS | IX |
| 958 | -0.01 | -0.14 | 0.13 | -847.3% | CECINAS GRANEL | IX |
| 1590 | -0.01 | -0.14 | 0.12 | -805.0% | CECINAS GRANEL | IX |
| 96 | -0.01 | -0.14 | 0.13 | -847.3% | CECINAS GRANEL | IX |
| 1815 | -0.01 | -0.12 | 0.11 | -732.2% | CECINAS GRANEL | IX |
| 788 | -0.01 | -0.13 | 0.12 | -766.8% | CECINAS GRANEL | IX |
| 1618 | -0.01 | -0.12 | 0.10 | -695.5% | CECINAS GRANEL | IX |
| 708 | -0.02 | -0.14 | 0.13 | -788.1% | CECINAS GRANEL | IX |

## Distribución del Error

- **Error Mínimo**: -102.72
- **Error Máximo**: 14.20
- **Error Promedio**: -0.67
- **Desviación Estándar del Error**: 4.36

*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*
