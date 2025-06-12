# Análisis de Correlaciones con SVR

## Variables Temporales

| Variable | Correlación con Mermas | Interpretación |
|----------|----------------------|----------------|
| mes_numero | nan | Correlación débil |
| dia_semana | -0.006 | Correlación débil |

## Top 5 Características más Correlacionadas

| Característica | Correlación | Interpretación |
|----------------|-------------|----------------|
| abastecimiento | 0.119 | Correlación débil |
| comuna | 0.005 | Correlación débil |
| dia_semana | -0.006 | Correlación débil |
| seccion | -0.011 | Correlación débil |
| motivo | -0.011 | Correlación débil |

## Análisis Temporal Detallado

### Patrones Mensuales

```
            mean   std
mes_numero            
12         -1.69  5.77
```

### Patrones por Día de la Semana

```
            mean   std
dia_semana            
0          -1.72  4.98
1          -1.68  8.16
2          -0.52  0.86
3          -1.65  3.82
4          -1.64  3.43
5          -1.73  3.58
6          -1.82  9.28
```
