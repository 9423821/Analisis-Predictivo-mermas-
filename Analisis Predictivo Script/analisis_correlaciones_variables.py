import pandas as pd
import matplotlib.pyplot as plt

def create_styled_table(ax, title, data_dict, header_color='#2c3e50'):
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos para la tabla
    table_data = [[k, f"{v['count']:,}", f"{v['merma']:.1f}%"] 
                  for k, v in data_dict.items()]
    
    # Crear tabla estilizada
    table = ax.table(cellText=table_data,
                    colLabels=['Categoría', 'Cantidad', 'Merma Promedio'],
                    loc='center',
                    cellLoc='center',
                    colColours=[header_color]*3,
                    colWidths=[0.5, 0.25, 0.25],
                    bbox=[0, 0, 1, 1])
    
    # Estilo de tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    for k, cell in table._cells.items():
        if k[0] == 0:  # Encabezados
            cell.set_text_props(color='white', weight='bold')
            cell.set_facecolor(header_color)
        else:  # Filas alternadas
            cell.set_facecolor('#ffffff' if k[0] % 2 == 0 else '#f8f9fa')
    
    plt.title(title, pad=20, size=12)

# Cargar datos
print("Cargando datos...")
data = pd.read_csv('mermas_actividad_unidad_2.csv', sep=';', decimal=',')

# Crear figura
plt.figure(figsize=(15, 10))

# Variables a analizar
variables = {
    'tienda': 'Tiendas',
    'negocio': 'Negocios',
    'abastecimiento': 'Tipos de Abastecimiento',
    'motivo': 'Motivos'
}

# Crear tablas para cada variable
for i, (var, title) in enumerate(variables.items(), 1):
    plt.subplot(2, 2, i)
    
    # Calcular resumen por categoría
    summary = data.groupby(var).agg({
        'merma_unidad': ['count', 'mean']
    })
    summary.columns = ['count', 'merma']
    
    # Convertir a diccionario
    summary_dict = {
        index: {
            'count': row['count'],
            'merma': row['merma']
        }
        for index, row in summary.iterrows()
    }
    
    create_styled_table(plt.gca(), title, summary_dict)

plt.tight_layout(pad=3.0)
plt.savefig('resumen_general_mermas.png', 
            bbox_inches='tight', 
            dpi=300, 
            facecolor='white', 
            edgecolor='none')
plt.show()

# Mostrar resumen en consola
print("\n=== RESUMEN POR CATEGORÍAS ===")
for var, title in variables.items():
    print(f"\n{title.upper()}")
    summary = data.groupby(var).agg({
        'merma_unidad': ['count', 'mean']
    })
    print(summary.round(2))
    print("-" * 50)