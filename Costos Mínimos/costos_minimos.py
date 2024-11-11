import numpy as np
import pandas as pd

def costos_minimos(costos, oferta, demanda, output_excel):
    m, n = costos.shape
    x = np.zeros((m, n), dtype=int)
    
    oferta_restante = oferta.copy()
    demanda_restante = demanda.copy()
    
    # Convertir la matriz de costos a float para manejar np.inf
    costos = costos.astype(float)
    costos_originales = costos.copy()
    
    iteracion = 1
    pasos = []  # Almacenar la información de cada iteración

    while np.sum(oferta_restante) > 0 and np.sum(demanda_restante) > 0:
        # Guardar los datos actuales en la lista de pasos
        pasos.append({
            'iteracion': iteracion,
            'matriz_asignacion': x.copy(),
            'oferta_restante': oferta_restante.copy(),
            'demanda_restante': demanda_restante.copy()
        })
        
        # Encontrar el índice del costo mínimo
        min_cost = np.min(costos)
        indices_min = np.argwhere(costos == min_cost)
        
        # Selección de índice según el costo mínimo
        if len(indices_min) == 1:
            i, j = indices_min[0]
        else:
            max_oferta = -1
            for idx in indices_min:
                i_temp, j_temp = idx
                if oferta_restante[i_temp] > max_oferta:
                    max_oferta = oferta_restante[i_temp]
                    i, j = i_temp, j_temp
        
        # Asignar el máximo posible
        asignacion = min(oferta_restante[i], demanda_restante[j])
        x[i, j] = asignacion
        
        # Actualizar oferta y demanda restantes
        oferta_restante[i] -= asignacion
        demanda_restante[j] -= asignacion
        
        # Actualizar los costos para evitar reutilizar filas o columnas agotadas
        if oferta_restante[i] == 0:
            costos[i, :] = np.inf
        if demanda_restante[j] == 0:
            costos[:, j] = np.inf
        
        iteracion += 1

    # Capturar la iteración final cuando la oferta y demanda se agotan
    pasos.append({
        'iteracion': iteracion,
        'matriz_asignacion': x.copy(),
        'oferta_restante': oferta_restante.copy(),
        'demanda_restante': demanda_restante.copy()
    })

    # Calcular el costo total usando la matriz de costos original
    costo_total = np.sum(x * costos_originales)

    # Guardar los resultados en Excel en una sola hoja
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Crear una hoja en blanco para asegurarnos de que la hoja de resultados sea visible
        writer.book.create_sheet(title="TempSheet")

        resultado_final = pd.DataFrame()

        for idx, paso in enumerate(pasos, start=1):
            # Crear un DataFrame para mostrar la matriz de asignación con las ofertas y demandas
            df = pd.DataFrame(paso['matriz_asignacion'], columns=[f"Destino {j+1}" for j in range(n)], 
                              index=[f"Origen {i+1}" for i in range(m)])
            df['Oferta'] = paso['oferta_restante']
            
            demanda_df = pd.DataFrame([list(paso['demanda_restante'])], columns=[f"Destino {j+1}" for j in range(n)])
            demanda_df.index = ['Demanda']
            
            # Combinar la matriz de asignación con la demanda en un solo DataFrame
            df_final = pd.concat([df, demanda_df])
            
            # Agregar una fila indicando la iteración, ajustando el número de columnas
            df_iteracion = pd.DataFrame([[f"Iteración {idx}"] + [''] * (df_final.shape[1] - 1)], columns=df_final.columns)
            
            # Añadir un espacio vacío entre iteraciones para mayor claridad
            df_espacio = pd.DataFrame([[''] * df_final.shape[1]], columns=df_final.columns)
            
            # Combinar todo en el DataFrame final
            resultado_final = pd.concat([resultado_final, df_iteracion, df_final, df_espacio])
        
        # Finalmente escribir todo en una sola hoja de Excel
        resultado_final.to_excel(writer, sheet_name="Resultados", index=False)
        
        # Eliminar la hoja temporal
        del writer.book["TempSheet"]
    
    return x, costo_total

path = os.path.dirname(os.path.abspath(__file__))

archivo_entrada = os.path.join(path, 'tabla_transporte.xlsx')

archivo_salida = 'resultado_costos_minimos_en_una_hoja.xlsx'

# Leer los datos del archivo Excel
df = pd.read_excel(archivo_entrada, header=None)

# Extraer costos, oferta y demanda
costos = df.iloc[1:-1, 1:-1].values  # Matriz de costos sin encabezados
oferta = pd.to_numeric(df.iloc[1:-1, -1], errors='coerce').fillna(0).values
demanda = pd.to_numeric(df.iloc[-1, 1:-1], errors='coerce').fillna(0).values

if np.sum(oferta) != np.sum(demanda):
    diferencia = abs(np.sum(oferta) - np.sum(demanda))
    if np.sum(oferta) < np.sum(demanda):
        costos = np.vstack([costos, np.zeros(costos.shape[1])])
        oferta = np.append(oferta, diferencia)
    else:
        costos = np.hstack([costos, np.zeros((costos.shape[0], 1))])
        demanda = np.append(demanda, diferencia)

# Ejecutar el algoritmo y guardar los resultados en una sola hoja de Excel
solucion, costo_total = costos_minimos(costos, oferta, demanda, archivo_salida)

print(f"\nCosto total: {costo_total}")
