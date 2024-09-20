import numpy as np
import pandas as pd
from typing import List, Tuple

# Función para calcular el ciclo más corto (mantén esta parte igual)
def find_shortest_cycle(matrix: List[List[int]], start: Tuple[int, int]) -> List[Tuple[int, int]]:
    pass

# Función para aplicar el algoritmo MODI
def modi_method(cost_matrix: np.ndarray, supply: List[int], demand: List[int], initial_solution: np.ndarray, writer: pd.ExcelWriter, step: int) -> np.ndarray:
    rows, cols = cost_matrix.shape
    u = np.full(rows, None)
    v = np.full(cols, None)
    solution = initial_solution.copy()
    u[0] = 0
    
    # Calcular multiplicadores iniciales
    while None in u or None in v:
        for i in range(rows):
            for j in range(cols):
                if solution[i, j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost_matrix[i, j] - u[i]
                    elif u[i] is None and v[j] is not None:
                        u[i] = cost_matrix[i, j] - v[j]

    while True:
        reduced_costs = np.zeros_like(cost_matrix)
        for i in range(rows):
            for j in range(cols):
                if solution[i, j] == 0:
                    reduced_costs[i, j] = cost_matrix[i, j] - (u[i] + v[j])

        if np.all(reduced_costs >= 0):
            print("La solución es óptima.")
            break

        entering_variable = np.unravel_index(np.argmin(reduced_costs), reduced_costs.shape)
        cycle = find_shortest_cycle(solution.tolist(), entering_variable)

        if cycle:
            min_val = min(solution[cycle[i]] for i in range(1, len(cycle), 2) if solution[cycle[i]] > 0)

            for i, (x, y) in enumerate(cycle):
                if i % 2 == 0:
                    solution[x, y] += min_val
                else:
                    solution[x, y] -= min_val

            u.fill(None)
            v.fill(None)
            u[0] = 0
            while None in u or None in v:
                for i in range(rows):
                    for j in range(cols):
                        if solution[i, j] > 0:
                            if u[i] is not None and v[j] is None:
                                v[j] = cost_matrix[i, j] - u[i]
                            elif u[i] is None and v[j] is not None:
                                u[i] = cost_matrix[i, j] - v[j]

        # Guardar el paso actual en Excel solo como DataFrame
        step_df = pd.DataFrame(solution, columns=[f'Destino {i+1}' for i in range(cols)], index=[f'Origen {i+1}' for i in range(rows)])
        step_df['Suministro'] = supply
        demand_df = pd.DataFrame([demand], columns=[f'Destino {i+1}' for i in range(cols)], index=['Demanda'])
        full_df = pd.concat([step_df, demand_df], axis=0)

        # Escribir este paso en la hoja de Excel, una vez por iteración
        full_df.to_excel(writer, sheet_name=f'Paso {step}')
        
        step += 1

    return solution

# Leer la solución inicial desde el archivo Excel
def read_excel_file(file_path: str):
    df = pd.read_excel(file_path, index_col=0)
    cost_matrix = df.iloc[:-1, :-1].to_numpy()  # Matriz de costos
    supply = df.iloc[:-1, -1].to_numpy()  # Suministro
    demand = df.iloc[-1, :-1].to_numpy()  # Demanda
    initial_solution = df.iloc[:-1, :-1].to_numpy()  # Solución inicial (igual que la matriz de costos)
    return cost_matrix, supply, demand, initial_solution

# Guardar los resultados en Excel
def save_results(file_path: str, cost_matrix: np.ndarray, supply: np.ndarray, demand: np.ndarray, initial_solution: np.ndarray):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        step = 1
        modi_method(cost_matrix, supply, demand, initial_solution, writer, step)
        # Guardar el archivo al final, no en cada iteración
        writer.save()

# Ejemplo de uso
file_path = 'tabla_solucion.xlsx'  # Ruta del archivo de entrada
cost_matrix, supply, demand, initial_solution = read_excel_file(file_path)

output_file = 'resultado_optimizar.xlsx'
save_results(output_file, cost_matrix, supply, demand, initial_solution)
