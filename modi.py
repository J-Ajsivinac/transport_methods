import numpy as np
from typing import List, Tuple

# Función para calcular el ciclo más corto que has proporcionado
def find_shortest_cycle(matrix: List[List[int]], start: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = len(matrix), len(matrix[0])
    shortest_cycle = None
    shortest_length = float('inf')
    
    def is_valid(x: int, y: int) -> bool:
        return 0 <= x < rows and 0 <= y < cols
    
    def has_vertical_horizontal_moves(path: List[Tuple[int, int]]) -> bool:
        vertical = any(path[i][0] != path[i + 1][0] for i in range(len(path) - 1))
        horizontal = any(path[i][1] != path[i + 1][1] for i in range(len(path) - 1))
        return vertical and horizontal
    
    def dfs(x: int, y: int, path: List[Tuple[int, int]], last_dir: str):
        nonlocal shortest_cycle, shortest_length
        
        if len(path) > 2 and (x, y) == start:
            if len(path) < shortest_length and has_vertical_horizontal_moves(path):
                shortest_cycle = path.copy()  # No agregar de nuevo el punto de inicio
                shortest_length = len(path)
            return
        
        if len(path) >= shortest_length:
            return
        
        directions = [('up', -1, 0), ('left', 0, -1), ('down', 1, 0), ('right', 0, 1)]
        for dir_name, dx, dy in directions:
            if dir_name == last_dir:
                continue
            
            new_x, new_y = x + dx, y + dy
            while is_valid(new_x, new_y):
                if matrix[new_x][new_y] != 0 or (new_x, new_y) == start:
                    if (new_x, new_y) not in path or ((new_x, new_y) == start and len(path) > 2):
                        if (new_x, new_y) == start:
                            dfs(new_x, new_y, path, dir_name)
                        else:
                            path.append((new_x, new_y))
                            dfs(new_x, new_y, path, dir_name)
                            path.pop()
                    if (new_x, new_y) == start:
                        return  # Terminamos la exploración en esta dirección si llegamos al inicio
                new_x, new_y = new_x + dx, new_y + dy
    
    if matrix[start[0]][start[1]] != 0:
        return []  # El punto de inicio debe ser una celda de agua
    
    dfs(start[0], start[1], [start], "")
    
    return shortest_cycle if shortest_cycle else []

# Función para aplicar el algoritmo MODI
def modi_method(cost_matrix: np.ndarray, supply: List[int], demand: List[int], initial_solution: np.ndarray) -> np.ndarray:
    rows, cols = cost_matrix.shape
    u = np.full(rows, None)  # Multiplicadores de fila
    v = np.full(cols, None)  # Multiplicadores de columna
    
    # Inicializar la solución actual
    solution = initial_solution.copy()
    
    # Fijar u[0] = 0
    u[0] = 0  
    # Calcular los multiplicadores iniciales
    while None in u or None in v:
        for i in range(rows):
            for j in range(cols):
                if (i, j) in [(x, y) for x, y in np.ndindex(solution.shape) if solution[x, y] > 0]:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost_matrix[i, j] - u[i]
                    elif u[i] is None and v[j] is not None:
                        u[i] = cost_matrix[i, j] - v[j]

    while True:
        # Calcular costos reducidos
        reduced_costs = np.zeros_like(cost_matrix)
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in [(x, y) for x, y in np.ndindex(solution.shape) if solution[x, y] > 0]:
                    reduced_costs[i, j] = cost_matrix[i, j] -(u[i] + v[j])

        # Verificar optimalidad
        if np.all(reduced_costs >= 0):
            print("La solución es óptima.")
            break

        # Encontrar el costo reducido mínimo
        entering_variable = np.unravel_index(np.argmin(reduced_costs), reduced_costs.shape)
        print(f"inicio: {entering_variable}")
        # Encontrar ciclo cerrado
        cycle = find_shortest_cycle(solution.tolist(), entering_variable)
        print("-----")
        print(f"cycle: {cycle}")
        print("-----")
        
        if cycle:
            # Actualizar la solución siguiendo el ciclo cerrado
            min_val = min(solution[cycle[i]] for i in range(1, len(cycle), 2) if solution[cycle[i]] > 0)

            for i, (x, y) in enumerate(cycle):
                if i % 2 == 0:
                    solution[x, y] += min_val
                else:
                    solution[x, y] -= min_val

            # Recalcular los multiplicadores después de los ajustes
            u.fill(None)  # Reiniciar u
            v.fill(None)  # Reiniciar v
            u[0] = 0  # Fijar de nuevo u[0]

            while None in u or None in v:
                for i in range(rows):
                    for j in range(cols):
                        if (i, j) in [(x, y) for x, y in np.ndindex(solution.shape) if solution[x, y] > 0]:
                            if u[i] is not None and v[j] is None:
                                v[j] = cost_matrix[i, j] - u[i]
                            elif u[i] is None and v[j] is not None:
                                u[i] = cost_matrix[i, j] - v[j]
            print("Solución actual:")
            print(solution)

    return solution

# Ejemplo de uso
cost_matrix = np.array([
    [8,5,4,3],
    [3,5,2,1],
    [3,2,1,0]
])

supply = np.array([500,2600,400])  # Suministro en los orígenes
demand = np.array([1000,1500,800,200])    # Demanda en los destinos

# Solución inicial como matriz
initial_solution = np.array([
    [500, 0, 0,0],
    [500, 1500, 600,0],
    [0,0,200,200]
])

# Aplicar el algoritmo MODI
optimal_solution = modi_method(cost_matrix, supply, demand, initial_solution)
print("Solución optimizada:")
print(optimal_solution)

# Resultado de Costos
print("Costo total:", np.sum(optimal_solution * cost_matrix))
