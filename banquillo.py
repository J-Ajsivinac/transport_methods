from typing import List, Tuple
import numpy as np

# Función para encontrar el ciclo más corto (como proporcionada)
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
                shortest_cycle = path.copy()
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
                        return
                new_x, new_y = new_x + dx, new_y + dy
    
    if matrix[start[0]][start[1]] != 0:
        return []
    
    dfs(start[0], start[1], [start], "")
    
    return shortest_cycle if shortest_cycle else []

# Función Stepping-Stone para optimizar la solución inicial
def stepping_stone_method(cost_matrix: np.ndarray, solution_matrix: np.ndarray, supplies: List[int], demands: List[int]) -> Tuple[np.ndarray, int]:
    rows, cols = cost_matrix.shape
    total_cost = np.sum(solution_matrix * cost_matrix)  # Costo total inicial
    
    def calculate_path_cost(path: List[Tuple[int, int]]) -> int:
        path_cost = 0
        for idx, (i, j) in enumerate(path):
            if idx % 2 == 0:  # Celda en + (suma)
                path_cost += cost_matrix[i, j]
            else:  # Celda en - (resta)
                path_cost -= cost_matrix[i, j]
        return path_cost

    while True:
        # Buscar una celda no asignada (celda de agua) para iniciar un ciclo
        for i in range(rows):
            for j in range(cols):
                if solution_matrix[i, j] == 0:
                    path = find_shortest_cycle(solution_matrix.tolist(), (i, j))
                    if path:
                        path_cost = calculate_path_cost(path)
                        if path_cost < 0:  # El ciclo puede reducir el costo
                            # Actualizar las cantidades de acuerdo al ciclo
                            min_quantity = min(solution_matrix[path[k][0], path[k][1]] for k in range(1, len(path), 2))
                            for k, (x, y) in enumerate(path):
                                if k % 2 == 0:
                                    solution_matrix[x, y] += min_quantity
                                else:
                                    solution_matrix[x, y] -= min_quantity
                            # Recalcular el costo total
                            total_cost = np.sum(solution_matrix * cost_matrix)
                            break
            else:
                continue
            break
        else:
            # No se puede mejorar más la solución
            break

    return solution_matrix, total_cost

# Ejemplo de uso
cost_matrix = np.array([
    [80,100,50,60],
    [140,90,70,80],
    [200,200,210,120],
    [210,150,170,160]
])
solution_matrix = np.array([
    [130, 20, 0,0],
    [0,100,30,0],
    [0,0,70,10],
    [0,0,0,60],
])
supplies = [150,130,80,60]
demands = [130,120,100,70]

optimized_solution, optimized_cost = stepping_stone_method(cost_matrix, solution_matrix, supplies, demands)
print("Solución optimizada:\n", optimized_solution)
print("Costo optimizado:", optimized_cost)
