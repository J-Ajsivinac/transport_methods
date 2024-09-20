from typing import List, Tuple

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

# Ejemplos de uso
matriz1 = [
    [130, 20, 0, 0],
    [0, 100, 30, 0],
    [0, 0, 70, 10],
    [0, 0, 0, 60]
]

matriz2 = [
    [500, 0, 0, 0],
    [300, 1500, 800, 0],
    [200, 0, 0, 200],
]

# Función auxiliar para imprimir resultados
def print_result(example: int, start: Tuple[int, int], cycle: List[Tuple[int, int]]):
    print(f"Ejemplo {example}:")
    print(f"Inicio = {start}")
    print(f"Ciclo = {' -> '.join(map(str, cycle))}")
    print(cycle)
    print()

# Ejemplo 1
inicio1 = (0, 2)
ciclo1 = find_shortest_cycle(matriz1, inicio1)
print_result(1, inicio1, ciclo1)

# Ejemplo 2
inicio2 = (2, 0)
ciclo2 = find_shortest_cycle(matriz1, inicio2)
print_result(2, inicio2, ciclo2)

# Ejemplo 3
inicio3 = (1, 3)
ciclo3 = find_shortest_cycle(matriz2, inicio3)
print_result(3, inicio3, ciclo3)
