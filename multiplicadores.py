import numpy as np

def stepping_stone_method(costs, allocation):
    m, n = costs.shape
    
    def find_cycle(i, j):
        visited = set()
        path = []
        
        def dfs(x, y, direction):
            if (x, y) in visited:
                return path
            visited.add((x, y))
            path.append((x, y))
            
            if direction == 'horizontal':
                for ny in range(n):
                    if ny != y and allocation[x, ny] > 0:
                        result = dfs(x, ny, 'vertical')
                        if result:
                            return result
            else:
                for nx in range(m):
                    if nx != x and allocation[nx, y] > 0:
                        result = dfs(nx, y, 'horizontal')
                        if result:
                            return result
            
            path.pop()
            visited.remove((x, y))
            return None
        
        for y in range(n):
            if y != j and allocation[i, y] > 0:
                result = dfs(i, y, 'vertical')
                if result:
                    return [result[k] for k in range(len(result)) if k % 2 == 0] + [(i, j)]
        return None

    while True:
        improvement = False
        for i in range(m):
            for j in range(n):
                if allocation[i, j] == 0:
                    cycle = find_cycle(i, j)
                    if cycle:
                        cost_diff = sum(costs[x, y] * (-1)**k for k, (x, y) in enumerate(cycle))
                        if cost_diff < 0:
                            min_allocation = min(allocation[x, y] for k, (x, y) in enumerate(cycle) if k % 2 == 1)
                            for k, (x, y) in enumerate(cycle):
                                allocation[x, y] += min_allocation * (-1)**k
                            improvement = True
                            break
            if improvement:
                break
        if not improvement:
            break
    
    return allocation
costs = np.array([
    [8, 10, 5],
    [14, 9, 7],
    [20, 20, 20]
])

initial_allocation = np.array([
    [10, 0, 10],
    [5, 15, 0 ],
    [10, 0, 0]
])

print("Asignación inicial:")
print(initial_allocation)
print("Costo inicial:", np.sum(costs * initial_allocation))

optimized_allocation = stepping_stone_method(costs, initial_allocation)
print("\nAsignación optimizada:")
print(optimized_allocation)

total_cost = np.sum(costs * optimized_allocation)
print(f"Costo total optimizado: {total_cost}")

if np.array_equal(initial_allocation, optimized_allocation):
    print("\nAdvertencia: La asignación no ha cambiado. Esto puede indicar que la solución inicial ya era óptima o que hay un problema en el algoritmo.")
else:
    print("\nLa asignación ha sido optimizada exitosamente.")