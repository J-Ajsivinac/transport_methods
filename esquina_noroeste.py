import numpy as np

def esquina_noroeste(oferta, demanda):
    m, n = len(oferta), len(demanda)
    x = np.zeros((m, n))
    i, j = 0, 0
    
    while i < m and j < n:
        cantidad = min(oferta[i], demanda[j])
        x[i, j] = cantidad
        oferta[i] -= cantidad
        demanda[j] -= cantidad
        
        if oferta[i] == 0:
            i += 1
        if demanda[j] == 0:
            j += 1
    
    return x

def agregar_variables_ficticias(oferta, demanda, costos):
    total_oferta = sum(oferta)
    total_demanda = sum(demanda)
    
    if total_oferta > total_demanda:
        demanda.append(total_oferta - total_demanda)
        for fila in costos:
            fila.append(0)  # Costo 0 para la demanda ficticia
    elif total_demanda > total_oferta:
        oferta.append(total_demanda - total_oferta)
        costos.append([0] * len(demanda))  # Costo 0 para la oferta ficticia
    
    return oferta, demanda, costos

def calcular_costo_total(solucion, costos):
    return np.sum(solucion * costos)

def resolver_problema_transporte(oferta, demanda, costos):
    oferta, demanda, costos = agregar_variables_ficticias(oferta, demanda, costos)
    solucion = esquina_noroeste(oferta.copy(), demanda.copy())
    costo_total = calcular_costo_total(solucion, np.array(costos))
    return solucion, costo_total

# Ejemplo de uso
oferta = [20, 25, 35]
demanda = [10,12,14,16,18]
costos = [
    [42,32,33,39,36],
    [34,36,37,32,37],
    [38,31,40,35,35]
]

solucion, costo_total = resolver_problema_transporte(oferta, demanda, costos)
print("Solución:")
print(solucion)
print("\nCosto total:", costo_total)