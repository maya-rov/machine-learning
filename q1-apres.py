import numpy as np
import time
import math

np.random.seed(42)
matriz_np = np.random.normal(loc=50, scale=15, size=(1000, 10))
matriz_lista = matriz_np.tolist()  # Converte para lista padrão do Python

# --- 1. TESTE COM NUMPY ---
inicio_np = time.perf_counter()

medias = np.mean(matriz_np, axis=0)
desvios = np.std(matriz_np, axis=0)
maximos = np.max(matriz_np, axis=0)
minimos = np.min(matriz_np, axis=0)
filtrada_np = matriz_np[matriz_np[:, 0] > medias[0]]

tempo_np = (time.perf_counter() - inicio_np) * 1000

# --- 2. TESTE COM LOOPS PYTHON PURO ---
inicio_py = time.perf_counter()

n_linhas = len(matriz_lista)
n_colunas = len(matriz_lista[0])

medias_py, desvios_py, maximos_py, minimos_py = [], [], [], []

for j in range(n_colunas):
    soma = 0.0
    v_max = matriz_lista[0][j]
    v_min = matriz_lista[0][j]

    # Média, Máximo e Mínimo
    for i in range(n_linhas):
        val = matriz_lista[i][j]
        soma += val
        if val > v_max:
            v_max = val
        if val < v_min:
            v_min = val

    media = soma / n_linhas
    medias_py.append(media)
    maximos_py.append(v_max)
    minimos_py.append(v_min)

    # Desvio Padrão
    soma_quad = sum((matriz_lista[i][j] - media) ** 2 for i in range(n_linhas))
    desvios_py.append(math.sqrt(soma_quad / n_linhas))

# Filtro da primeira coluna
filtrada_py = [linha for linha in matriz_lista if linha[0] > medias_py[0]]

tempo_py = (time.perf_counter() - inicio_py) * 1000

print(f"Tempo NumPy:       {tempo_np:.4f} ms")
print(f"Tempo Python Puro: {tempo_py:.4f} ms")
print(f"O NumPy foi {tempo_py / tempo_np:.1f}x mais rápido.")
