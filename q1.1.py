import numpy as np

# Definindo uma semente (seed) para reprodutibilidade dos resultados
rng = np.random.default_rng(seed=42)

# 1. Gerar duas matrizes aleatórias grandes (1000 linhas x 10 colunas)
# Matriz A: distribuição normal padrão (média ~0, desvio padrão ~1)
matriz_a = rng.standard_normal((1000, 10))

# Matriz B: distribuição uniforme entre 0 e 100
matriz_b = rng.uniform(low=0.0, high=100.0, size=(1000, 10))

print(f"Formato da Matriz A: {matriz_a.shape}")
print(f"Formato da Matriz B: {matriz_b.shape}\n")


# 2. Calcular estatísticas descritivas de cada coluna (axis=0 opera verticalmente por coluna)
def exibir_estatisticas(matriz, nome):
  medias = np.mean(matriz, axis=0)
  desvios = np.std(matriz, axis=0)
  maximos = np.max(matriz, axis=0)
  minimos = np.min(matriz, axis=0)

  print(f"--- Estatísticas por Coluna ({nome}) ---")
  for col in range(matriz.shape[1]):
    print(
        f"Coluna {col:2d} | Média: {medias[col]:6.2f} | "
        f"Desvio Padrão: {desvios[col]:5.2f} | "
        f"Mín: {minimos[col]:6.2f} | Máx: {maximos[col]:6.2f}"
    )
  print()
  return medias


medias_a = exibir_estatisticas(matriz_a, "Matriz A - Normal")
medias_b = exibir_estatisticas(matriz_b, "Matriz B - Uniforme")

# 3. Filtrar a matriz: linhas onde o valor da 1ª coluna (índice 0) > média da 1ª coluna
media_col0_a = medias_a[0]
condicao_a = matriz_a[:, 0] > media_col0_a
matriz_a_filtrada = matriz_a[condicao_a]

media_col0_b = medias_b[0]
condicao_b = matriz_b[:, 0] > media_col0_b
matriz_b_filtrada = matriz_b[condicao_b]

print("--- Resultados da Filtragem ---")
print(
    f"Matriz A: média da 1ª coluna = {media_col0_a:.4f} | "
    f"Linhas restantes: {matriz_a_filtrada.shape[0]} de {matriz_a.shape[0]}"
)
print("Primeiras 5 linhas filtradas da Matriz A (mostrando 4 primeiras colunas):")
print(np.round(matriz_a_filtrada[:5, :4], 3), "\n")

print(
    f"Matriz B: média da 1ª coluna = {media_col0_b:.4f} | "
    f"Linhas restantes: {matriz_b_filtrada.shape[0]} de {matriz_b.shape[0]}"
)
print("Primeiras 5 linhas filtradas da Matriz B (mostrando 4 primeiras colunas):")
print(np.round(matriz_b_filtrada[:5, :4], 3))
