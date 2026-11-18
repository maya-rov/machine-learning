import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Carregar os dados como DataFrame do Pandas
iris_bunch = load_iris(as_frame=True)
df = iris_bunch.frame

# 2. Substituir os números (0, 1, 2) pelos nomes das espécies para a legenda
df["espécie"] = df["target"].map(dict(enumerate(iris_bunch.target_names)))
df = df.drop(columns=["target"])

# 3. Configurar o estilo visual e gerar a matriz de dispersão
sns.set_theme(style="ticks")
sns.pairplot(df, hue="espécie", palette="viridis", markers=["o", "s", "D"])

# 4. Exibir o gráfico
plt.suptitle("Matriz de Dispersão - Dataset Iris", y=1.02)
plt.show()

