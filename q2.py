from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Carregar o dataset Iris diretamente dos datasets embutidos no Scikit-Learn
iris = load_iris()

# 2. Separar os dados de entrada (features, X) dos rótulos de saída (targets, y)
X = iris.data
y = iris.target

# 3. Dividir os dados em 70% para treinamento e 30% para teste (Hold-out simples)
# random_state=42 garante reprodutibilidade; stratify=y mantém a proporção das 3 espécies
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Total de amostras: {X.shape[0]}")
print(f"Amostras de treino (70%): {X_train.shape[0]}")
print(f"Amostras de teste (30%): {X_test.shape[0]}\n")

# 4. Instanciar, treinar (fit) e testar (predict) dois algoritmos diferentes

# --- Modelo 1: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# --- Modelo 2: Support Vector Machine (SVM) ---
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Exibição e comparação dos resultados
print("=== Resultados: Random Forest ===")
print(f"Acurácia: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print(classification_report(y_test, rf_pred, target_names=iris.target_names))

print("=== Resultados: Support Vector Machine (SVM) ===")
print(f"Acurácia: {accuracy_score(y_test, svm_pred) * 100:.2f}%")
print(classification_report(y_test, svm_pred, target_names=iris.target_names))
