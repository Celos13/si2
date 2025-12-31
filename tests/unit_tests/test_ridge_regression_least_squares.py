from si.io.csv_file import read_csv
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.model_selection.split import train_test_split

# Lê o ficheiro cpu.csv a partir de um caminho absoluto
data = read_csv(
    r"C:\Users\filip\OneDrive\Attachments\Ambiente de Trabalho\SIB_gh\si2\datasets\cpu\cpu.csv",
    sep=",",
    features=True,
    label=True
)

# Divide o dataset em treino (80%) e teste (20%)
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Cria o modelo de Ridge Regression com λ=1.0 e com standardização das features
model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)

# Ajusta o modelo com os dados de treino
model.fit(train)

# Imprime o MSE no conjunto de teste usando o score do modelo
print("MSE (test) =", model.score(test))

# Faz previsões no conjunto de teste
pred = model.predict(test)
# Mostra as primeiras 5 previsões
pred[:5]
