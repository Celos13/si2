from si.io.csv_file import read_csv
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.model_selection.split import train_test_split

# carregar dados
data = read_csv("datasets/cpu.csv", sep=",", features=True, label=True)

# split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# modelo
model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)

# train
model.fit(train)

# score
print("MSE (test) =", model.score(test))

# previsões
pred = model.predict(test)
pred[:5]
