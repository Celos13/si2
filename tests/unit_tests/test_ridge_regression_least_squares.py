from si.io.csv_file import read_csv
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.model_selection.split import train_test_split

data = read_csv(r"C:\Users\filip\OneDrive\Attachments\Ambiente de Trabalho\SIB_gh\si2\datasets\cpu\cpu.csv", sep=",", features=True, label=True)

train, test = train_test_split(data, test_size=0.2, random_state=42)

model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)

model.fit(train)

print("MSE (test) =", model.score(test))

pred = model.predict(test)
pred[:5]
