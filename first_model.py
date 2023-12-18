import pandas as pd

from sklearn.tree import DecisionTreeRegressor

iowa_file_path = './data/home.csv'

home_data = pd.read_csv(iowa_file_path)


feature_names = [
"LotArea",
"YearBuilt",
"1stFlrSF",
"2ndFlrSF",
"FullBath",
"BedroomAbvGr",
"TotRmsAbvGrd"]
y = home_data['SalePrice']
X = home_data[feature_names]

# print(X.describe())
# print(X.head())

melbourne_model = DecisionTreeRegressor(random_state=1) 

iowa_model = melbourne_model.fit(X, y)

predictions = iowa_model.predict(X)

print(y.head())
print(predictions)
