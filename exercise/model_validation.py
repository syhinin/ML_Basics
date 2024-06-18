import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../data/home.csv'

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

# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# Calculate the Mean Absolute Error in Validation Data
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)


