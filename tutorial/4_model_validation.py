import pandas as pd

# Load data
melbourne_file_path = '../data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]
print('--------------------')
print(X.head())
print('--------------------')


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# Once we have a model, here is how we calculate the mean absolute error (**MAE**):
from sklearn.metrics import mean_absolute_error

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)

print(y)
print(mean_absolute_error(val_y, val_predictions))
