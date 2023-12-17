import pandas as pd

iowa_file_path = './data/home.csv'
home_features = ['TotRmsAbvGrd', 'TotalBsmtSF', 'BedroomAbvGr', 'YrSold', 'SalePrice']

home_data = pd.read_csv(iowa_file_path)

avg_lot_size = home_data['LotArea'].mean().round()
newest_home_age = 2023 - home_data['YearBuilt'].max()

## Will show titles off data set
# print(home_data.columns)

## Will show a bunch of selected titles
# print(home_data[home_features].describe().round())

## Will describe your model with min,avg,max parameters
# print(home_data.describe())

## Will print head(first ~7 rows) of your model 
# print(home_data.head())

# print(home_data)
# print(avg_lot_size)
# print(newest_home_age)




