import pandas as pd

iowa_file_path = './data/home.csv'

home_data = pd.read_csv(iowa_file_path)

avg_lot_size = home_data['LotArea'].mean().round()
newest_home_age = 2023 - home_data['YearBuilt'].max()

print(home_data)
print(avg_lot_size,)
print(newest_home_age,)
