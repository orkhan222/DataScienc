# Task4
import pandas as pd

df = pd.read_csv('Datasets/simple_dataset.csv')

rate = 0.18
df['Price'] = df['Price'].apply(lambda x: x * rate)

# upper
df['Item'] = df['Item'].apply(lambda x: x.upper())

# Rating>=4
df_filtered = df[df['Rating'] >= 4]

# category
category = df.groupby('Category')['Price'].mean()
print(df)
print('='*80)

print(df_filtered)
print('='*80)

print(category)
print('='*80)