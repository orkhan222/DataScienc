import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

file_path = 'Datasets/acs2015_county_data.csv'
data = pd.read_csv(file_path)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

outlier_indices = []
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    outlier_indices.extend(detect_outliers_iqr(data, column).index.tolist())

outlier_counts = Counter(outlier_indices)


n = 5  
rows_to_remove = [idx for idx, count in outlier_counts.items() if count > n]

data_cleaned = data.drop(rows_to_remove)


state_counts = data_cleaned['State'].value_counts()
top_states = state_counts[:10]
top_states['Other'] = state_counts[10:].sum()
plt.figure(figsize=(10, 10))
top_states.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Counties Among Top 10 States (and Others)')
plt.ylabel('')  
plt.show()


selected_variables = ['TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 'Income', 'IncomePerCap', 'Unemployment', 'Poverty']
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
axes = axes.flatten()  
for i, column in enumerate(selected_variables):
    data_cleaned[column].hist(ax=axes[i], bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlabel('Value')
plt.tight_layout()
plt.show()


scatter_configs = [
    {'x': 'TotalPop', 'y': 'Income', 'title': 'Total Population vs. Income'},
    {'x': 'Unemployment', 'y': 'Poverty', 'title': 'Unemployment vs. Poverty'},
    {'x': 'Hispanic', 'y': 'IncomePerCap', 'title': 'Hispanic Population (%) vs. Income Per Capita'}
]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for i, config in enumerate(scatter_configs):
    axes[i].scatter(data_cleaned[config['x']], data_cleaned[config['y']], color='blue', alpha=0.5)
    axes[i].set_title(config['title'])
    axes[i].set_xlabel(config['x'])
    axes[i].set_ylabel(config['y'])
plt.tight_layout()
plt.show()
