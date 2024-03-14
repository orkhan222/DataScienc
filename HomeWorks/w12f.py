import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


f = r"C:\Users\user\Desktop\DataAnalyst\Datasets\w12_demand.csv"
df = pd.read_csv(f)


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


df.drop(columns=df.columns[0], inplace=True)


df = df.asfreq('D').fillna(method='ffill')


df['day_of_week'] = df.index.dayofweek 
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  

cal = calendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())
df['is_holiday'] = df.index.isin(holidays).astype(int)


decompose_cols = ['Furniture', 'Office Supplies', 'Technology']
decomp_results = {}

for col in decompose_cols:
    decomp_results[col] = seasonal_decompose(df[col], model='additive', extrapolate_trend='freq')


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

for i, col in enumerate(decompose_cols):
    decomp_results[col].trend.plot(ax=axes[i], title=f'Trend in {col}')
    axes[i].set_ylabel('Demand')

plt.tight_layout()
plt.show()
