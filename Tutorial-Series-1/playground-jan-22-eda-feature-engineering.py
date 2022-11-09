# Feature Engineering
# divide 'date' to datetime() object using pd.to_datetime() now  get year,month,
# day,dayofweek,weekday,weekofyear etc

# EDA
train.groupby('country').agg({'num_sold':'sum'}) 
# unstack() is used to add a new col to dataframe
# Correlations against datetime object -> 
train.loc[train['product']=='Kaggle Mug', feature_list].corr().iloc[-1:, :]
# p-value 
pearsonr(train.loc[train['product']=='Kaggle Mug','num_sold'], train.loc[condition_row,c])
# Distributions
from pmdarima import auto_arima
import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(train.loc[train['product']=='Kaggle Mug','num_sold'],
                                order=(1,1,1),
                                seasonal_order=(1, 1, 1, 7))
results = mod.fit(disp=False)
display(results.summary().tables[1])

sns.set(font_scale=1.5)
results.plot_diagnostics(figsize=(16, 8))
plt.show()