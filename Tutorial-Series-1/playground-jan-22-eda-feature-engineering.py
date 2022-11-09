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

mod = sm.tsa.statespace.SARIMAX(...)
results = mod.fit(disp=False)
display(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

import datetime as dt

def get_ts_dict(df):
	# for country_list ,store_list,product_list -> time_series_dict[key] = selected_pd
	# here key is created using combining 3 elements from the above list

ts_dict = get_ts_dict(train)
key_list = list(ts_dict.keys())

def ts_plot(ts_dict, key_list, figsize=(24, 24)):
    # for every row & col use index to create df & row,col for axes
    # key -> key_list[index] --> 
    # df = ts_dict[key]
    # lineplot -> df,'date','num_sold'
	# x_label -> date_range
	# axvline -> df['date'].dt.year
            

from scipy.signal import periodogram

def plot_periodogram(ts, detrend='linear', ax=None, title=''):
	fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
	# fs -> ratio of '1Y' & '1D' in Date scale
    #freqencies, spectrum = periodogram(...)
    

import holidays
import datetime

def feature_eng(df):
    #### Date
    #### Peak
    #### Till The Next Holiday
    def get_country_holidays(country, years_list):
    	# festivities -> get each country holiday list
    	# festivities_df -> df using festivities
    	# festivities_df['date'] -> convert datetime() format
    	# festivities_df -> update for any specific festival without holiday
    	# additional_dates -> common to all not included in holiday calendar
        # additional_festivities_df -> add additional_dates to festivities_df
        # append additional_festivities_df to festivities_df + sort by 'date'
        

    def days_till_next_holiday(country, date):
    	# country_holidays_dates -> get complete customized list of holidays for each country
    	# return min number of days remaining for next holiday
        
    # apply on df['days_till_next_holiday']
    #### Seasonality
    # date_range -> we're assuming season transition to be 60 days
    # use fourier on this dataset -> in_sample(),lag(),drop(),merge()
    
    #### GDP
    # gdp_melt_df -> get year wise gdp value for list of countries
    # df -> merge gdp_melt_df
    
    #### GDP Percentage Change Between Years
    # same like above but here gdp% is calculated
    # drop unnecessary cols
