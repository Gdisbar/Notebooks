data_UK_1h.index
# DatetimeIndex(['2010-12-01 08:00:00', '2010-12-01 09:00:00',
#                '2010-12-01 10:00:00', '2010-12-01 11:00:00',
#                '2010-12-01 12:00:00', '2010-12-01 13:00:00',
#                '2010-12-01 14:00:00', '2010-12-01 15:00:00',
#                '2010-12-01 16:00:00', '2010-12-01 17:00:00',
#                ...
#                '2011-12-08 16:00:00', '2011-12-08 17:00:00',
#                '2011-12-08 18:00:00', '2011-12-08 19:00:00',
#                '2011-12-08 20:00:00', '2011-12-09 09:00:00',
#                '2011-12-09 10:00:00', '2011-12-09 11:00:00',
#                '2011-12-09 12:00:00', '2011-12-09 13:00:00'],
#               dtype='datetime64[ns]', name='InvoiceDate_1h', length=2945, freq=Non

data_UK_1h.index[0] # Timestamp('2010-12-01 08:00:00')
data_UK_1h.index[-1] # Timestamp('2011-12-09 13:00:00')
data_UK_1h.index[-1]+timedelta(hours=1) # Timestamp('2011-12-09 14:00:00')
t # np.arrange(...).astype(datetime)
# array([datetime.datetime(2010, 12, 1, 8, 0),
#        datetime.datetime(2010, 12, 1, 9, 0),
#        datetime.datetime(2010, 12, 1, 10, 0),
#        datetime.datetime(2010, 12, 1, 11, 0),
#        datetime.datetime(2010, 12, 1, 12, 0)], dtype=object)

train_test['ds'].isin(data_UK_1h.index) 

# 0       True
# 1       True
# 2       True
# 3       True
# 4       True
#         ... 
# 8953    True
# 8954    True
# 8955    True
# 8956    True
# 8957    True
# Name: ds, Length: 8958, dtype: bool


train_test.head(5) 

# 	          ds 	           y 	     cap 	floor
# 0 	2010-12-01 08:00:00 	161.32 	    20000 	-2000
# 1 	2010-12-01 09:00:00 	1203.09 	20000 	-2000
# 2 	2010-12-01 10:00:00 	8490.18 	20000 	-2000
# 3 	2010-12-01 11:00:00 	3916.21 	20000 	-2000
# 4 	2010-12-01 12:00:00 	8149.77 	20000 	-2000

train_df.columns # Index(['ds', 'y', 'cap', 'floor'], dtype='object')
preds.columns 
# Index(['ds', 'trend', 'cap', 'yhat_lower', 'yhat_upper', 'trend_lower',
#        'trend_upper', 'Christmas Day', 'Christmas Day_lower',
#        'Christmas Day_upper', 'Christmas Day (Observed)',
#        'Christmas Day (Observed)_lower', 'Christmas Day (Observed)_upper',
#        'Columbus Day', 'Columbus Day_lower', 'Columbus Day_upper',
#        'Independence Day', 'Independence Day_lower', 'Independence Day_upper',
#        'Independence Day (Observed)', 'Independence Day (Observed)_lower',
#        'Independence Day (Observed)_upper', 'Labor Day', 'Labor Day_lower',
#        'Labor Day_upper', 'Martin Luther King Jr. Day',
#        'Martin Luther King Jr. Day_lower', 'Martin Luther King Jr. Day_upper',
#        'Memorial Day', 'Memorial Day_lower', 'Memorial Day_upper',
#        'New Year's Day', 'New Year's Day_lower', 'New Year's Day_upper',
#        'New Year's Day (Observed)', 'New Year's Day (Observed)_lower',
#        'New Year's Day (Observed)_upper', 'Thanksgiving', 'Thanksgiving_lower',
#        'Thanksgiving_upper', 'Veterans Day', 'Veterans Day_lower',
#        'Veterans Day_upper', 'Washington's Birthday',
#        'Washington's Birthday_lower', 'Washington's Birthday_upper', 'daily',
#        'daily_lower', 'daily_upper', 'holidays', 'holidays_lower',
#        'holidays_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
#        'multiplicative_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper',
#        'yearly', 'yearly_lower', 'yearly_upper', 'additive_terms',
#        'additive_terms_lower', 'additive_terms_upper', 'yhat'],
#       dtype='object')

preds.head(5)
#               ds 	          trend 	     cap 	yhat_lower 	    yhat_upper 	    trend_lower 	trend_upper 	Christmas Day 	Christmas Day_lower 	Christmas Day_upper 	... 	weekly 	weekly_lower 	weekly_upper 	yearly 	yearly_lower 	yearly_upper 	additive_terms 	additive_terms_lower 	additive_terms_upper 	yhat
# 0 	2011-11-20 23:00:00 	1570.287048 	20000 	-2043.477501 	1908.846554 	1570.287048 	1570.287048 	0.0 	0.0 	0.0 	... 	-0.845385 	-0.845385 	-0.845385 	0.067785 	0.067785 	0.067785 	0.0 	0.0 	0.0 	-50.942028
# 1 	2011-11-21 00:00:00 	1570.569843 	20000 	-2223.613542 	1877.785218 	1570.569843 	1570.569843 	0.0 	0.0 	0.0 	... 	-0.911898 	-0.911898 	-0.911898 	0.067814 	0.067814 	0.067814 	0.0 	0.0 	0.0 	-177.596016


test_df[['yhat', 'label', 'ds']].head(5)

#  	    yhat 	label 	ds
# 8511 	0.0 	true 	2011-11-20 23:00:00
# 8512 	0.0 	true 	2011-11-21 00:00:00
# 8513 	0.0 	true 	2011-11-21 01:00:00
# 8514 	0.0 	true 	2011-11-21 02:00:00
# 8515 	0.0 	true 	2011-11-21 03:00:00

preds[['yhat', 'label', 'ds']].head(5)
#  	          yhat 	    label 	ds
# 0 	-50.942028 	    pred 	2011-11-20 23:00:00
# 1 	-177.596016 	pred 	2011-11-21 00:00:00
# 2 	-252.467416 	pred 	2011-11-21 01:00:00
# 3 	-104.783331 	pred 	2011-11-21 02:00:00
# 4 	178.343148 	    pred 	2011-11-21 03:00:00

df_result.head(5)

#  	    yhat 	label 	ds
# 8511 	0.0 	true 	2011-11-20 23:00:00
# 8512 	0.0 	true 	2011-11-21 00:00:00
# 8513 	0.0 	true 	2011-11-21 01:00:00
# 8514 	0.0 	true 	2011-11-21 02:00:00
# 8515 	0.0 	true 	2011-11-21 03:00:00