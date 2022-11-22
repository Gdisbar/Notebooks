Patterns
-----------
X[t]=T[t]+S[t]+C[t]+I[t]
T[t]=Trend,S[t]=seasonal,C[t]=cyclical,I[t]= irregular/residual

# Seasonal -> It is observed when there is a distinct repeated pattern 
# observed between regular intervals due to seasonal 
# factors: annual, monthly or weekly
The additive model is Y[t] = T[t] + S[t] + I[t]
The multiplicative model is Y[t] = T[t] * S[t] * I[t] where
Y[t] = T[t] * S[t] * I[t] ~ =log(T[t])+log(S[t])+log(I[t])

dcmp = sm.tsa.seasonal_decompose(data_col,period=12,model='multiplicative')

# Dependence -> if past values of a series carry some information about 
# the future behavior -> autocorrelation and partial autocorrelation

# ACF at lag k measures a linear dependence between X[t] and X[t+k], 
# while PACF captures the dependence between those values correcting 
# for all the intermediate effects.

Mean function of time series: μ[t]=E[X[t]]

Autocovariance function of a time series: 
γ(s,t)=Cov(X[s],X[t])=E[X[s]X[t]]−E[X[s]]E[X[t]]

which leads to the following definitions of ACF / PACF:

Autocorrelation: 
ρ(u,t+u)=Cor(X[u],X[t+u])=Cov(X[t],X[t+u])/Var(X[t])Var(X[t+u])

Partial autocorrelation: ϕ(u)=Cor(X[t],X[t+u]|X[t+1],…,X[t+u−1])


Stationary -> 
--------------
# white noise series (T[t] & S[t] can''t be stationary but C[t]
# can sometime be) - Algorithms are likely to yield better predictions if 
# we apply them to stationary processes, because we do not need to worry 
# about e.g. concept drift between our training and test sets.

plot_acf(xseries['noise'], lags = 25) # coverting to stationary series
plot_pacf(xseries['noise'], lags = 25)

# checking if the series is stationary - mean,variarnce,autocorrelation

X = series.passengers.values
split =  int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean() 
var1, var2 = X1.var(), X2.var()
autocorr1,autocorr2 = acf(X1, nlags=10),acf(X2, nlags=10)

# more strict criteria for stationary

F[X](X[t1],…,X[tn])=F[X](X[t1+τ],…,X[tn+τ]),F[X]=cdf,{X[t]}=stocastic process

a process {X[t]} is weakly stationary if it satisfies:

μ[t]=μ[t+τ] -> constant mean
γ(s,t)=γ(s−t) -> covariance only depends on distance in time between variables
Var(X[t])<∞

autocorrelation : ρ(u)=γ(u)/γ(0)
# converting to stationary
differencing the series
taking the log of the series
power transforms

Differencing -> 
----------------
# stabilize the mean of a time series by removing changes 
# in the level of a time series using lag -> differencing at lag=1 is best

Lag operator of order d: ∇[d]X[t]=X[t]−X[t−d] 

Tests for stationarity
---------------------------------
# Augmented Dickey-Fuller (ADF)
# Kwiatkowski–Phillips–Schmidt–Shin (KPSS)
# Philips-Perron (PP)

ADF test is a unit root test. It determines how strongly a time series 
is defined by a trend. 
H0:time series can be represented by a unit root that is not stationary.
above α: Accepts H0, the data has a unit root and is non-stationary.

# We can decompose the series to check the components one by one: 
# which parts are responsible for the non-stationary behavior?

# skip missing value at start as adfuller() doesn't handle that
adf_stat,p_val = adfuller(decomposition.trend[10:-10]) 
# ADF Statistic: 1.025948 
# p-value: 0.994532    -> H0 not rejected
adf_stat,p_val = adfuller(decomposition.seasonal[10:-10]) 
# ADF Statistic: -7873162363427143.000000
# p-value: 0.000000 -> H0 rejected
adf_stat,p_val = adfuller(decomposition.resid[10:-10]) 
# ADF Statistic: -5.029170
# p-value: 0.000019

# ADF test check for a very specific form of non-stationarity, 
# namely variation in the presence of a linear trend (existence of 
# a single unit root) - while the seasonal component is clearly not 
# stationary (see graph above), it is a qualitatively different kind 
# of behavior.


series['passengers2'] = np.log(series['passengers']) #log transformation
adf_stat,p_val = adfuller(series.passengers2)
# p-value: 0.422367 -> H0 not rejected,correct transformation
series['passengers3'] = series['passengers'].diff() # get rid of the trend
adf_stat,p_val = adfuller(series.passengers3[10:])
# The augmented Dickey–Fuller (ADF) statistic, used in the test, 
# is a negative number. The more negative it is, the stronger the 
# rejection of the hypothesis that there is a unit root at some level 
# of confidence.
# ADF Statistic: -2.830154 -> amplification increased from 1.025948 
# p-value: 0.054094
# applying mor than 1 transformation + confirm using plot
series['passengers4'] = series['passengers'].apply(np.log).diff()
# ADF Statistic: -3.086110 -> now it's stationary
# p-value: 0.027598 -> below α
plot_acf(series['passengers4'][10:], lags = 10)

Exponential smoothing
===========================
# exponential moving average (EMA) assigns exponentially decreasing 
# weights over time. The functions as a low-pass filter that removes 
# high-frequency noise (and can be formulated as a special case of a 
# more general problem of recursive filtering).

Simple Exponential Smoothing - seasonal pattern/no clear trend
------------------------------------------------------------------
# Brown method is defined by the relationship:
S[t]=αX[t]+(1−α)S[t−1] where α ∈(0,1) or equivalently:
S[t]=S[t−1]+α(X[t]−S[t−1])
α = how quickly we will "forget" the last available true observation.


# With the setup of the above equation, we have the following form of a 
# long term forecast: which means simply that out of sample, our 
# forecast is equal to the most recent value of the smoothed series. 
X[t+h]=S[t] 

for alpha_sm in [0.2 , 0.5, 0.9]: # higher value means better overlap
	df.plot.line()
	fit1 = SimpleExpSmoothing(df).fit(smoothing_level = alpha_sm,optimized=False)
	fcast1 = fit1.forecast(12).rename('alpha = ' + str(alpha_sm))
	fcast1.plot(marker='o', color='red', legend=True)
	fit1.fittedvalues.plot(  color='red')
	plt.plot()

# does small α mean heavy smoothing or hardly any at all? 
# The idea that the coefficient closer to 1 means less smoothing is 
# merely a convention.

Well beyound the datapoints it can''t predict the future value correctly.

Double Exponential Smoothing - seasonal + trend component
-----------------------------------------------------------
# Holt Method is defined by:
S[t]=αX[t]+(1−α)(S[t−1]+b[t−1])
b[t]=β(S[t]−S[t−1])+(1−β)b[t−1] where S[1]=X[1],b[1]=X[1]−X[0],α,β ∈ (0,1)
The forecast h steps ahead is defined : X[t+h]=S[t]+hb[t]
# The forecast function is no longer flat but trending: h-step-ahead 
# forecast is equal to the last 
# estimated level plus h times the last estimated trend value.

df.plot.line()

fit1 = Holt(df).fit(smoothing_level=0.5,smoothing_slope=0.5,optimized=False)
fcast1 = fit1.forecast(12).rename("Holt's linear trend")
fit1.fittedvalues.plot(color='red')
fcast1.plot(color='red', legend=True)

plt.show()

# However, it is simply an extrapolation of the most recent 
# (smoothed) trend in the data which means we can expect the 
# forecast to turn negative shortly. This is suspicious in general, 
# and clearly renders the forecast unusable in the domain context.


Triple Exponential Smoothing - smoothed seasonal+trend 
-------------------------------------------------------
# smoothed seasonal component = seasonal+period, Holt-Winters, is defined by:

S[t]=α(X[t]−c[t−L])+(1−α)(S[t−1]+b[t−1])
b[t]=β(S[t]−S[t−1])+(1−β)b[t−1]
c[t]=γ(X[t]−S[t−1]−b[t−1])+(1−γ)c[t−L] with α,β,γ ∈ (0,1)

# The most important addition is the seasonal component to explain 
# repeated variations around intercept and trend, and it will be specified 
# by the period. For each observation in the season, there is a separate 
# component; for example, if the length of the season is 7 days 
# (a weekly seasonality), we will have 7 seasonal components, one for 
# each day of the week. An obvious, yet worth repeating caveat: it makes 
# sense to estimate seasonality with period L
# only if your sample size is bigger than 2L

The forecast h steps ahead is defined by : X[t+h]=S[t]+hb[t]+c[t−L+h] % L


df.plot.line()
fit1 = ExponentialSmoothing(df,seasonal_periods=12,trend='add',seasonal='add')
fit1 = fit1.fit(smoothing_level=0.5,use_boxcox=True)
fit1.fittedvalues.plot(color='red')
fit1.forecast(12).rename("Holt-Winters smoothing").plot(color='red', legend=True)
-----------------------------------------------------------------------
Seasonal               |  N(None) | A(Additive) | M(Multiplicative)
Trend                  |          |             |
-----------------------|-----------------------------------------------
N (None)			   | (N,N)    | (N,N)       |   (N,M)
-----------------------------------------------------------------------
A(Additive)		  	   | (A,N)    | (N,A)       |   (A,M)
----------------------------------------------------------------------
A[d](Additive damped)  | (A[d],N) | (A[d],A)    |   (A[d],M)
------------------------------------------------------------------------


# (N,N) is simple exponential smoothing
# (A,N) is Holt's linear trend method
# (A,A) corresponds to additive Holt-Winters method



Anomaly detection
-------------------
# select a window size w
# calculate rolling mean / standard deviation with window w
# demean and normalize by sd: 

Z[t]= ∣X[t]−X[m]|/σ[m]
# Z-score measures number of sd away from mean ⟹ values above 3 
# indicate extremely unlikely realization

# pick a window size 
window_size = 25

# calculate rolling mean and standard deviation
xroll = series['value'].rolling(window_size)
series['mean_roll'] = xroll.mean()
series['sd_roll'] = xroll.std()

# calculate the Z-score
series['zscore'] = np.abs((series['value']-series['mean_roll'])/series['sd_roll'])
series['zscore'].plot()
# check which observations are out of range
series.loc[series['zscore'] > 3][['timestamp', 'value']]
# split into train-test
cutoff_date = '2005-12-31'
xtrain, xvalid  = df.loc[df.index <= cutoff_date], df.loc[df.index > cutoff_date]
# smoothing
fit1 = ExponentialSmoothing(xtrain['value'].values, seasonal_periods=12, trend='mul', seasonal='mul')
fit1 = fit1.fit(use_boxcox=True)
# check parameters
fit1.params_formatted
# What do the residuals look like?
prediction = fit1.forecast(len(xvalid)).copy()
xresiduals = xvalid['value'] - prediction
plot_acf(xresiduals, lags = 25)
plot_pacf(xresiduals, lags = 25)

# The behavior of ACF / PACF (statistically significant autocorrelations) 
# suggests that there is some first- and second-order dependence that 
# our Holt-Winter model cannot capture.
xvalid['prediction'] = prediction
xvalid.plot()

# As you can see from the graph above, the model is doing a 
# decent job for the first few years in the sample, but starts 
# to overestimate the consumption afterwards - indicating perhaps 
# a change in the nature of the trend (which would be consistent 
# with the ACF pattern above). This confirms the intuition that 
# there are aspects of the data generating process that are not 
# adequately captured by our three prameter model - but for 
# something that simple, you can make a solid case it is acceptable.
======================
TS-1b: Prophet       |
======================
𝑋𝑡=𝑇𝑡+𝑆𝑡+𝐻𝑡+𝜖𝑡 where
𝑇𝑡 : trend component
𝑆𝑡 : seasonal component (weekly, yearly)
𝐻𝑡 : deterministic irregular component (holidays)
𝜖𝑡 : noise 

Generalized Additive Models (GAM)
------------------------------------
The core mathematical idea behind Prophet is the Kolmogorov-Arnold 
representation theorem, which states that multivariate function could be 
represented as sums and compositions of univariate functions:

𝑓(𝑥1,…,𝑥𝑛)=∑(0,2𝑛)Φ𝑞(∑𝑝=(1,𝑛)𝜙(𝑞,𝑝)(𝑥(𝑝)))

The theorem has no constructive proof suitable for modeling ⟹ simplification 
is necessary:

𝑓(𝑥1,…,𝑥𝑛)=Φ(∑(1,𝑛)𝜙𝑝(𝑥(𝑝)))

where Φ is a smooth monotonic function. This equation gives a 
general representation of GAM models and a familiar variant of this 
approach is the class of Generalized Linear Models :

Φ−1[E(𝑌)]=𝛽0+𝑓1(𝑥1)+𝑓2(𝑥2)+⋯+𝑓𝑚(𝑥𝑚).

The smooth functions in the context of Prophet are the trend, 
seasonal and holiday components - we can isolate each individual function 
and evaluate its effect in prediction, which makes such models easier to 
interpret. we estimate through backfitting algorithm →

convergence
The prophetic core

So how does that work in practice? We take a GAM-style decomposition as 
our starting point:

𝑋𝑡=𝑇(𝑡)+𝑆(𝑡)+𝐻(𝑡)+𝜖𝑡


Trend model
-------------------
The Prophet library implements two possible trend models.
-------------------------------------------------------------
Linear Trend
-------------------
The first, default trend model is a simple Piecewise Linear Model with a 
constant rate of growth. It is best suited for problems without saturating 
growth and takes advantage of the fact that a broad class of shapes can be 
approximated by a piecewise linear function.


Nonlinear growth
-----------------
The first one is called Nonlinear, Saturating Growth. It is represented in 
the form of the logistic growth model:

Seasonality
---------------
When dealing with data in practical applications, it is frequently necessary 
to take into account multiple seasonal patterns occurring in parallel; 
a classic example would be data related to energy consumption: there are 
morning vs evening patterns (intraday), workdays vs weekend (weekly) and 
during the year (annual). Modeling them explicitly tends to be cumbersome 
(you need to add more equations in exponential smoothing or introduce dummies 
in ARIMA), which is one of the issues Prophet was designed to overcome. 
The core of the underlying logic is a the Fourier expansion:

how you can specify different seasonality patterns ?
-----------------------------------------------------
Frequency shenanigans
--------------------------
all_valid = np.where(~np.isnan(xdat['PJME']))
print(len(all_valid[0]))
print(all_valid[0])
print(xdat[xdat.PJME==32896])

145366
[ 32896  32897  32898 ... 178259 178260 178261]
                  Datetime     PJME
44827  2003-08-21 07:00:00  32896.0
51700  2004-11-08 17:00:00  32896.0
57338  2004-03-18 17:00:00  32896.0
81046  2007-07-04 18:00:00  32896.0
103108 2010-12-27 05:00:00  32896.0
106252 2010-08-18 07:00:00  32896.0
158040 2016-09-20 09:00:00  32896.0


print(df.loc[3])

Datetime    1998-12-31 04:00:00
AEP                         NaN
COMED                       NaN
DAYTON                      NaN
DEOK                        NaN
DOM                         NaN
DUQ                         NaN
EKPC                        NaN
FE                          NaN
NI                          NaN
PJME                        NaN
PJMW                        NaN
PJM_Load                27596.0
Name: 3, dtype: object

print(xdat.head(5)) # Daily
          ds    y
0 1998-04-01  0.0
1 1998-04-02  0.0
2 1998-04-03  0.0
3 1998-04-04  0.0
4 1998-04-05  0.0


all_ix=np.where(xdat['y'] > 0)
print(all_ix)
print(len(all_ix[0]))
(array([1371, 1372, 1373, ..., 7427, 7428, 7429]),)
6059

ix : 1371

print(xdat.head(5)) # Monthly
          ds    y
0 1998-04-30  0.0
1 1998-05-31  0.0
2 1998-06-30  0.0
3 1998-07-31  0.0
4 1998-08-31  0.0


Seasonality specification
------------------------------
Apart from deciding on which frequencies to model explicitly, 
we have more options to setup our Prophet model. First, there is 
seasonality_mode - additive or multiplicative:

xlist = inspect.getfullargspec(Prophet).args
# we've these options in Prophet
['self',
 'growth',
 'changepoints',
 'n_changepoints',
 'changepoint_range',
 'yearly_seasonality',
 'weekly_seasonality',
 'daily_seasonality',
 'holidays',
 'seasonality_mode',
 'seasonality_prior_scale',
 'holidays_prior_scale',
 'changepoint_prior_scale',
 'mcmc_samples',
 'interval_width',
 'uncertainty_samples',
 'stan_backend']
Depending on the problem at hand, we might want to allow strong effects 
of the seasonal component on the forecast - or have it reduced. 
This intuition can be quantified by adjusting the
seasonality_prior_scale argument, which imapcts the extent to which the 
seasonality model will fit the data (remark for those 
with Bayesian exposure: works pretty much the way a prior would).


Last but not least, we can - as we usually ought to - use interval forecast, 
i.e. have our point estimates combined with uncertainty. 
By default the parameter mcmc_samples is set to 0, so to get the interval 
around seasonality, you must do full Bayesian sampling; uncertainty around 
trend can be calculated with Maximum A Posteriori (MAP) estimate.

Prophet in Holidays
--------------------
lower_window and upper_window: those two parameters allow us to 
incorporate the effect before/after the date, respectively

Outliers
-------------
We can use the built-in Prophet functionality to deal with 
outliers - for the sake of clarity of exposition - replacing the dubious 
observations with None (not NaN - it is an important distinction to keep 
in mind).

Another type of situation we may encounter is a few points whose values are 
extremely off, so as a result the seasonality estimate is impacted.
As before, the simplest solution is to get rid of those points and leave 
the algorithm to interpolate within the sample:

Performance evaluation
--------------------------