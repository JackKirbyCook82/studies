import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm, uniform

from utilities.concepts import concept
from variables import Date, Geography
from realestate.economy import Economy, Curve, Rate, Loan, Broker
from realestate.households import Household
from realestate.housing import Housing
from realestate.markets import Personal_Property_Market

pd.set_option('display.max_columns', 10)
pd.set_option("display.precision", 0)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
np.set_printoptions(precision=3, suppress=True)

mortgage = lambda x, n: Loan('mortgage', balance=x, duration=n, rate=0.05, basis='year')
studentloan = lambda x, n: Loan('studentloan', balance=x, duration=n, rate=0.07, basis='year')

lornez_function = lambda x, a: x * math.exp(-a * (1-x))
lornez_integral = lambda a: (math.exp(-a) + a - 1) / (a**2)
norm_pdf = lambda x, *args, average, stdev, **kwargs: norm.ppf(x, loc=average, scale=stdev)
uniform_pdf = lambda x, *args, lower, upper, **kwargs: uniform.ppf(x, loc=lower, scale=upper-lower)

def lornez(*args, average, gini, quantiles, function, integral, **kwargs):  
    assert hasattr(function, '__call__') and hasattr(integral, '__call__')
    assert isinstance(quantiles, (list, np.ndarray))
    assert 0 <= gini <= 1
    zero_function = lambda a: 1 - 2 * integral(a) - gini
    coefficient = fsolve(zero_function, 1)
    quantiles = np.array([0, *quantiles, 1])
    sizes = np.diff(quantiles)    
    shares = np.diff(np.vectorize(lambda x: function(x, coefficient))(quantiles)) 
    return sizes.astype('float64'), (shares * average / sizes).astype('int64')

def distribution(*args, quantiles, function, **kwargs):
    assert isinstance(quantiles, (list, np.ndarray))
    assert hasattr(function, '__call__')
    quantiles = np.array([0, *quantiles, 1])
    sizes = np.diff(quantiles)  
    avgquantiles = (quantiles[1:] + quantiles[:-1])/2
    avgvalues = function(avgquantiles, **kwargs)
    return sizes.astype('float64'), avgvalues.astype('int64')

Space = concept('space', ['unit', 'sqft'])
Quality = concept('quality', ['yearbuilt'])
Housing.customize(concepts=dict(space=Space, quality=Quality), parameters=('space', 'quality',))
Household.customize(parameters=('consumption', 'housing',))

geography = Geography({'state':1, 'county':1})
date = Date({'year':2010}) 
broker = Broker(commissions=0.06) 

income_profile = np.array([10000, 20000, 75000, 125000]) / 12
saving_profile = np.array([0, 0.05, 0.10, 0.15]) 
savingrate = Curve(income_profile, saving_profile, extrapolate='last', method='linear',)
wealthrate = Rate.flat(2000, 0.02, basis='year')    
valuerate = Rate.flat(2000, 0.05, basis='year')
rentrate = Rate.flat(2000, 0.035, basis='year')     
incomerate = Rate.flat(2000, 0.035, basis='year')
inflationrate = Rate.flat(2000, 0, basis='year')

household = dict(age=30, race='White', education='Bachelors', children='W/OChildren', size=1, language='English', 
                 housing_income_ratio=0.3, poverty_housing=1930, poverty_consumption=10000/12)
financials = dict(risktolerance=1, discountrate=0.018, savingrate=savingrate)
housing = dict(unit='House', sqft=1500, valuerate=valuerate, rentrate=rentrate)    
neighborhood = dict()   


def createHousings(size, density, yearbuilts, prices):
    assert all([isinstance(item, np.ndarray) for item in (density, yearbuilts,)])
    assert density.shape == yearbuilts.shape
    for x, y in zip(density.flatten(), yearbuilts.flatten()): 
        count = np.round(x * size, decimals=0).astype('int64')
        yield Housing.create(geography=geography, date=date, housing=dict(count=count, yearbuilt=y, **housing), neighborhood=neighborhood, prices=prices)
                
def createHouseholds(size, density, incomes, economy):
    assert all([isinstance(item, np.ndarray) for item in (density, incomes,)])
    assert density.shape == incomes.shape
    for x, y in zip(density.flatten(), incomes.flatten()):
        count = np.round(x * size, decimals=0).astype('int64')
        yield Household.create(date=date, household=dict(count=count, **household), financials=dict(income=y, **financials), economy=economy)
        

def main(*args, households, housings, income, yearbuilt, **kwargs):    
    xinc, yinc = lornez(**income, quantiles=households['quantiles'], function=lornez_function, integral=lornez_integral)
    xyrblt, yyrblt = distribution(**yearbuilt, quantiles=housings['quantiles'], function=uniform_pdf)     
    prices = dict(sqftprice=100, sqftrent=1, sqftcost=0.5)
    economy = Economy(date=Date({'year':2000}), purchasingpower=1, wealthrate=wealthrate, incomerate=incomerate, inflationrate=inflationrate)
    households = [item for item in createHouseholds(households['size'], xinc, yinc, economy)]
    housings = [item for item in createHousings(housings['size'], xyrblt, yyrblt, prices)]
    market = Personal_Property_Market('renter', households=households, housings=housings)
    market.equilibrium(*args, economy=economy, broker=broker, **kwargs)
    print(market.table('housings'))        
 
    
if __name__ == "__main__": 
    inputParms = {}
    inputParms['households'] = dict(size=100, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    inputParms['housings'] = dict(size=100, quantiles=[0.25, 0.5, 0.75])
    inputParms['income'] = dict(average=50000/12, gini=0.35)
    inputParms['yearbuilt'] = dict(lower=1950, upper=2010)
    main(**inputParms)


    
    
    
    
    
    
    
    