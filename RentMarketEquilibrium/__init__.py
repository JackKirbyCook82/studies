import math
import os.path
import numpy as np
import pandas as pd
from itertools import product, chain
from scipy.optimize import fsolve
from scipy.stats import norm, uniform, beta

import visualization as vis
from utilities.concepts import concept
from utilities.narrays import wtaverage_vector, wtstdev_vector
from utilities.dispatchers import key_singledispatcher as keydispatcher
from variables import Date, Geography
from realestate.economy import Economy, Curve, Rate, Broker
from realestate.households import Household
from realestate.housing import Housing
from realestate.markets import Personal_Property_Market

pd.set_option('display.max_columns', 10)
pd.set_option("display.precision", 3)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
np.set_printoptions(precision=3, suppress=True)

INCQTILE = 'Income[%{:.0f}-%{:.0f}]'
RENTQTILE = 'Rent[%{:.0f}-%{:.0f}]'
FILENAME = 'RentMarketEquilibrium'
LIFETIMES = {'adulthood':15, 'retirement':65, 'death':95}  

lornez_function = lambda x, a: x * math.exp(-a * (1-x))
lornez_integral = lambda a: (math.exp(-a) + a - 1) / (a**2)
norm_pdf = lambda x, *args, average, stdev, **kwargs: norm.ppf(x, loc=average, scale=stdev)
uniform_pdf = lambda x, *args, lower, upper, **kwargs: uniform.ppf(x, loc=lower, scale=upper-lower)
beta_pdf = lambda x, *args, a, b, lower, upper, **kwargs: beta.ppf(x, a, b, loc=lower, scale=upper-lower)
beta_generator = lambda x: chain(((1, 1) for i in range(1)), ((a, b) for a, b in product(range(2, x+1), range(2, x+1))))

excelfile = lambda filename: os.path.join(os.path.dirname(os.path.realpath(__file__)), '.'.join([filename, 'xlsx']))
spreadsheet = lambda dataframe, filename: dataframe.to_excel(excelfile(filename), index=False, header=True)
load = lambda filename: pd.read_excel(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.'.join([filename, 'xlsx'])))


def lornez(*args, average, gini, quantiles, function, integral, **kwargs):  
    assert isinstance(quantiles, (list, np.ndarray))
    assert hasattr(function, '__call__') and hasattr(integral, '__call__')    
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

def meshdistribution(*args, distributions, **kwargs):
    assert isinstance(distributions, list)
    distributions = [distribution(*args, **values, **kwargs) for values in distributions]
    xvalues = [x.flatten() for x in np.meshgrid(*[items[0] for items in distributions])]
    yvalues = [y.flatten() for y in np.meshgrid(*[items[1] for items in distributions])]    
    sizes = np.array([np.prod(list(x)) for x in zip(*xvalues)])
    values = np.array([np.array(list(y)) for y in zip(*yvalues)]).transpose()
    return sizes.astype('float64'), tuple([*values.astype('int64')])


Space = concept('space', ['unit', 'sqft'])
Quality = concept('quality', ['yearbuilt'])
Location = concept('location', ['rank'])
Housing.customize(concepts=dict(space=Space, quality=Quality, location=Location), parameters=('space', 'quality', 'location',))
Household.customize(lifetimes=LIFETIMES)

date = Date({'year':2010}) 
broker = Broker(commissions=0.06) 

income_profile = np.array([12000, 24000, 75000, 125000]) / 12
saving_profile = np.array([0, 0.05, 0.10, 0.15]) 
savingrate = Curve(income_profile, saving_profile, extrapolate='last', method='linear')
wealthrate = Rate.flat(2000, 0.02, basis='year')    
valuerate = Rate.flat(2000, 0.05, basis='year')
rentrate = Rate.flat(2000, 0.035, basis='year')     
incomerate = Rate.flat(2000, 0.035, basis='year')
inflationrate = Rate.flat(2000, 0, basis='year')

household = dict(age=30, race='White', education='Bachelors', children='W/OChildren', size=1, language='English', housing_income_ratio=0.35, poverty_housing=1900, poverty_consumption=(12000/12)*0.65)
financials = dict(risktolerance=1, discountrate=0.018, savingrate=savingrate)
housing = dict(unit='House', valuerate=valuerate, rentrate=rentrate)    
neighborhood = dict()   


def createHouseholds(size, density, incomes, *args, economy, **kwargs):
    assert all([isinstance(item, np.ndarray) for item in (density, incomes,)])
    assert density.shape == incomes.shape
    for x, inc in zip(density.flatten(), incomes.flatten()):
        count = np.round(x * size, decimals=0).astype('int64')
        yield Household.create(date=date, household=dict(count=count, **household), financials=dict(income=inc, **financials), economy=economy)
    
def createHousings(size, density, yearbuilts, sqfts, ranks, *args, prices, **kwargs):
    assert all([isinstance(item, np.ndarray) for item in (density, yearbuilts, sqfts, ranks,)])
    assert density.shape == yearbuilts.shape == sqfts.shape == ranks.shape
    for x, yrblt, sqft, rank in zip(density.flatten(), yearbuilts.flatten(), sqfts.flatten(), ranks.flatten()): 
        geography = Geography({'state':1, 'county':rank})
        count = np.round(x * size, decimals=0).astype('int64')
        yield Housing.create(geography=geography, date=date, housing=dict(count=count, yearbuilt=yrblt, sqft=sqft, rank=rank, **housing), neighborhood=neighborhood, prices=prices)
                


def calculate(*args, households, housings, income, yearbuilt, sqft, rank, **kwargs):
    economy = Economy(date=Date({'year':2000}), purchasingpower=1, wealthrate=wealthrate, incomerate=incomerate, inflationrate=inflationrate)
    prices = dict(sqftprice=100, sqftrent=1, sqftcost=0.5)    
    hhsizes, hhvalues = lornez(*args, **income, **kwargs)
    hgsizes, hgvalues = meshdistribution(*args, distributions=[yearbuilt, sqft, rank], **kwargs)
    ihouseholds = [ihousehold for ihousehold in createHouseholds(households, hhsizes, hhvalues, economy=economy)]
    ihousings = [ihousing for ihousing in createHousings(housings, hgsizes, *hgvalues, prices=prices)]
    market = Personal_Property_Market('renter', households=ihouseholds, housings=ihousings)
    market(*args, economy=economy, broker=broker, **kwargs)
    

def main(*args, **kwargs):
    calculate(*args, **kwargs)
    print(Household.table())
    print(Housing.table())

    
if __name__ == "__main__": 
    inputParms = {}
    inputParms['households'] = 10000
    inputParms['housings'] = 10000
    inputParms['income'] = dict(average=50000/12, gini=0.35, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], function=lornez_function, integral=lornez_integral)
    inputParms['yearbuilt'] = dict(lower=1930, upper=2015, quantiles=[0.25, 0.5, 0.75], function=uniform_pdf)
    inputParms['sqft'] = dict(lower=500, upper=3500, quantiles=[0.25, 0.5, 0.75], function=uniform_pdf)
    inputParms['rank'] = dict(lower=1, upper=100, quantiles=[0.25, 0.5, 0.75], function=uniform_pdf)
    main(**inputParms)



    
    
    
    
    
    