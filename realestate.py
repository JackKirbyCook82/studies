import math
import os.path
import numpy as np
import pandas as pd
from itertools import product, chain
from scipy.optimize import fsolve
from scipy.stats import norm, uniform, beta

from utilities.concepts import concept
from variables import Date, Geography
from realestate.economy import Economy, Curve, Rate, Loan, Broker
from realestate.households import Household
from realestate.housing import Housing
from realestate.markets import Personal_Property_Market

pd.set_option('display.max_columns', 10)
pd.set_option("display.precision", 3)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
np.set_printoptions(precision=3, suppress=True)

mortgage = lambda x, n: Loan('mortgage', balance=x, duration=n, rate=0.05, basis='year')
studentloan = lambda x, n: Loan('studentloan', balance=x, duration=n, rate=0.07, basis='year')

lornez_function = lambda x, a: x * math.exp(-a * (1-x))
lornez_integral = lambda a: (math.exp(-a) + a - 1) / (a**2)
norm_pdf = lambda x, *args, average, stdev, **kwargs: norm.ppf(x, loc=average, scale=stdev)
uniform_pdf = lambda x, *args, lower, upper, **kwargs: uniform.ppf(x, loc=lower, scale=upper-lower)
beta_pdf = lambda x, *args, a, b, lower, upper, **kwargs: beta.ppf(x, a, b, loc=lower, scale=upper-lower)
beta_generator = lambda x: chain(((1, 1) for i in range(1)), ((a, b) for a, b in product(range(2, x+1), range(2, x+1))))
excelfile = lambda filename: os.path.join(os.path.dirname(os.path.realpath(__file__)), '.'.join([filename, 'xlsx']))
spreadsheet = lambda dataframe, filename: dataframe.to_excel(excelfile(filename), index=False, header=True)

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

income_profile = np.array([12000, 24000, 75000, 125000]) / 12
saving_profile = np.array([0, 0.05, 0.10, 0.15]) 
savingrate = Curve(income_profile, saving_profile, extrapolate='last', method='linear',)
wealthrate = Rate.flat(2000, 0.02, basis='year')    
valuerate = Rate.flat(2000, 0.05, basis='year')
rentrate = Rate.flat(2000, 0.035, basis='year')     
incomerate = Rate.flat(2000, 0.035, basis='year')
inflationrate = Rate.flat(2000, 0, basis='year')

household = dict(age=30, race='White', education='Bachelors', children='W/OChildren', size=1, language='English', 
                 housing_income_ratio=0.35, poverty_housing=1900, poverty_consumption=(12000/12)*0.65)
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
        

def main(*args, filename, betashape, avgincomes, giniindexes, households, housings, yearbuilts, **kwargs):    
    records = []
    economy = Economy(date=Date({'year':2000}), purchasingpower=1, wealthrate=wealthrate, incomerate=incomerate, inflationrate=inflationrate)
    prices = dict(sqftprice=100, sqftrent=1, sqftcost=0.5)
    for avgincome in avgincomes:
        for giniindex in giniindexes:
            xinc, yinc = lornez(average=avgincome, gini=giniindex, quantiles=households['quantiles'], function=lornez_function, integral=lornez_integral)
            ihouseholds = [ihousehold for ihousehold in createHouseholds(households['size'], xinc, yinc, economy)]
            for a, b in beta_generator(betashape):
                 print('Calculating(avgincome={average:.0f}, giniindex={gini}, betashape=({a}, {b}))'.format(average=avgincome, gini=giniindex, a=a, b=b))
                 xyrblt, yyrblt = distribution(**yearbuilts, a=a, b=b, quantiles=housings['quantiles'], function=beta_pdf)  
                 ihousings = [ihousing for ihousing in createHousings(housings['size'], xyrblt, yyrblt, prices)]
                 market = Personal_Property_Market('renter', households=ihouseholds, housings=ihousings)
                 try: 
                     market(*args, economy=economy, broker=broker, **kwargs)
                     incomes = {'Income[%{:.0f}-%{:.0f}]'.format(i*100, j*100):ihousehold.financials.income for i, j, ihousehold in zip([0, *households['quantiles']], [*households['quantiles'], 1], ihouseholds)}
                     rents = {'Rent[%{:.0f}-%{:.0f}]'.format(i*100, j*100):ihousing.rentercost for i, j, ihousing in zip([0, *housings['quantiles']], [*housings['quantiles'], 1], ihousings)}
                     avgrent = np.average(np.array([ihousing.rentercost for ihousing in ihousings]))
                     records.append({'GiniIndex':giniindex, 'AvgIncome':avgincome, 'AvgRent':avgrent, 'BetaShapeA':a, 'BetaShapeB':b, **incomes, **rents})
                     print('Success')
                 except Exception: print('Failure')
                 Housing.clear()
            Household.clear()
    records = pd.DataFrame(records)
    records.index.name = 'Calculation'
    spreadsheet(records, filename)

    
if __name__ == "__main__": 
    inputParms = {}
    inputParms['filename'] = 'RentMarketEquilibriumBeta'
    inputParms['households'] = dict(size=10000, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    inputParms['housings'] = dict(size=10000, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    inputParms['yearbuilts'] = dict(lower=1900, upper=2000)
    inputParms['avgincomes'] = [i/12 for i in [40000, 50000, 60000, 80000]]
    inputParms['giniindexes'] = [0.25, 0.3, 0.35, 0.4, 0.45]
    inputParms['betashape'] = 4
    main(**inputParms)


    
    
    
    
    
    
    
    