import os.path
import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
from itertools import product

from parsers import DictorListParser
from specs import specs_fromfile
from utilities.concepts import concept
from variables import Variables, Date, Geography
from realestate.economy import Economy, Rate, Loan
from realestate.households import Household
from realestate.housing import Housing

pd.set_option('display.max_columns', 10)
pd.set_option("display.precision", 1)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
np.set_printoptions(precision=2, suppress=True)

rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
specsfile = os.path.join(rootdir, 'uscensus', 'specs.csv')
specsparsers = {'databasis': DictorListParser(pattern=';=')}
specs = specs_fromfile(specsfile, specsparsers)
custom_variables = Variables.create(**specs, name='USCensus')
noncustom_variables = Variables.load('date', 'geography', 'geopath', name='USCensus')
variables = custom_variables.update(noncustom_variables)

mortgage = lambda x, n: Loan('mortgage', balance=x, duration=n, rate=0.05, basis='year')
studentloan = lambda x, n: Loan('studentloan', balance=x, duration=n, rate=0.07, basis='year')

lornez_function = lambda x, a: x * math.exp(-a * (1-x))
lornez_integral = lambda a: (math.exp(-a) + a - 1) / (a**2)

def lornez(average, gini_index, *args, quantiles, function, integral, **kwargs):  
    assert hasattr(function, '__call__') and hasattr(integral, '__call__')
    assert isinstance(quantiles, (list, np.ndarray))
    assert 0 <= gini_index <= 1
    zero_function = lambda a: 1 - 2 * integral(a) - gini_index
    coefficient = fsolve(zero_function, 1)
    quantiles = np.array([0, *quantiles, 1])
    sizes = np.diff(quantiles)    
    shares = np.diff(np.vectorize(lambda x: function(x, coefficient))(quantiles)) 
    return sizes, shares * average / sizes

def normal(average, stdev, *args, quantiles, **kwargs):
    assert isinstance(quantiles, (list, np.ndarray))
    quantiles = np.array([0, *quantiles, 1])
    sizes = np.diff(quantiles)  
    avgquantiles = (quantiles[1:] + quantiles[:-1])/2
    avgvalues = norm.ppf(avgquantiles, loc=average, scale=stdev)
    return sizes, avgvalues

Space = concept('space', ['unit', 'sqft'])
Quality = concept('quality', ['yearbuilt'])
Housing.customize(concepts=dict(space=Space, quality=Quality), parameters=('space', 'quality',), variables=variables)
Household.customize(parameters=('spending', 'housing',), variables=variables)

geography = Geography({'state':1, 'county':1, 'tract':1})
date = Date({'year':2010}) 
 
discountrate = Rate(2000, 0.018, basis='year')
wealthrate = Rate(2000, 0.02, basis='year')    
valuerate = Rate(2000, 0.05, basis='year')
rentrate = Rate(2000, 0.035, basis='year')     
incomerate = Rate(2000, 0.035, basis='year')
inflationrate = Rate(2000, 0, basis='year')

prices = dict(sqftprice=100, sqftrent=1, sqftcost=0.5)
economy = dict(date=2000, purchasingpower=1, wealthrate=wealthrate, incomerate=incomerate, inflationrate=inflationrate)
household = dict(age=30, race='White', education='Bachelors', children='W/OChildren', size=1, language='English')
financials = dict(savings=0.2, risktolerance=1, discountrate=discountrate)
housing = dict(unit='House', valuerate=valuerate, rentrate=rentrate)    
neighborhood = dict()   


def createHouseholds(incomes):
    for income in incomes:
        yield Household.create(date=date, household=household, financials=dict(income=income, **financials))

def createHousings(sqfts, yearbuilts):
    for sqft, yearbuilt in product(sqfts, yearbuilts):
        yield Housing.create(geography=geography, date=date, housing=dict(sqft=sqft, yearbuilt=yearbuilt, **housing), neighborhood=neighborhood, **prices)


def main(*args, incomes, sqfts, yearbuilts, **kwargs):
    myEconomy = Economy(**economy)
    myHouseholds = [household for household in createHouseholds(incomes)]
    myHousings = [housing for housing in createHousings(sqfts, yearbuilts)]
    

if __name__ == "__main__": 
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    x, y = lornez(50000, 0.35, quantiles=quantiles, function=lornez_function, integral=lornez_integral)
    print(x, y)
    x, y = normal(1750, 750, quantiles=quantiles)
    print(x, y)    
    x, y = normal(1985, 10, quantiles=quantiles)
    print(x, y)
    


    
    
    
    
    
    
    
    