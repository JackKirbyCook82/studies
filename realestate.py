import os.path
import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm, uniform

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
norm_pdf = lambda x, *args, average, stdev, **kwargs: norm.ppf(x, loc=average, scale=stdev)
uniform_pdf = lambda x, *args, lower, upper, **kwargs: uniform.ppf(x, loc=lower, scale=lower+upper)

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

createHousing = lambda count, sqft, yearbuilt: Housing.create(geography=geography, date=date, housing=dict(count=count, sqft=sqft, yearbuilt=yearbuilt, **housing), neighborhood=neighborhood, **prices)
createHousehold = lambda count, income: Household.create(date=date, household=dict(count=count, **household), financials=dict(income=income, **financials))


def createHousings(size, density, sqfts, yearbuilts):
    assert all([isinstance(item, np.ndarray) for item in (density, sqfts, yearbuilts,)])
    assert density.shape == sqfts.shape == yearbuilts.shape
    for ijx, iy, jy in zip(density.flatten(), sqfts.flatten(), yearbuilts.flatten()): 
        yield createHousing(np.round(ijx * size, decimals=0).astype('int64'), iy, jy)
                
def createHouseholds(size, density, incomes):
    assert all([isinstance(item, np.ndarray) for item in (density, incomes,)])
    assert density.shape == incomes.shape
    for x, y in zip(density.flatten(), incomes.flatten()):
        yield createHousehold(np.round(x * size, decimals=0).astype('int64'), y)
        

def main(*args, households, housings, income, sqft, yearbuilt, **kwargs):    
    xinc, yinc = lornez(**income, quantiles=households['quantiles'], function=lornez_function, integral=lornez_integral)
    xsqft, ysqft = distribution(**sqft, quantiles=housings['quantiles'], function=norm_pdf)
    xyrblt, yyrblt = distribution(**yearbuilt, quantiles=housings['quantiles'], function=norm_pdf)    
  
    ixHG, jxHG = np.meshgrid(xsqft, xyrblt)
    iyHG, jyHG = np.meshgrid(ysqft, yyrblt)
    xHG = np.multiply(ixHG, jxHG)

    households = [item for item in createHouseholds(households['size'], xinc, yinc)]
    housings = [item for item in createHousings(housings['size'], xHG, iyHG, jyHG)]
    for item in households: print(str(item))
    for item in housings: print(str(item))
    
    
if __name__ == "__main__": 
    inputParms = {}
    inputParms['households'] = dict(size=100, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    inputParms['housings'] = dict(size=100, quantiles=[0.25, 0.5, 0.75])
    inputParms['income'] = dict(average=50000, stdev=750, gini=0.35)
    inputParms['sqft'] = dict(average=1750, stdev=750)
    inputParms['yearbuilt'] = dict(average=1980, stdev=10)
    main(**inputParms)


    
    
    
    
    
    
    
    