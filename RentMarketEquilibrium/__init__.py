import math
import os.path
import numpy as np
import pandas as pd
from itertools import product, chain
from scipy.optimize import fsolve
from scipy.stats import norm, uniform, beta

import visualization as vis
from utilities.convergence import History, DurationDampener, OscillationConverger, ConvergenceError
from variables import Date, Geography
from realestate.economy import Economy, Rate, Broker
from realestate.households import Household
from realestate.housing import Housing
from realestate.markets import Personal_Property_Market

pd.set_option('display.max_columns', 10)
pd.set_option("display.precision", 0)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
np.set_printoptions(precision=5, linewidth=150, suppress=True)

INCQTILE = 'Income[%{:.0f}-%{:.0f}]'
RENTQTILE = 'Rent[%{:.0f}-%{:.0f}]'
FILENAME = 'RentMarketEquilibrium'

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
    return sizes.astype('float64'), avgvalues.astype('float64')

def meshdistribution(*args, distributions, **kwargs):
    assert isinstance(distributions, list)
    distributions = [distribution(*args, **values, **kwargs) for values in distributions]
    xvalues = [x.flatten() for x in np.meshgrid(*[items[0] for items in distributions])]
    yvalues = [y.flatten() for y in np.meshgrid(*[items[1] for items in distributions])]    
    sizes = np.array([np.prod(list(x)) for x in zip(*xvalues)])
    values = np.array([np.array(list(y)) for y in zip(*yvalues)]).transpose()
    return sizes.astype('float64'), tuple([*values.astype('float64')])


Housing.customize(parameters=('location', 'quality', 'space',), concepts=dict())
Household.customize(parameters=tuple(), lifetimes={'adulthood':15, 'retirement':65, 'death':95})

wealthrate = Rate.flat(2000, 0.02, basis='year')    
valuerate = Rate.flat(2000, 0.05, basis='year')
rentrate = Rate.flat(2000, 0.035, basis='year')     
incomerate = Rate.flat(2000, 0.035, basis='year')
inflationrate = Rate.flat(2000, 0, basis='year')
depreciationrate = Rate.flat(2000, 0, basis='year')

date = Date({'year':2010}) 
broker = Broker(commissions=0.06) 
rates = dict(wealthrate=wealthrate, incomerate=incomerate, inflationrate=inflationrate, depreciationrate=depreciationrate)    
economy = Economy(date=Date({'year':2000}), purchasepower=1, housingpower=1000, **rates)
   

def createHouseholds(count, density, incomes, *args, economy, **kwargs):
    assert all([isinstance(item, np.ndarray) for item in (density, incomes,)])
    assert density.shape == incomes.shape
    for x, inc in zip(density.flatten(), incomes.flatten()):
        i = np.ceil(x * count).astype('int64')
        financials = {'income':inc, 'risktolerance':1, 'discountrate':0.018}
        yield Household.create(count=i, date=date, age=30, household={}, financials=financials, economy=economy)
    
def createHousings(count, density, locations, qualities, spaces, *args, prices, **kwargs):
    assert all([isinstance(item, np.ndarray) for item in (density, locations, qualities, spaces,)])
    for x, loc, qual, space in zip(density.flatten(), locations.flatten(), qualities.flatten(), spaces.flatten()): 
        geography = Geography({'state':1, 'county':loc})
        i = np.ceil(x * count).astype('int64')
        housing = {'location':loc, 'quality':qual, 'space':space, 'valuerate':valuerate, 'rentrate':rentrate}
        yield Housing.create(count=i, date=date, geography=geography, housing=housing, prices=prices)
                
def createMarket(*args, households, housings, income, locations, qualities, spaces, **kwargs):
    prices = dict(price=100000, rent=500, cost=0.5)    
    hhsizes, hhvalues = lornez(*args, **income, **kwargs)
    hgsizes, hgvalues = meshdistribution(*args, distributions=[locations, qualities, spaces], **kwargs)
    ihouseholds = [ihousehold for ihousehold in createHouseholds(households, hhsizes, hhvalues, economy=economy)]
    ihousings = [ihousing for ihousing in createHousings(housings, hgsizes, *hgvalues, prices=prices)]
    return Personal_Property_Market('renter', *args, households=ihouseholds, housings=ihousings, economy=economy, broker=broker, date=date, **kwargs)

def plotMarket(history, *args, period, **kwargs):
    fig = vis.figures.createplot((14, 14), title=None)
    ax = vis.figures.createax(fig, x=1, y=1, pos=1)
    vis.plots.line_plot(ax, history.table(period))
    vis.figures.showplot(fig)

def plotHousing(history, index, *args, period, colors=['b', 'g', 'r', 'r'], **kwargs):
    fig = vis.figures.createplot((10, 10), title=None)
    ax = vis.figures.createax(fig, x=1, y=1, pos=1)
    vis.plots.line_plot(ax, history[index](period), colors=colors)
    vis.figures.showplot(fig)


def main(*args, **kwargs):
    history = History() 
    converger =  OscillationConverger(period=50, btol=25, ttol=1, otol=0.5)
    dampener = DurationDampener(period=25, size=0.1)
    market = createMarket(*args, stepsize=0.1, maxsteps=500, history=history, converger=converger, dampener=dampener, **kwargs)   
    try: market(*args, economy=economy, broker=broker, date=date, **kwargs)          
    except ConvergenceError:  pass
    plotMarket(history, *args, period=1, **kwargs)
    plotHousing(history, 0, *args, period=25, **kwargs)
    

if __name__ == "__main__": 
    inputParms = {}
    inputParms['households'] = 1250
    inputParms['housings'] = 1000
    inputParms['income'] = dict(average=60000/12, gini=0.3, quantiles=[0.2, 0.4, 0.6, 0.8], function=lornez_function, integral=lornez_integral)
    inputParms['locations'] = dict(lower=0, upper=2, quantiles=[0.5], function=uniform_pdf)
    inputParms['qualities'] = dict(lower=0, upper=2, quantiles=[0.5], function=uniform_pdf)
    inputParms['spaces'] = dict(lower=0, upper=2, quantiles=[0.5], function=uniform_pdf)
    main(**inputParms)



    
    
    
    
    
    