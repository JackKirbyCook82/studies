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
    assert hasattr(function, '__call__') and hasattr(integral, '__call__')
    assert isinstance(quantiles, (list, np.ndarray))
    assert 0 <= gini <= 1
    zero_function = lambda a: 1 - 2 * integral(a) - gini
    coefficient = fsolve(zero_function, 1)
    quantiles = np.array([0, *quantiles, 1])
    sizes = np.diff(quantiles)    
    shares = np.diff(np.vectorize(lambda x: function(x, coefficient))(quantiles)) 
    return sizes.astype('float64'), (shares * average / sizes).astype('int64')

#def distribution(*args, quantiles, function, **kwargs):
#    assert isinstance(quantiles, (list, np.ndarray))
#    assert hasattr(function, '__call__')
#    quantiles = np.array([0, *quantiles, 1])
#    sizes = np.diff(quantiles)  
#    avgquantiles = (quantiles[1:] + quantiles[:-1])/2
#    avgvalues = function(avgquantiles, **kwargs)
#    return sizes.astype('float64'), avgvalues.astype('int64')


Space = concept('space', ['unit', 'sqft'])
Quality = concept('quality', ['yearbuilt'])
Location = concept('location', ['rank'])
Housing.customize(concepts=dict(space=Space, quality=Quality, location=Location), parameters=('space', 'quality', 'location',))
Household.customize(lifetimes=LIFETIMES)

geography = Geography({'state':1, 'county':1})
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

household = dict(age=30, race='White', education='Bachelors', children='W/OChildren', size=1, language='English', 
                 housing_income_ratio=0.35, poverty_housing=1900, poverty_consumption=(12000/12)*0.65)
financials = dict(risktolerance=1, discountrate=0.018, savingrate=savingrate)
housing = dict(unit='House', sqft=1500, valuerate=valuerate, rentrate=rentrate)    
neighborhood = dict()   


#def createHousings(size, density, sqfts, yearbuilts, ranks, prices):
#    assert all([isinstance(item, np.ndarray) for item in (density, yearbuilts,)])
#    assert density.shape == yearbuilts.shape
#    for x, y in zip(density.flatten(), yearbuilts.flatten()): 
#        count = np.round(x * size, decimals=0).astype('int64')
#        yield Housing.create(geography=geography, date=date, housing=dict(count=count, yearbuilt=y, **housing), neighborhood=neighborhood, prices=prices)
                
#def createHouseholds(size, density, incomes, economy):
#    assert all([isinstance(item, np.ndarray) for item in (density, incomes,)])
#    assert density.shape == incomes.shape
#    for x, y in zip(density.flatten(), incomes.flatten()):
#        count = np.round(x * size, decimals=0).astype('int64')
#        yield Household.create(date=date, household=dict(count=count, **household), financials=dict(income=y, **financials), economy=economy)
    
    
#def calculate(*args, filename, betashape, avgincomes, giniindexes, households, housings, yearbuilts, **kwargs):    
#    records = []
#    economy = Economy(date=Date({'year':2000}), purchasingpower=1, wealthrate=wealthrate, incomerate=incomerate, inflationrate=inflationrate)
#    prices = dict(sqftprice=100, sqftrent=1, sqftcost=0.5)
#    for avgincome, giniindex in product(avgincomes, giniindexes):
#        xinc, yinc = lornez(average=avgincome, gini=giniindex, quantiles=households['quantiles'], function=lornez_function, integral=lornez_integral)
#        ihouseholds = [ihousehold for ihousehold in createHouseholds(households['size'], xinc, yinc, economy)]
#        for a, b in beta_generator(betashape):
#            print('Calculating(avgincome={average:.0f}, giniindex={gini}, betashape=({a}, {b}))'.format(average=avgincome, gini=giniindex, a=a, b=b))
#            xyrblt, yyrblt = distribution(**yearbuilts, a=a, b=b, quantiles=housings['quantiles'], function=beta_pdf)  
#            ihousings = [ihousing for ihousing in createHousings(housings['size'], xyrblt, yyrblt, prices)]
#            market = Personal_Property_Market('renter', households=ihouseholds, housings=ihousings)
#            try: 
#                market(*args, economy=economy, broker=broker, **kwargs)
#                incomes = {INCQTILE.format(i*100, j*100):ihousehold.financials.income for i, j, ihousehold in zip([0, *households['quantiles']], [*households['quantiles'], 1], ihouseholds)}
#                rents = {RENTQTILE.format(i*100, j*100):ihousing.rentercost for i, j, ihousing in zip([0, *housings['quantiles']], [*housings['quantiles'], 1], ihousings)}
#                incomearray = np.array([ihousehold.financials.income for ihousehold in ihouseholds])
#                rentarray = np.array([ihousing.rentercost for ihousing in ihousings])
#                stats = {'AvgIncome':wtaverage_vector(incomearray, xinc), 'StdIncome':wtstdev_vector(incomearray, xinc), 'AvgRent':wtaverage_vector(rentarray, xyrblt), 'StdRent':wtstdev_vector(rentarray, xyrblt)}
#                records.append({'GiniIndex':giniindex, 'BetaShapeA':a, 'BetaShapeB':b, **stats, **incomes, **rents})
#                print('Success')
#            except Exception: print('Failure')
#            Housing.clear()
#        Household.clear()
#    records = pd.DataFrame(records)
#    records.index.name = 'Calculation'
#    spreadsheet(records, filename)


#def table(*args, filename, **kwargs):
#    try: dataframe = load(filename)[['GiniIndex', 'BetaShapeA', 'BetaShapeB', 'AvgIncome', 'AvgRent']].dropna(axis=1)
#    except KeyError: dataframe = load(filename)[['GiniIndex', 'AvgIncome', 'AvgRent']].dropna(axis=1)    
#    print(dataframe)

#@keydispatcher
#def plot(method, *args, **kwargs): raise KeyError(method)

#@plot.register('averagerent')
#def plot_avgrent(*args, filename, betashape, rentlimit=None, **kwargs):
#    dataframe = load(filename).dropna(axis=1)
#    fig = vis.figures.createplot((10,10), title=None)   
#    ax = vis.figures.createax(fig, x=1, y=1, pos=1, projection='3D', limits={'z':(0, rentlimit)})
#    if filename == UNIFORMFILENAME:
#        vis.plots.surface_plot(ax, dataframe, *args, x='AvgIncome', y='GiniIndex', z='AvgRent', density=75, **kwargs)
#        vis.figures.setnames(title='Average Rent Uniform')
#    elif filename == BETAFILENAME and betashape is not None:
#        for a, b in beta_generator(betashape): 
#            vis.plots.surface_plot(ax, select(dataframe, a, b), *args, x='AvgIncome', y='GiniIndex', z='AvgRent', density=75, **kwargs)
#            vis.figures.setnames(title='Average Rent Beta')
#    else: raise ValueError(filename, betashape)
#    vis.figures.setnames(ax, names=dict(x='AvgIncome', y='GiniIndex', z='AvgRent'))
#    vis.figures.showplot(fig)    

#@plot.register('quantilerent')
#def plot_qtilerent(*args, filename, housings, a=None, b=None, rentlimit=None, **kwargs):
#    if filename == UNIFORMFILENAME: 
#        dataframe = load(filename).dropna(axis=1)
#        title = 'Quantile Rent Uniform'
#    elif all([filename == BETAFILENAME, a is not None, b is not None]):
#        dataframe = select(load(filename).dropna(axis=1), a=a, b=b)
#        title = 'Quantile Rent Beta w/ A={} & B={}'.format(a, b)
#    else: raise ValueError(filename, a, b)
#    fig = vis.figures.createplot((10,10), title=title)   
#    ax = vis.figures.createax(fig, x=1, y=1, pos=1, projection='3D', limits={'z':(0, rentlimit)})    
#    for i, j in zip([0, *housings['quantiles']], [*housings['quantiles'], 1]):
#        column = RENTQTILE.format(i*100, j*100)
#        vis.plots.surface_plot(ax, dataframe, *args, x='AvgIncome', y='GiniIndex', z=column, density=75, **kwargs)
#    vis.figures.setnames(ax, names=dict(x='AvgIncome', y='GiniIndex', z='QuantileRent'))
#    vis.figures.showplot(fig)  

    
def main(*args, recalculate, filename, **kwargs):
    pass

    
if __name__ == "__main__": 
    inputParms = {}
    inputParms['households'] = 10000
    inputParms['housings'] = 10000
    inputParms['avgincome'] = 50000
    inputParms['giniindex'] = 0.35
    inputParms['yearbuilt'] = dict(lower=1930, upper=2015)
    inputParms['sqft'] = dict(lower=500, upper=4000)
    inputParms['rank'] = dict(lower=1, upper=100)
    main(**inputParms)


    
    
    
    
    
    
    
    