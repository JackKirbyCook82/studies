import numpy as np

O = lambda n: np.ones(n)
N = lambda n: np.arange(n)
R = lambda r, n: (O(n) * np.array(1+r)) ** N(n)
PMT = lambda pv, r, n: -np.pmt(r, n, pv)

def CIPV(pv, r, q, i, n):
    X = R(r, i) / R(q, i)
    Y =  (np.array(1) - R(r, i)) / R(q, i)
    CX = np.sum(X)
    CY = np.sum(Y)
    return pv * r * CX + PMT(pv, r, n) * CY

price = 315000
term = 30*12
downpayment = 5/100
discount = 5/100
rates = np.array([2.5, 2.625, 2.75, 2.875, 3, 3.125, 3.25, 3.375, 3.5, 3.625])/100
points = np.array([-5143, -3767, -1613, -21, 1209, 2146, 2726, 3941, 4896, 5545])
period = 10*12

cipv = np.vectorize(lambda x: CIPV(price * (1-downpayment), x, discount, period, term))(rates)
valuation = np.convolve(cipv, np.array([1, -1]), mode='valid')
market = np.convolve(points, np.array([1, -1]), mode='valid')
npv = np.round(np.cumsum(np.concatenate([np.array([0]), market - valuation])))

for rate, value in zip(rates, npv): print('{:.04f} % | $ {:.0f}'.format(rate, value))