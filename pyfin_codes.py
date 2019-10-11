#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        pyfin_codes.py

    DESCRIPTION
        例子1.计算欧式期权的蒙特卡洛估算函数 Mento Carlo Evaluation in BSM model.

    MODIFIED  (MM/DD/YY)
        Na  10/06/2019

"""
__VERSION__ = "1.0.0.10062019"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# configuration

# consts

# functions
### 例子1.计算欧式期权的蒙特卡洛估算函数
def mentocarlo_eurex():
    """蒙特卡洛估值"""
    S0 = 100.   # 初始股票指数水平S0
    K = 105.    # 欧式看涨期权的行权价格K
    T = 1.0     # 到期时间T
    r = 0.05    # 固定无风险短期利率r
    sigma = 0.2 # 固定波动率sigma

    I = 100000  # number of simulations, I个伪随机数
    np.random.seed(1000)
    z = np.random.standard_normal(I)    # 一个标准正态分布随机变量
    # 所有到期指数水平ST(i)：一个随机变量
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    # 到期时期权的所有内在价值hT(i)
    hT = np.maximum(ST-K, 0)
    # 蒙特卡洛估算函数估计的期权现值C0
    C0 = np.exp(-r*T) * np.sum(hT) / I

    print('蒙特卡洛估算函数估计的 European Call Option 期权现值:\t{:5.3f}'.format(C0))

    # visualize
    fname = r'D:\workspace\pyfinanalysis\pyfin\data\stockprice.xlsx'
    xlsx_file = pd.ExcelFile(fname)
    djia = xlsx_file.parse('DJIA')
    print('DJIA examples:\n{}'.format(djia.tail()))

    # 对数收益率
    djia['Log_Ret'] = np.log(djia['Close'] / djia['Close'].shift(1))
    # 波动率：通过滚动窗的标准差计算
    djia['Volatility'] = djia['Log_Ret'].rolling(window=100, center=False).std()
    # in Ipython
    # get_ipython().magic('matplotlib inline')
    djia[['Close', 'Volatility']].plot(subplots=True, color='blue', figsize=(8, 6), grid=True)

    # result
    return C0

def compare_speed():
    """高性能计算例子"""
    # use python
    loops = 25000000
    import math
    a = range(1, loops)
    def f(x):
        return 3 * math.log(x) + math.cos(x) ** 2
    # in ipython
    # get_ipython().magic('timeit r = [f(x) for x in a]')

    import timeit
    setup = '''loops = 25000000
import math
a = range(1, loops)
def f(x):
    return 3 * math.log(x) + math.cos(x) ** 2
    '''
    # import os
    # os.system(timeit.timeit(stmt='r = [f(x) for x in a]', setup='setup))
    print(timeit.Timer(stmt='r = [f(x) for x in a]', setup=setup).repeat(7, 10000))

    # use numpy
    import numpy as np
    a = np.arange(1, loops)
    # in ipython
    # get_ipython().magic('timeit r = 3 * np.log(a) + np.cos(a) ** 2')

    # in ipython
    # use numexpr
    # import numexpr as ne
    # ne.set_num_threads(1)
    # f = '3 * log(a) + cos(a) **2'
    # get_ipython().magic('timeit r = ne.evaluate(f)')

def npfin_practice():
    # pv：年利率3%，按季度付息，每月付10，5年后获取1376.09633204，求现值
    val_pv = np.pv(rate=0.03/4, nper=4*5, pmt=-10, fv=1376.09633204)
    print('年利率3%，按季度付息，每月付10，5年后获取1376.09633204，现值:\t{:.2f}'.format(-val_pv))

    # fv：年利率3%，按季度付息，本金1000，每月付10，求5年后的终值
    val_fv = np.fv(rate=0.03/4, nper=4*5, pmt=-10, pv=-1000)
    print('年利率3%，按季度付息，本金1000，每月付10，终值:\t{:.2f}'.format(val_fv))

    # fvals = []
    # for i in range(1, 11):
    #     fvals.append(np.fv(rate=0.03/4, nper=4*i, pmt=-10, pv=-1000))
    # plt.plot(fvals, 'bo')
    # plt.title('年利率3%，按季度付息，本金1000，每月付10，1-10年后的终值')
    # plt.show()

    # pmt：年利率1%，贷款1000万，求每月的分期付款
    val_pmt = np.pmt(rate=0.01/12, nper=12*30, pv=10000000)
    print('年利率1%，贷款1000万，每月的分期付款：\t{:.2f}'.format(-val_pmt))

    # nper：年利率10%，贷款9000，每月付款100，求付款期数
    val_nper = np.nper(rate=0.10/12, pmt=-100, pv=9000)
    print('年利率10%，贷款9000，每月付款100，付款期数：\t{:.2f}'.format(val_nper))

    # irr：现金流[-100, 38, 48, 90, 17, 36]，求内部收益率
    val_irr = np.irr(values=[-100, 38, 48, 90, 17, 36])
    print('现金流[-100, 38, 48, 90, 17, 36]，内部收益率：\t{:.0f}%'.format(100 * val_irr))

    # mirr：现金流[-100, 38, 48, 90, 17, 36]，投资成本3%，现金再投资收益率3%，求修正内部收益率
    val_mirr = np.mirr(values=[-100, 38, 48, 90, 17, 36], finance_rate=0.03, reinvest_rate=0.03)
    print('现金流[-100, 38, 48, 90, 17, 36]，投资成本3%，现金再投资收益率3%，求修正内部收益率：\t{:.0f}%'\
          .format(100 * val_mirr))

    # rate：贷款9000，每月付100，贷款167个月，求利率
    val_rate = 12 * np.rate(nper=167, pmt=-100, pv=9000, fv=0)
    print('贷款9000，每月付100，贷款167个月，利率：\t{:.0f}%'.format(100 * val_rate))

    # npv：随机在0-100之间取5个数作为现金流，利率3%，求净现值
    cashflows = np.random.randint(100, size=5)
    cashflows = np.insert(cashflows, 0, -100)
    print('Cashflows: {}'.format(cashflows))
    val_npv = np.npv(rate=0.03, values=cashflows)
    print('利率3%，上述现金流下，净现值：\t{:.2f}'.format(val_npv))


def main():
    # mentocarlo_eurex()
    npfin_practice()


# classes

# main entry
if __name__ == "__main__":
    main()
    