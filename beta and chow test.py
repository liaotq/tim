# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:32:49 2020

@author: liaot
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm


def capm(p_rtn, mkt_rtn):
    '''Calculate the alpha and beta of the portfolio.'''
    Y = p_rtn
    X = mkt_rtn 
    # Add the intercept
    X1 = sm.add_constant(X)
#    regr = regression.linear_model.OLS(p_rtn,mkt_rtn).fit()
    model = sm.OLS(Y,X1)
    results = model.fit()
    alpha = results.params[0]
    beta = results.params[1]
    return (alpha,beta)

def ffm(p_rtn,factor_rtn):
    '''Compute the coefficients of the Fama French 3 factor model.'''
    Y = p_rtn
    X = pd.DataFrame(data=np.ones((len(factor_rtn),1)),columns = ['intercept'])
    X = pd.concat([X,factor_rtn],axis = 1) 
    a = np.linalg.inv(np.dot(X.T,X)) 
    b = np.dot(a,X.T)
    regr_coef = np.dot(b,Y)
    resid = Y - np.dot(X,regr_coef)
    return regr_coef,resid
    
#%% 
if __name__ == '__main__':
   
#    demo = pd.read_csv('./Data/1974 DEMO.csv')
#    repub = pd.read_csv('./Data/1974 REPUB.csv')
#    
#    # Compute the return of the S&P500 and the two portfolios
#    spy_rtn = demo['market'] / demo['market'].shift(1) - 1
#    spy_rtn = spy_rtn.fillna(0)
#    demo_rtn = demo['portfolio'] / demo['portfolio'].shift(1) - 1
#    demo_rtn = demo_rtn.fillna(0)
#    #demo_rtn[demo_rtn > 0.2] = 0.2
#    demo['port_rtn'] = demo_rtn
#    demo['mkt_rtn'] = spy_rtn
#    repub_rtn = repub['portfolio'] / repub['portfolio'].shift(1) - 1
#    repub_rtn = repub_rtn.fillna(0)
#    #repub_rtn[repub_rtn > 0.2] = 0.2
#    repub['port_rtn'] = repub_rtn 
#    repub['mkt_rtn'] = spy_rtn
#    
#    
#    
#     
#   # Calculate the betas for the whole overlapping period
#    repub_beta = cal_beta(repub_rtn[1714:,],spy_rtn[1714:,])
#    demo_beta = cal_beta(demo_rtn[1714:,],spy_rtn[1714:,])
#    
#    repub_r = repub[repub['Party'].isin(['0'])]
#    repub_d = repub[repub['Party'].isin(['1'])]  
#    demo_r = demo[demo['Party'].isin(['0'])]
#    demo_d = demo[demo['Party'].isin(['1'])]
#    
#    rr_beta = cal_beta(repub_r['port_rtn'],repub_r['mkt_rtn'])
#    rd_beta = cal_beta(repub_d['port_rtn'],repub_d['mkt_rtn'])
#    
#    dr_beta = cal_beta(demo_r['port_rtn'],demo_r['mkt_rtn'])
#    dd_beta = cal_beta(demo_d['port_rtn'],demo_d['mkt_rtn'])
#    
#    plt.plot(demo['port_rtn'].loc[1714:,])
#    plt.title('Democratic portfolio returns')
#    plt.show()
#    plt.plot(repub['port_rtn'].loc[1714:,])
#    plt.title('Republican portfolio returns')
#    plt.show()
#
    stock_data = pd.read_csv('./Data/30 stocks log daily return.csv')
    portfolio_data = pd.read_csv('./Data/data.csv')
    ff = pd.read_csv('./Data/FFdata_daily.csv')
    ff = ff[ff['Date'] >= 19800729]
    ff = ff.reset_index(drop=True)
    del_lst = []
    for i in range(len(portfolio_data)):
        if (portfolio_data['logret_rep'][i] == 0 and portfolio_data['logret_dem'][i] == 0 and portfolio_data['SPX Index'][i]==0):
            portfolio_data.drop(index=i,inplace=True)
            stock_data.drop(index=i,inplace=True)
            del_lst += [i]
#    portfolio_data.to_csv('new_data.csv')
    stock_data.to_csv('30 stocks log daily return(new).csv')
#    
    
    
    
#%%    
    # CAPM
    repub_alpha,repub_beta = capm(portfolio_data['logret_rep'],portfolio_data['SPX Index'])
    demo_alpha,demo_beta = capm(portfolio_data['logret_dem'],portfolio_data['SPX Index'])
    print('The alpha and beta of the republican portfolio throughout the history are:', 252*repub_alpha,repub_beta)
    print('The alpha and beta of the democratic portfolio throughout the history are:', 252*demo_alpha,demo_beta)
    
    under_repub = portfolio_data[portfolio_data['Party'].isin(['0'])]
    under_demo = portfolio_data[portfolio_data['Party'].isin(['1'])]
    
    under_repub = under_repub.reset_index(drop = True)
    under_demo = under_demo.reset_index(drop = True)
    
    
    rr_alpha,rr_beta = capm(under_repub['logret_rep'],under_repub['SPX Index'])
    dr_alpha,dr_beta = capm(under_repub['logret_dem'],under_repub['SPX Index'])
    
    rd_alpha,rd_beta = capm(under_demo['logret_rep'],under_demo['SPX Index'])
    dd_alpha,dd_beta = capm(under_demo['logret_dem'],under_demo['SPX Index'])

    print('The alpha and beta of democratic portfolio under demo party are:',252 * dd_alpha,dd_beta)
    print('The alpha and beta of democratic portfolio under repub party are:',252 * dr_alpha,dr_beta) 
    
    print('The alpha and beta of republican portfolio under demo party are:',252 * rd_alpha,rd_beta)
    print('The alpha and beta of republican portfolio under repub party are:',252 * rr_alpha,rr_beta)
    
    
#    # FF three-factor model
    ff_copy = ff.drop(columns=['Date'])
    ff_copy = np.log(1+ff_copy)
    r_coef,r_resid = ffm(portfolio_data['logret_rep'][:len(ff_copy)],ff_copy.iloc[:,:3])
    d_coef,d_resid = ffm(portfolio_data['logret_dem'][:len(ff_copy)],ff_copy.iloc[:,:3])
    
#    rr_coef,rr_resid = ffm(under_repub['logret_rep'])
#    
    
    
#%%
#    demo = pd.read_csv('./Data/1974 DEMO.csv')
#    repub = pd.read_csv('./Data/1974 REPUB.csv')
#    
#    # choose the date '07/28/1980' as the beginning day 
#    demo_price = demo.iloc[1713:,1:17].reset_index(drop = True)
#    demo_price.index = demo.iloc[1713:,0]
#    
#    repub_price = repub.iloc[1713:,1:17].reset_index(drop = True)
#    repub_price.index = repub.iloc[1713:,0]
#    
#    # Compute the log returns for each stocks
#    lgrtn_demo = np.log(demo_price / demo_price.shift(1))
#    lgrtn_demo = lgrtn_demo.dropna(axis = 0, how = 'all')
#    lgrtn_repub = np.log(repub_price / repub_price.shift(1))
#    lgrtn_repub = lgrtn_repub.dropna(axis = 0, how = 'all')
#    
#    # Use the log returns of each stock to compute the return of the portfolio
#    lgrtn_demo['portfolio'] = np.mean(lgrtn_demo, axis = 1)
#    lgrtn_repub['portfolio'] = np.mean(lgrtn_repub, axis = 1)
#    
#    # Add the 'Party' column
#    lgrtn_demo['Party'] = np.array(demo['Party'][1714:])
#    lgrtn_repub['Party'] = np.array(repub['Party'][1714:])
    # Plot the log returns of the two portfolios
#    plt.plot(lgrtn_demo['portfolio'])
#    plt.title('Democratic portfolio returns')
#    plt.show()
#    plt.plot(lgrtn_repub['portfolio'])
#    plt.title('Republican portfolio returns')
#    plt.show()
   
#    demo_beta = cal_beta(lgrtn_demo['portfolio'],lgrtn_demo['market'])
#    repub_beta = cal_beta(lgrtn_repub['portfolio'],lgrtn_repub['market'])
#    
#    
#    # Divide the stocks of each portfolio into two groups
#    repub_r = lgrtn_repub[lgrtn_repub['Party'].isin(['0'])]
#    repub_d = lgrtn_repub[lgrtn_repub['Party'].isin(['1'])]  
#    demo_r = lgrtn_demo[lgrtn_demo['Party'].isin(['0'])]
#    demo_d = lgrtn_demo[lgrtn_demo['Party'].isin(['1'])]
#    
#    rr_beta = cal_beta(repub_r['portfolio'],repub_r['market'])
#    rd_beta = cal_beta(repub_d['portfolio'],repub_d['market'])
#    
#    dr_beta = cal_beta(demo_r['portfolio'],demo_r['market'])
#    dd_beta = cal_beta(demo_d['portfolio'],demo_d['market'])
    
    
#%%    
#    #########  Resample lgrtn into the yearly frequency
#    dport_lgrtn = under_demo['logret_dem']
#    dport_lgrtn.index = pd.to_datetime(under_demo['Dates'])
#    dport_ann_lgrtn = (dport_lgrtn.resample('Y', convention='start')).ffill()
#
#    rport_lgrtn = under_repub['logret_rep']
#    rport_lgrtn.index = pd.to_datetime(under_repub['Dates'])
#    rport_ann_lgrtn = (rport_lgrtn.resample('Y', convention='start')).ffill()

    
    
    
    # F-test
#    F = np.var(dport_ann_lgrtn) / np.var(rport_ann_lgrtn)
#    df1 = len(dport_ann_lgrtn) - 1
#    df2 = len(rport_ann_lgrtn) - 1
#    p_value = 1 - 2 * abs(0.5 - stats.f.cdf(F, df1, df2))
#    
#    
#    rr_lgrtn = repub_r['portfolio']
#    rr_lgrtn.index = pd.to_datetime(rr_lgrtn.index)
#    
#    
#    F2 = np.var(dport_ann_lgrtn) / np.var(rport_ann_lgrtn)
#    df1 = len(dport_ann_lgrtn) - 1
#    df2 = len(rport_ann_lgrtn) - 1
#    p_value = 1 - 2 * abs(0.5 - stats.f.cdf(F, df1, df2))
    
    