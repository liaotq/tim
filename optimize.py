# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:18:14 2020

@author: liaot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import sys
sys.path.append(r'C:\MyFile\BU MSMF\IAQF')
from garch import predict_var


def opt_with_constriant(R, C, G, c,a=1):
    
    R = (np.matrix(R)).T
#    C_inv = np.linalg.inv(C)
    U,d,V=np.linalg.svd(C)
    U0 = np.matrix(U)
    V0 = np.matrix(V)
    D0=np.matrix(np.diag(d))
    C_inv = U0 * D0.I * V0
    GC_invG = np.dot(np.dot(G,C_inv),G.T)
    gcg_inv = np.linalg.inv(GC_invG)
    l = gcg_inv * (np.dot(np.dot(G,C_inv),R) - 2 * a * c)
    w = (1 / 2*a) * C_inv * (R - np.dot(G.T,l))
    return w

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




if __name__ == '__main__':    

    # clean the data
    portfolio_data = pd.read_csv('./Data/new_data.csv')
#    portfolio_data = portfolio_data.iloc[1713:,]
    portfolio_date = portfolio_data['Dates']
    portfolio_data.drop(columns=['Dates'],inplace=True)
    portfolio_data.index = pd.to_datetime(portfolio_date)
    
#    idx = portfolio_data.index
    
    data = pd.read_csv('./Data/30 stocks log daily return(newest).csv')
#    data = data.iloc[1713:,]
    date = data['Dates']
    data.drop(columns=['Dates'],inplace=True)
    data.index = pd.to_datetime(date)
#    data = data.loc[idx]
#%%    
    
    stocks = data.columns[:30]
#    start_year = '2000'
#    end_year = '2020'
#    
#    start_date = start_year + '-01-20'
#    end_date = end_year +'-01-20'
#    next_year = str(int(end_year)+1) + '-01-20'
    
    
    
#    sliced_data1 = data.loc['2009-01-20':'2012-01-20']######################
#    sliced_data2 = data.loc['2013-01-20':'2016-01-20']
    
    sliced_data1 = data[data['party'].isin(['1'])]  # democratic
    sliced_data2 = data[data['party'].isin(['0'])]  # republican
    
    alpha1 = []
    beta1 = []
    
    alpha2 = []
    beta2 = []
    
    del_lst = []
    
    for stock in stocks:
        
        stk_rtn1 = sliced_data1[stock].dropna()
        stk_rtn2 = sliced_data2[stock].dropna()
        
        # check if either of the stock return sequences under two periods is empty
        if len(stk_rtn1)== 0 or len(stk_rtn2) == 0:
            print(stock + ' has all nan.')
            del_lst += [stock]
            continue    
        temp_idx1 = stk_rtn1.index
        mkt_rtn1 =  portfolio_data.loc[temp_idx1]['SPX Index']
        alpha,beta = capm(stk_rtn1,mkt_rtn1)
        alpha1+= [alpha]
        beta1+=[beta]
        
        
        temp_idx2 = stk_rtn2.index
        mkt_rtn2 =  portfolio_data.loc[temp_idx2]['SPX Index']
        alpha_,beta_ = capm(stk_rtn2,mkt_rtn2)
        alpha2+=[alpha_]
        beta2+=[beta_]
    
    new_stocks = stocks.drop(del_lst)


 #%%   
    
#    t1 = sts.adfuller(mkt)  # ADF test
#    output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
#    output['value']['Test Statistic Value'] = t1[0]
#    output['value']['p-value'] = t1[1]
#    output['value']['Lags Used'] = t1[2]
#    output['value']['Number of Observations Used'] = t1[3]
#    output['value']['Critical Value(1%)'] = t1[4]['1%']
#    output['value']['Critical Value(5%)'] = t1[4]['5%']
#    output['value']['Critical Value(10%)'] = t1[4]['10%']
#    print(output)
#    
#    #ACF and PACF     
#    lag_acf1 = sts.acf(mkt, nlags=33)
#    lag_pacf1 = sts.pacf(mkt, nlags=33, method='ols')
#    plt.plot(lag_pacf1)
#    plt.title('The PACF plot of the democratic portfolio:')
#    plt.xlabel('lag')
#    plt.ylabel('Coefficient')
#    plt.axhline(y = 0.1, color = 'red')
#    plt.axhline(y = -0.1, color = 'red')
#    plt.show()
#        
#    mkt = portfolio_data.loc[:'2020-01-20']['SPX Index']      #############  
#    order = (1,1)
#    model = ARMA(mkt,order).fit()
#    p1 = list(model.params)
#    unconditonal_mean = p1[0] / (1 - p1[1])
#        
#    forecast_rtn = []
#    cumsum = 1
#    for i in range(1,252+1):
#        
#        cond_mean = mkt[-1] * p1[1] ** i + p1[0] * cumsum
#        cumsum += p1[1] ** i
#        forecast_rtn += [cond_mean]
        
    #%% test forecast mkt rtn
#    real_mktrtn = portfolio_data.loc['2019-01-28':]['SPX Index']
#    forecast_mktrtn = pd.DataFrame(forecast_rtn,index=real_mktrtn.index)
#    plt.plot(real_mktrtn)
#    plt.plot(forecast_mktrtn)
#    plt.show()
#    
    
    
    
    
    
 #%%   
     # Construct objective function and optimize
     
#    forecast_cumsum = np.cumsum(forecast_rtn)
#    mktrtn_T =  np.sum(forecast_rtn)
#    r1 = pd.DataFrame(data=mktrtn_T * beta1.values() + alpha1)
    
#    forecast_rtn * (np.array(beta1) - np.array(beta2))
    mktrtn_T = 0.005
    alpha1 = np.array(alpha1)    
    alpha2 = np.array(alpha2)
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)
    r1 = mktrtn_T * beta1 + alpha1   # democrat
    r2 = mktrtn_T * beta2 + alpha2    # republican
    r1minr2 = r1 - r2
#    cov_mat = np.matrix(r1minr2).T * np.matrix(r1minr2)
#    cov_mat = data.iloc[:,:30].cov()
    history_price = data.loc['2015-07-07':].iloc[:,:30]###########
    cov_mat = predict_var(history_price,252)
    
    G = np.matrix(np.ones((1,len(new_stocks))))
    c = np.matrix([1])
    w = opt_with_constriant(r1minr2,cov_mat,G,c)
    new_w = w/w.sum()  
    weights = pd.DataFrame(new_w,index=new_stocks,columns=['optimized_weights'])
#    weights.to_csv('optimized weights.csv')
    
    
#%%  plot the cumulative return  
    
#    ret = data.loc['2016-01-20':'2017-01-20'][new_stocks]
#    result = np.matrix(ret) * new_w
#    
##    bmk = np.mean(ret,axis=1)
#    bmk = portfolio_data.loc['2016-01-20':'2017-01-20']['SPX Index']
#    result = pd.DataFrame(result,index = bmk.index)
#    
##%%
#    plt.plot(bmk.cumsum(),color='blue')
#    plt.plot(result.cumsum(),color='red')
#    plt.show()
    
#%%   compute the at-the-money strike
    price = pd.read_csv('./Data/stock price.csv')
    date = price['Dates']
    price.drop(columns=['Dates'],inplace=True)
    price.index = pd.to_datetime(date)
    current_price = price.loc['2020-01-29'][:30]  ##############
#    standardized_price = (current_price - current_price.mean())/(current_price.max() - current_price.min())
    K = np.array(current_price) * new_w
    
    
    
    
    
    
    
    