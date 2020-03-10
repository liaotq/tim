# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:13:04 2020

@author: liaot
"""

import pandas as pd
import numpy as np
#from scipy import stats
import matplotlib.pyplot as plt
#from statsmodels.tsa.stattools import acf,pacf,plot_acf,plot_pacf
#from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.graphics.gofplots import qqplot
from arch import arch_model

def optimization(mean_ret_0,mean_ret_1,cov_0,cov_1,a=5):
    G = np.matrix(np.ones((1,30)))
    U,d,V=np.linalg.svd(cov_0)
    U0 = np.matrix(U)
    V0 = np.matrix(V)
    D0=np.matrix(np.diag(d))
    
    U,d,V=np.linalg.svd(cov_1)
    U1 = np.matrix(U)
    V1 = np.matrix(V)
    D1=np.matrix(np.diag(d))
    
    C0_inv = U0 * D0.I * V0
    C1_inv = U1 * D1.I * V1
    c = np.matrix([1])
    
    R0 = np.matrix(mean_ret_0).T
    R1 = np.matrix(mean_ret_1).T
    Lambda = (G*(C0_inv-C1_inv)*G.T).I*(G*(C0_inv-C1_inv)*(R0-R1)-2*a*c)
    w = 1/2/a * (C0_inv-C1_inv)*((R0-R1)-G.T*Lambda)
    
    return w

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

 #%%
if __name__ == '__main__':    
    data = pd.read_csv('./Data/30 stocks log daily return(new).csv')
    data = data.iloc[1713:,]
    date = data['Dates']
    data.drop(columns=['Dates'],inplace=True)
    data.index = pd.to_datetime(date)
    
    portfolio_data = pd.read_csv('./Data/new_data.csv')
    portfolio_data = portfolio_data.iloc[1713:,]
    portfolio_date = portfolio_data['Dates']
    portfolio_data.drop(columns=['Dates'],inplace=True)
    portfolio_data.index = pd.to_datetime(portfolio_date)
  
    #####
    insample_data = data[:'2008-01-20']
    #####
    
    under_repub = insample_data[insample_data['party'].isin(['0'])]
    under_demo = insample_data[insample_data['party'].isin(['1'])]
    
    under_repub = under_repub.dropna(axis=1,how='all')
    under_demo = under_demo.dropna(axis=1,how='all')

    #%%%
    rnstk = under_repub.shape[1]
    dnstk = under_demo.shape[1]
    rstocks = under_repub.columns[:rnstk-1]
    dstocks = under_repub.columns[:rnstk-1]
    stocks = list(set(rstocks).union(set(dstocks)))
    

    r1 = {}
    r2 = {}
    order = (1,0,0)
    # Use the AR(1) to forecast expected return
    for stock in stocks:
        test_data1 = under_demo[stock]
        test_data1.dropna(inplace=True)
        tempModel1 = ARIMA(test_data1,order).fit()
        p1 = list(tempModel1.params)
        expected_rtn1 = p1[0] / (1 - p1[1])
        r1[stock] = expected_rtn1
        
        
        test_data2 = under_repub[stock]
        test_data2.dropna(inplace=True) 
        tempModel2 = ARIMA(test_data2,order).fit()
        p2 = list(tempModel2.params)
        expected_rtn2 = p2[0] / (1 - p2[1])
        r2[stock] = expected_rtn2
    r1_df = pd.DataFrame(data=r1.values(),index=r1.keys(),columns=['Expected rtn under democrat'])
    r2_df = pd.DataFrame(data=r2.values(),index=r2.keys(),columns=['Expected rtn under republican'])
    under_demo_mean = np.array(r1_df)
    under_repub_mean = np.array(r2_df)
    
    
    # Use the simple average as the expected return for 30 stocks
    comp_r1 = under_demo.mean()
    comp_r2 = under_repub.mean()
    
    #%% 
    # r1-r2
    
    r1minr2 = np.matrix(r1_df.iloc[:,0] - r2_df.iloc[:,0])
    r1minr2 = r1minr2 * 252
    
    # Covariance
    
#    under_demo_cov = under_demo.iloc[:,:30].cov()
#    under_repub_cov = under_repub.iloc[:,:30].cov()
    
    cov_mat = r1minr2.T * r1minr2
#    cov_mat = cov_mat * np.sqrt(252)
    
    ##  optimization
    G = np.matrix(np.ones((1,r1minr2.shape[1])))
    c = np.matrix([1])
    w = opt_with_constriant(r1minr2,cov_mat,G,c)
    new_w = w/w.sum()
    ret = data.loc['2008-01-20':'2009-01-20'][stocks]
#    ret = np.matrix(ret)
    result = np.matrix(ret) * new_w
    
    bmk = np.mean(ret,axis=1)
    result = pd.DataFrame(result,index = bmk.index)
#    plt.figure()
#    plt.plot(result.cumsum(),color = 'red') #实线
#    plt.plot(bmk.cumsum(),color='blue') #虚线
#    plt.show()
#    
    #%% backtest
    
    order = (1,0,0)
#    stock = 'GOOG'
#    test_data1 = under_demo[stock]
#    test_data1.dropna(inplace=True)
#    tempModel1 = ARIMA(test_data1,order).fit()
#    p1 = list(tempModel1.params)
#    expected_rtn1 = p1[0] / (1 - p1[1])
#    r1[stock] = expected_rtn1
    
    
#    test_data2 = under_repub['GOOG']
#    test_data2.dropna(inplace=True) 
#    tempModel2 = ARIMA(test_data2,order).fit()
#    p2 = list(tempModel2.params)
#    expected_rtn2 = p2[0] / (1 - p2[1])
#  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #%%
#    portfolio_data = pd.read_csv('./Data/new_data.csv')
#    date = portfolio_data['Dates']
#    portfolio_data.drop(columns=['Dates'])
#    portfolio_data.index = pd.to_datetime(date)
#    
#    ##### Democratic ###########
#    # Dickey-Fuller test
#    demo_lgrtn = portfolio_data['SPX Index']
#    demo_lgrtn = demo_lgrtn#[:'2018-12-31']
#    t1 = sts.adfuller(demo_lgrtn)  # ADF检验
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
#    lag_acf1 = sts.acf(demo_lgrtn, nlags=33)
#    lag_pacf1 = sts.pacf(demo_lgrtn, nlags=33, method='ols')
#    plt.plot(lag_pacf1)
#    plt.title('The PACF plot of the democratic portfolio:')
#    plt.xlabel('lag')
#    plt.ylabel('Coefficient')
#    plt.axhline(y = 0.1, color = 'red')
#    plt.axhline(y = -0.1, color = 'red')
#    plt.show()
#
#    order = (1,1)
#    demoModel = ARMA(demo_lgrtn,order).fit()
#    p1 = list(demoModel.params)
#    demo_Ertn = p1[0] / (1 - p1[1])
#    
#    predicts = demoModel.predict(start='2019-01-02',dynamic=True)
#    
###    
##    
    
#%%
    ########## Republican  ##################
#    repub_lgrtn = portfolio_data['logret_rep']
#    t2 = sts.adfuller(repub_lgrtn)  # ADF检验
#    output2=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
#    output2['value']['Test Statistic Value'] = t2[0]
#    output2['value']['p-value'] = t2[1]
#    output2['value']['Lags Used'] = t2[2]
#    output2['value']['Number of Observations Used'] = t2[3]
#    output2['value']['Critical Value(1%)'] = t2[4]['1%']
#    output2['value']['Critical Value(5%)'] = t2[4]['5%']
#    output2['value']['Critical Value(10%)'] = t2[4]['10%']
#    print(output2)
#    
#    #ACF and PACF     
#    lag_acf2 = sts.acf(repub_lgrtn, nlags=20)
#    lag_pacf2 = sts.pacf(repub_lgrtn, nlags=20, method='ols')
#    plt.plot(lag_pacf2)
#    plt.title('The PACF plot of the republican portfolio:')
#    plt.xlabel('lag')
#    plt.ylabel('Coefficient')
#    plt.axhline(y = 0.1, color = 'red')
#    plt.axhline(y = -0.1, color = 'red')
#    plt.show()
#    
##    order = (1,1)
#    repubModel = ARMA(repub_lgrtn,order).fit()
#    p2 = list(repubModel.params)
#    repub_Ertn = p2[0] / (1 - p2[1])
#    
    #%%
