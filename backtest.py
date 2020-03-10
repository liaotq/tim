# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:44:15 2020

@author: liaot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import sys
sys.path.append(r'C:\MyFile\BU MSMF\IAQF')
from garch import predict_var


def opt_with_constriant(R, C, a):
    G = np.matrix(np.ones((1,len(R))))
    c = np.matrix([1])
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

def regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    results = model.summary()
    return model, results

def regression_res(tickers,ret_0,ret_1):
    res_0 = []
    res_1 = []
    new_tickers = []
    for i in tickers:
        y1 = ret_0[i].dropna()
        y2 = ret_1[i].dropna()
        if(len(y1)>0 and len(y2)>0):
            y1 = y1 - ret_0.loc[y1.index]['RF']
            y2 = y2 - ret_1.loc[y2.index]['RF']
            x1 = ret_0.loc[y1.index]['SPX Index'] - ret_0.loc[y1.index]['RF']
            res_0.append(regression(x1,y1)[0])
            x2 = ret_1.loc[y2.index]['SPX Index'] - ret_1.loc[y2.index]['RF']
            res_1.append(regression(x2,y2)[0])
            new_tickers += [i]
    return res_0, res_1, new_tickers


def pred_ret(regression_res_0,regression_res_1, ret=0.005):
    ret_list_0=[]
    for i in range(len(regression_res_0)):
        ret_list_0.append(regression_res_0[i].predict([0,ret])[0])
    ret_list_1=[]
    for i in range(len(regression_res_1)):
        ret_list_1.append(regression_res_1[i].predict([0,ret])[0])
    return ret_list_0, ret_list_1


if __name__ == '__main__':    

    # clean the data
    portfolio_data = pd.read_csv('./Data/new_data.csv')
    portfolio_date = portfolio_data['Dates']
    portfolio_data.drop(columns=['Dates'],inplace=True)
    portfolio_data.index = pd.to_datetime(portfolio_date)
    
    
    data = pd.read_csv('./Data/30 stocks log daily return(newest).csv')
    data_date = data['Dates']
    data.drop(columns=['Dates'],inplace=True)
    data.index = pd.to_datetime(data_date)
    data['RF'] = data['RF'] / 252
    stocks = data.columns[:30]
    
    price = pd.read_csv('./Data/stock price.csv')
    price_date = price['Dates']
    price.drop(columns=['Dates'],inplace=True)
    price.index = pd.to_datetime(price_date)
    
    
    ############### backtest begins  ####################
    a=0.01   # risk aversion 
    
    start_year = '2000'
    end_year = '2016'
    date_lst = []
#    valid_lst = []
    nyear = int((int(end_year) - int(start_year)) / 4)
    for n in range(0,nyear+1):
        temp = int(start_year) + 4 * n
        temp_date = str(temp)+'-01-20'
#        valid_date = str(temp+1)
        date_lst += [temp_date]
        
        
    for date in date_lst:
        history_data = data.loc[:date]
        valid_date = str(int(date[:4])+1) + '-01-20'
        valid_data = data.loc[date:valid_date]
        sliced_data1 = history_data[history_data['party'].isin(['1'])]  # democratic
        sliced_data2 = history_data[history_data['party'].isin(['0'])]  # republican
        
        
        
        demo_res,repub_res,new_stocks = regression_res(stocks,sliced_data1,sliced_data2)
#        print(1)
        demo_ret,repub_ret = pred_ret(demo_res,repub_res, ret=0.01)
#        print(1)
        ret_diff = np.array(demo_ret) - np.array(repub_ret)   
        ret_diff_ = np.array(repub_ret) - np.array(demo_ret)
        cov_mat = predict_var(history_data[new_stocks].dropna(axis=0,how='any'),252)
#        cov_mat = np.matrix(ret_diff-np.mean(ret_diff)).T * np.matrix(ret_diff-np.mean(ret_diff))
        w = opt_with_constriant(ret_diff, cov_mat,a)
        w_ = opt_with_constriant(ret_diff_, cov_mat,a)
        new_w = w/w.sum()
        new_w_ = w_/w_.sum()
#        print(1)
        
        
        ret = valid_data[new_stocks]
        result = np.matrix(ret) * new_w
        result_ = np.matrix(ret) * new_w_
        
        bmk = valid_data['SPX Index']
        result = pd.DataFrame(result,index = bmk.index)
        result_ = pd.DataFrame(result_,index = bmk.index)
    
        plt.plot(bmk.cumsum(),color='blue')
        plt.plot(result.cumsum(),color='red')
        plt.show()
        
        
        plt.plot(bmk.cumsum(),color='blue')
        plt.plot(result_.cumsum(),color='red')
        plt.show()
        
        
        print(np.mean(np.abs(new_w-new_w_)))
        
        
#%%        
        ## adding risk-free asset
        
       
        current_price = price.loc[date][new_stocks]  
#    standardized_price = (current_price - current_price.mean())/(current_price.max() - current_price.min())
        K = np.array(current_price) * new_w
        
        mu = np.mean(result)
        sigma = np.std(result)
        alpha = mu + 0.5 * sigma ** 2
        r0 = np.mean(history_data['RF']) / 252
        w_star = (alpha.values - r0) / (a* K * sigma.values ** 2)
        
        
        
        
        print(1)
















