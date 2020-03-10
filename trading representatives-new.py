# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:25 2019

@author: liaot
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

def simplex_projection_selfnorm2(v,b):
    while (max(abs(v)) > 1e6):
        v = v / 10
        break
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    c = np.array(range(1, len(u)+1, 1))
    sample = u - (sv - b) / c
    sample = sample[sample>0]
    rho = np.argmin(sample)
    theta = (sv[rho] - b) / (rho+1)
    w = np.maximum(v - theta, 0)
    return w



def trading_representative(close, pr, component, window_size, alpha, end_date):
    pick = comp_df.loc[comp_df['Date'] <= end_date][-2:]
    ## find the intersection of the S&P500 stocks during the last two months
    pick = list(pick['Symbols'].values)
    stocks = list(set(pick[0]).intersection(set(pick[1])))
    
    # find the corresponding index for end_date and start_date
    end_idx = factors[0][factors[0]['Date'].values <= end_date].index[-1]   #t
    start_idx = end_idx - window_size - 1   #t-6
   
    sliced_pr = pr.iloc[start_idx:end_idx+1, :].loc[:,stocks]
    sliced_pr[sliced_pr>1.5] = np.nan
    sliced_pr[sliced_pr<0.5] = np.nan
    sliced_pr.dropna(axis=1, how='any',inplace = True)
    stks = list(set(stocks).intersection(set(sliced_pr.columns)))

#    # slice computed EMA, SMA, PP
    sliced_factors = []
    for factor in factors:
        tmp = factor.loc[start_idx:end_idx, stks]
        tmp.dropna(axis=1, how='any',inplace = True)
        stks = list(set(stks).intersection(set(tmp.columns)))
        sliced_factors.append(tmp)

    for i, sfactor in enumerate(sliced_factors):
        sliced_factors[i] = sfactor.loc[:, stks].apply(lambda x: simplex_projection_selfnorm2(x,1),axis=1)

    sliced_pr = sliced_pr[stks]
    
    rhat_list = []
    rt = (sliced_pr.iloc[-window_size:,]).reset_index(drop=True)

    for sfactor in sliced_factors:
        rhat_list.append((sfactor.iloc[-1-window_size:-1,]).reset_index(drop=True) * rt)
    
    min_list = []
    for rhat in rhat_list:
        min_list.append( np.min(np.sum(rhat,axis=1)) )
    
    idx = np.argmax(np.array(min_list))
    
    base = sliced_factors[idx].iloc[-1,:]
    
    wlist = []
    final_weights = None
    for sfactor in sliced_factors:
        if final_weights  is not None:
            final_weights = final_weights + sfactor.iloc[-1, :] * np.exp((-(sfactor.iloc[-1,:]-base)**2).sum()/(2*0.0025)) 
        else:
            final_weights = sfactor.iloc[-1, :] * np.exp((-(sfactor.iloc[-1,:]-base)**2).sum()/(2*0.0025)) 
        wlist.append(sfactor.iloc[-1, :])

    w = simplex_projection_selfnorm2(final_weights,1)

    return  w,wlist

if __name__ == '__main__':
    # global variables
    win_size = 5    # time window
    alpha = 0.5     # parameter in computing EMA

    with open('./sp500-historical-components.json','r') as f:
        total_sp500 = json.loads(f.read())   # total_sp500 is a list

    # clean the data
    comp_df = pd.DataFrame(data = total_sp500)
    comp_df = comp_df.sort_values(by = 'Date', ascending = True)  # sort the dataframe by date
    comp_df = comp_df.reset_index(drop = True)
    comp_df['Date'] = pd.to_datetime(comp_df['Date'], format = '%Y/%m/%d')
    total_mon = len(comp_df['Date'])
    
    # find the union of the stocks over the entire period
    symbols = comp_df['Symbols']
    union = symbols[0]
    for i in range(1,len(symbols)):
        union = list(set(union).union(set(symbols[i])))
    
    # construct a big dataframe that stores all the price relatives of all stocks over the entire period
    close = {}
    for stock in union:
        filename = './Code-master(new)/data1/' + stock + '.csv'
        file = pd.read_csv(filename)
        file.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)          
        # extract the adj close prices
        adj_close = file['Adj Close']
        adj_close.index = file['Date']
        close[stock] = adj_close
    close_df = pd.DataFrame(close)
    
    close_df = close_df.dropna(axis = 0, how = 'all')
#    close_df = close_df.dropna(axis = 1, how = 'any')
    pr_df = close_df / close_df.shift(1)

#    pr_df = pr_df.drop('2007-01-03',axis = 0)
#    pr_df = pr_df.dropna(axis=1,how='any')
    
    # adjust the close_df to the same length as pr_df
#    close_df = close_df[1:]
    

    [T,nstk] = pr_df.shape
    # compute EMA, SMA, PP for the entire period
    EMA = np.ones(shape = (T,nstk))
    for i in range(1,T):
        EMA[i] = (1 - alpha) * EMA[i-1] / pr_df.iloc[i,] + alpha
    EMA = pd.DataFrame(data = EMA, columns = pr_df.columns, index = pr_df.index)
    SMA = close_df.rolling(win_size).mean()/close_df
    PP = close_df.rolling(win_size).max() / close_df
#    PP = PP[pr_df.columns]
    LL = close_df.rolling(win_size).min() / close_df  
    #VOL = (close_df.rolling(win_size).max()-close_df.rolling(win_size).min()) / close_df + 1
    
    #print(EMA)
    #W_EMA = EMA.apply(lambda x: simplex_projection_selfnorm2(x,1),axis=1)
    
    
    
    factors = [SMA,LL,EMA,PP]
    factors_name = ['sma','ll','ema','pp']
    for i,v in enumerate(factors):
        factors[i].reset_index(inplace=True)
    
    ##  =============== ##
    ## =================##
    start_date = "2008-03-01"
    end_date = "2019-01-01"
    date_list = close_df.loc[start_date:, :].index.tolist()
    ww = {}
    result = []
    result_single = {}
    for ff in factors_name:
        result_single[ff] = []
    bmk = []
    
    date = '2018-12-15'
    for date in date_list:


        if date >= end_date:
            break
        
        for next_date in date_list:
            if next_date > date:
                break
    
        # call the trading_representative function
        w, wlist = trading_representative(close_df, pr_df, comp_df, win_size, alpha, end_date = date)
        
        ret1 = pr_df.loc[next_date, w.inderex.tolist()]
        tmp_result = (w*ret1).sum()

        for i, wi in enumerate(wlist):
            result_single[factors_name[i]].append((wi * ret1).sum()) 

        bmk_result = (1*ret1).mean()
        
        ww[date] = w
        result.append(tmp_result)
        bmk.append(bmk_result)
        print(date)


    result = pd.DataFrame(result)
    result.columns = ['s']
    for i,v in result_single.items():
        result[i] = v

    result['bmk'] = bmk
    saved_result = result.copy()
    result[result>1.2] = 1
    result[result<0.8] = 1
    
    result.cumprod().plot()
    
        
#        
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # test the correlation between the trading representatives
#        stks = SMA.columns
#        fdr_s = []
#        fdr_p = []
#        fdr_e = []
#        s_p = []
#        s_e = []
#        p_e = []
#        
#        for s in stks:
##            fdr_sma = FDR[s].corr(SMA[s])
##            fdr_pp = FDR[s].corr(PP[s])
##            fdr_ema = FDR[s].corr(EMA[s])
#            sma_pp = SMA[s].corr(PP[s])
#            sma_ema = SMA[s].corr(EMA[s])
#            pp_ema = PP[s].corr(EMA[s])
##            fdr_s.append(fdr_sma)
##            fdr_p.append(fdr_pp)
##            fdr_e.append(fdr_ema)
#            s_p.append(sma_pp)
#            s_e.append(sma_ema)
#            p_e.append(pp_ema)
#        corr = pd.DataFrame(data = [s_p, s_e, p_e], columns = stks, index = ['SMA_PP','SMA_EMA','PP_EMA'])
#        corr_mean = np.mean(corr,axis = 1)



#        corr = {}
#        for s in stks:
#            temp_df = pd.DataFrame.merge(EMA[s], SMA[s], how = 'left', on = 'Date')
#            temp_df = pd.DataFrame.merge(temp_df, PP[s], how = 'left', on = 'Date')
#            temp_df.columns = ['EMA','SMA','PP']
#            temp_df = temp_df.dropna()
#            corr[s] = temp_df.corr()
        #cross time test
#        ema_mean = EMA.mean()
#        sma_mean = SMA.mean()
#        pp_mean = PP.mean()
#        tr_mean = pd.DataFrame({'ema':ema_mean,'sma':sma_mean,'pp':pp_mean})
#        tr_corr = tr_mean.corr()    
            
        # cross market test
#        ema = np.mean(EMA, axis = 1)
#        ema.name = 'EMA'
#        sma = np.mean(SMA, axis = 1)
#        sma.name = 'SMA'
#        pp = np.mean(PP, axis = 1)
#        pp.name = 'PP'
#        cross_mkt_tr = pd.DataFrame.merge(ema, sma, how = 'left', on = 'Date')
#        cross_mkt_tr = pd.DataFrame.merge(cross_mkt_tr, pp, how = 'left', on = 'Date')
#        corr = cross_mkt_tr.corr()    
#    