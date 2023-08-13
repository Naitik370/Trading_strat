import yfinance as yf
import numpy as np
from sklearn import linear_model as lgr,metrics
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
import matplotlib.pyplot as plt

class Money:
    df=None
    logi_fit=None
    days=None
    
    def __init__(self,ticker:str,start:str,end:str,Investment:float) -> None:
        start = pd.to_datetime(start) + pd.DateOffset(days=-5)
        self.inv = Investment
        self.df = yf.download(tickers=ticker,start=start,end=end)
        self.preprocess()
        self.df=self.df.reset_index()
        self.dat=self.df
    
    def preprocess(self)-> None:
        """Preprocess the input data"""
        self.df.columns = ["Open","High","Low","Close","Adj_Close","Volume"]
        self.df=self.df.drop(["Open","High","Low","Close","Volume"],axis=1)
        self.df["Percent_Change"] = self.df["Adj_Close"].pct_change()
        self.df.dropna(inplace=True)
        self.df["Today"] = self.df["Adj_Close"].pct_change()>0
        self.df["Today"] = self.df["Today"].replace({True: 1, False: 0})
            
    def shift(self,days:int) -> None:
        '''Enter the number of days to get the daywise difference as "X"'''
        for i in range(1,days):
            self.dat[f"{i}_D"] = self.dat["Percent_Change"].shift(i)
    
    def logistic(self,x,y):
        logr = lgr.LogisticRegression(class_weight="balanced")
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=False)
        logr.fit(x_train,y_train)
        pred=logr.predict(x_test)
        conf=metrics.confusion_matrix(list(y_test),pred)
        accuracy=(conf[0][0]+conf[1][1])/y_test.shape[0]
        return logr,pred,accuracy
    
    def logi_cv(self)->int:
        logr=None
        max=[0,0]
        for i in range(1,8):
            self.shift(i+1)
            data=self.dat.dropna()
            lst=[]
            for j in range(i):
                lst.append(f'{j+1}_D')
            logr,pred,accuracy=self.logistic(data[lst],data['Today'])
            if max[0]<accuracy: 
                max=[accuracy,i]
                self.logi_fit=logr
        self.days=max[1]
        print(f'Days Back-tracked {max[1]},',f'Accuracy: {(max[0]*100).round(3)}%')
        
    def BackTest(self):
        self.df.dropna(inplace=True)
        lst=[]
        for i in range(1,self.days+1):
            lst.append(f'{i}_D')
        x = self.df[lst]
        y = self.df["Today"]
        self.df["pred"] = self.logi_fit.predict(x)
        self.df["returns"] = self.df["pred"]*self.df["Percent_Change"]
        self.df["cum_strat_returns"] = ((self.df["returns"]+1).cumprod()-1)
        self.df["cum_returns"] = ((self.df["Percent_Change"]+1).cumprod()-1)
        print(f"Strategy Returns: {(self.inv*(1+self.df['cum_strat_returns'].iloc[-1]))}")
        print(f"Holding Returns: {(self.inv*(1+self.df['cum_returns'].iloc[-1]))}")
        plt.plot(self.df['Date'],self.df['cum_strat_returns'])
        plt.plot(self.df['Date'],self.df['cum_returns'])
    
            
ob=Money("INDHOTEL.NS",start='2022-04-06',end='2023-04-12',Investment=100000)
ob.logi_cv()
ob.BackTest()