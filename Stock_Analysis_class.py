import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
import matplotlib as mpl
import yfinance as yf
import mplfinance as mpf
import pandas_ta as pta
import yfinance as yf
from typing import Tuple
from datetime import date, timedelta


def data_year_period(yearnum: int) -> Tuple[str,str]:
    delta = timedelta(days=365*yearnum)
    today = date.today()
    yes = today - delta
    yes_str = yes.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    return yes_str, today_str

def getstock(stocktick, startday, endday):
    df = yf.download(stocktick,start = startday ,end=endday)
    df_adj = df.reindex(columns=[ 'Open', 'High', 'Low','Close'])#.rename(columns={"Adj Close":"Close"})
    return df_adj 

def draw(data):
        mpf.plot(data,type='candle')
        return 0

def splitdata(original_data_df, split_portion = 0.2):
    length = len(original_data_df)
    test_length = int(length * split_portion)
    train_df = original_data_df.iloc[:-test_length]
    test_df = original_data_df.iloc[-test_length:]
    return train_df, test_df 

    
class StockAnalysis:
    def __init__(self, price_data):
        self.date = price_data.index
        self.open_price = price_data.Open
        self.high_price = price_data.High
        self.low_price = price_data.Low
        self.close_price = price_data.Close

    def calculate_average(self, price_type: str, n:int ) -> float:
        """Calculate the average of the close prices."""
        if price_type == 'o':
            return sum(self.open_price[-n:]) / n
        elif price_type == 'h':
            return sum(self.high_price[-n:]) / n
        elif price_type == 'l':
            return sum(self.low_price[-n:]) / n
        elif price_type == 'c':
            return sum(self.close_price[-n:]) / n
        else:
            return "Can't find"
    
    def calculate_range(self, n:int) -> float:
        """Calculate the price range (difference between high and low prices)."""
        return max(self.high_price[-n:]) - min(self.low_price[-n:])
    
    def calculate_open_to_high(self, n:int) -> float:
        
        result = (self.high_price[-n:] - self.open_price[-n:]).tolist()
        result.sort()
        print("Mean: ", np.mean(result))

        # Calculate the percentile
        percentile = np.percentile(result, 95)
        print("95th percentile:", percentile)
        percentile = np.percentile(result, 75)
        print("75th percentile:", percentile)
        percentile = np.percentile(result, 50)
        print("50th percentile:", percentile)
        percentile = np.percentile(result, 25)
        print("25th percentile:", percentile)
        return 0       
    
    def calculate_open_to_low(self, n:int) -> float:
        
        result = (self.open_price[-n:] - self.low_price[-n:]).tolist()
        result.sort()
        print("Mean: ", np.mean(result))

        # Calculate the percentile
        percentile = np.percentile(result, 95)
        print("95th percentile:", percentile)
        percentile = np.percentile(result, 75)
        print("75th percentile:", percentile)
        percentile = np.percentile(result, 50)
        print("50th percentile:", percentile)
        percentile = np.percentile(result, 25)
        print("25th percentile:", percentile)
        return 0
    
    def days_price_volitility(self, in_days:int , n:int = 0)-> list:

        result = []
        #define data range:
        temp_date = self.date[-n:]
        temp_open_price = self.open_price[-n:]
        temp_high_price = self.high_price[-n:]
        temp_low_price = self.low_price[-n:]
        temp_close_price = self.close_price[-n:]

        #find the highest price and lowest price in days


        for i in range(n+1-in_days):
            local_high = max(temp_high_price[i:i+in_days])
            local_low = min(temp_low_price[i:i+in_days])
            local_diff = local_high-local_low
            result.append(local_diff)

        result.sort()

        # Calculate the percentile
        #percentile = np.percentile(result, 75)
        #print("75th percentile:", percentile)

        # Calculate summary statistics
        mean = np.mean(result)
        median = np.median(result)
        minimum = np.min(result)
        maximum = np.max(result)
        std_dev = np.std(result)

        print("Mean:", mean)
        print("Median:", median)
        print("Minimum:", minimum)
        print("Maximum:", maximum)
        print("Standard Deviation:", std_dev)
        
        return result
    
    def daily_return_volatility(self, n: int = 0) -> Tuple[float, float]:

        # this method is for simple monte carlo used
        temp_close_price = self.close_price[-n:]

        #daily returns
        daily_returns = np.diff(temp_close_price) / temp_close_price[:-1]
        #daily means
        daily_mean_return = np.mean(daily_returns)

        #daily variance of returns (There is no NaN values in the numpy array)
        daily_variance = ((daily_returns - daily_mean_return)**2).sum() / (len(daily_returns) -1)
        daily_volatility = daily_variance**.5

        return daily_mean_return, daily_volatility
    
    def annualized_return_volatility(self, n: int = 0) -> Tuple[float, float]:

        # this method is for simple monte carlo used
        temp_close_price = self.close_price[-n:]

        #daily returns
        daily_returns = np.diff(temp_close_price) / temp_close_price[:-1]
        #daily means
        daily_mean_return = np.mean(daily_returns)

        #daily variance of returns (There is no NaN values in the numpy array)
        daily_variance = ((daily_returns - daily_mean_return)**2).sum() / (len(daily_returns) -1)
        daily_volatility = daily_variance**.5

        # Assuming 252 trading days in a year for annualization
        trading_days_per_year = 252

        # Annualize μ and σ
        annualized_mu = (1 + daily_mean_return) ** trading_days_per_year - 1
        annualized_sigma = daily_volatility * np.sqrt(trading_days_per_year)

        return annualized_mu, annualized_sigma
    
class MonteCarloStimulate:
    def __init__(self, dr, dv):
        self.daily_return = dr
        self.daily_volatility = dv
        self.annualized_return = (1 + dr)**252 - 1
        self.annualized_volatility = 252**0.5 * dv

    def simple_monte_carlo(self,last_price,T_years = 1, simulations_num = 1000):

        df = pd.DataFrame()
        last_price_list = []
        T_days = 252 * T_years

        for x in range(simulations_num):
            count = 0
            price_list = []
            price = last_price * (1 + np.random.normal(0, self.daily_volatility))
            price_list.append(price)
            
            for y in range(T_days):
                if count == T_days-1:
                    break
                price = price_list[count]* (1 + np.random.normal(0, self.daily_volatility))
                price_list.append(price)
                count += 1

            df = pd.concat([df,pd.Series(price_list)],axis = 1)
            last_price_list.append(price_list[-1])

            if x %500 ==0:
                print("No.",x,"Simulation")

        print("Finish Simulation!")
                
        fig = plt.figure()
        fig.suptitle("Simple Monte Carlo Simulation with" +"mu: "+ str(round(self.annualized_return,2)) +" sigma: " + str(round(self.annualized_volatility,2)))
        plt.plot(df)
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.show()

        print("Expected price: ", round(np.mean(last_price_list),2))
        print("Quantile (5%): ",np.percentile(last_price_list,5))
        print("Quantile (95%): ",np.percentile(last_price_list,95))

        plt.hist(last_price_list,bins=100)
        plt.axvline(np.percentile(last_price_list,5), color='r', linestyle='dashed', linewidth=2)
        plt.axvline(np.percentile(last_price_list,95), color='r', linestyle='dashed', linewidth=2)
        plt.show()

    def Geometric_monte_carlo(self, s0, T_years = 1, dt = 1/ 252, simulations_num = 1000):

        print("Geometric Monte Carlo result")
        paths = []
        last_price_list = []

        for x in range(simulations_num):

            prices=[s0]
            time = 0

            while(time+dt <= T_years):
                prices.append(prices[-1]*np.exp((self.annualized_return - 0.5*(self.annualized_volatility**2))*dt + self.annualized_volatility*np.random.normal(0, np.sqrt(dt))))
                time += dt
            if T_years - (time) > 0:
                prices.append(prices[-1]*np.exp((self.annualized_return - 0.5*(self.annualized_volatility**2))*(T_years-time) + self.annualized_volatility*np.random.normal(0, np.sqrt(T_years-time))))

            paths.append(prices)
            last_price_list.append(round(prices[-1],2))


        fig = plt.figure()
        fig.suptitle("Monte Carlo Simulation: GBM with" +"mu: "+ str(round(self.annualized_return,2)) +" sigma: " + str(round(self.annualized_volatility,2)) )
        plt.xlabel('Day')
        plt.ylabel('Price')

        for path in paths:
            plt.plot(path)

        plt.show()
        
        
        print("Expected price: ", round(np.mean(last_price_list),2))
        print("Quantile (5%): ",np.percentile(last_price_list,5))
        print("Quantile (95%): ",np.percentile(last_price_list,95))

        plt.hist(last_price_list,bins=100)
        plt.axvline(np.percentile(last_price_list,5), color='r', linestyle='dashed', linewidth=2)
        plt.axvline(np.percentile(last_price_list,95), color='r', linestyle='dashed', linewidth=2)
        plt.show()
   
class GridTradeStrategy:

    def __init__(self, buylowest, sellhighest, line_number, capital, commission = 0.001425, taxrate = 0.003):

        self.buylowest = buylowest
        self.sellhighest = sellhighest
        self.linenum = line_number
        self.capital = capital
        self.commission = commission
        self.taxrate = taxrate
        

        grid_diff = round((sellhighest- buylowest)/(line_number - 1),8)
        line_list = []
        priceline = buylowest
        for i in range(line_number):
            line_list.append(priceline)
            priceline += grid_diff

        self.linelist = line_list

        self.dfresult = pd.DataFrame({})
        
    def getportionindex(self,price):
        count = 0
        for l in self.linelist:
            if price != l and price > l:
                count +=1
        if price > self.linelist[-1]:
            count -= 1
        return count
    
    def findgridline(self,close_previous,close_new):
        result = 0
        for l in self.linelist:
            if close_previous < l:
                if close_new < l:
                    break
                elif close_new >= l:
                    result = l
            elif close_previous == l:
                result = l
            elif close_previous > l:
                if close_new > l:
                    continue
                elif close_new <= l:
                    result = l
        return result
    
    def findstartposition(self, startprice):
        result = 0
        for l in self.linelist:
            if startprice > l:
                continue
            else:
                result = l
                break
        return result
    
    def spend_on_stock(self, price):
            # mean find price in the index of gridline ex 190 is index 0
            index = self.getportionindex(price)
            #if it is index 0, that mean spend all money, 
            #so using gridline number -1 (because there is one gridline is $0 ) minus index
            return self.capital/(len(self.linelist)-1)*(len(self.linelist)-1-index)
    
    def totalasset(self,lastprice, stockshare, remain_cash):
        return round(stockshare*lastprice + remain_cash , 2)
    
    def run(self, prices: list) -> pd.DataFrame():

        df = pd.DataFrame(prices)

        ## see how the price pass each gridline
        passing_gridline = [ self.findstartposition(prices.iloc[0]) ]
        for i in range(1,len(df)):
            passing_gridline.append(self.findgridline(df.Close.iloc[i-1],df.Close.iloc[i]))
        df['Passing_Price'] = passing_gridline
        
        #print(df[df.Passing_Price != 0])

        # running strategy
        spend_on_stock_list =[]
        stockflow=[]
        stocknum = []
        moneylist = []
        moneylist_tc = []

        last_cash_flow_in_stock = 0.0
        flowingstock = 0
        total_stock = 0
        money_onhand = self.capital 
        money_onhand_tax_com = self.capital

        for i in df.Passing_Price:
            if i == 0:
                
                spend_on_stock_list.append(0.0)
                stockflow.append(flowingstock)
                stocknum.append(total_stock)
                moneylist.append(money_onhand)
                moneylist_tc.append(money_onhand_tax_com)
            else:
                
                #target money spend
                spendon = self.spend_on_stock(i)
                

                if spendon != 0:
                    
                    #temp means change in spending on stock cash amount
                    temp = spendon - last_cash_flow_in_stock
                    spend_on_stock_list.append(temp)
                    last_cash_flow_in_stock = spendon
                    
                    flowingstock = (temp)//i
                    stockflow.append(flowingstock)
                    
                    total_stock+=flowingstock
                    stocknum.append(total_stock)
                    
                    money_onhand -= flowingstock*i
                    moneylist.append(money_onhand)
                    
                    if flowingstock > 0:# buy in
                        money_onhand_tax_com -= round(flowingstock*i*(1+self.commission),2)
                        moneylist_tc.append(money_onhand_tax_com)
                    elif flowingstock < 0:#sold out stock (com + tax)
                        money_onhand_tax_com -= round(flowingstock*i*(1-self.commission-self.taxrate),2)
                        moneylist_tc.append(money_onhand_tax_com)
                    else:
                        moneylist_tc.append(money_onhand_tax_com)
                    
                    flowingstock = 0
                    
                else:
                    spend_on_stock_list.append(0.0)
                    stockflow.append(flowingstock)
                    stocknum.append(total_stock)
                    moneylist.append(money_onhand)
                    moneylist_tc.append(money_onhand_tax_com)

        df['Cash_spend_on_stock'] = spend_on_stock_list
        df["Flowing_Stock"] = stockflow
        df["Stock_On_Hand"] = stocknum
        df["Money_Onhand"] = moneylist
        df["Money_Onhand_tax_com"] = moneylist_tc
        df['Asset'] = self.totalasset(df.Close,df.Stock_On_Hand,df.Money_Onhand_tax_com)

        self.dfresult = df
        
        
        

        return df
    
    def plotresultdf(self):

        fig = plt.figure(figsize=(12,8))
        fig.suptitle("Asset under GridStratey with\n" +"lowbound: "+ str(self.buylowest) +", highbound: " +  \
                     str(self.sellhighest) +", gridline#: " + str(self.linenum) + "\n Asset return:" +str(round(self.dfresult.Asset.iloc[-2]/self.capital - 1,4)*100) + "%" )
        plt.xlabel('Day')
        plt.ylabel('Asset')

        plt.plot(self.dfresult.Asset)
        plt.axhline( y = self.capital, color = 'r', linestyle = '--')
        plt.show()

class GridOptimization:
    def __init__(self, gridline_range: tuple, lowbuy_range: tuple, highsell_range: tuple ,testprices: pd.DataFrame()):
        
        self.gridline_range = range(gridline_range[0], gridline_range[1] + 1)
        self.lowbuy_range = range(lowbuy_range[0], lowbuy_range[1] + 1)
        self.highsell_range = range(highsell_range[0], highsell_range[1] + 1)
        self.testprices = testprices
    
    def optimize(self) -> pd.DataFrame():
        
        gridline = []
        lowbound = []
        highbound = []
        profits = []
        gridinterval = []

        for l in self.gridline_range:
            for buy in self.lowbuy_range:
                for sell in self.highsell_range:
                    temp = GridTradeStrategy(buy, sell, l, 1000000).run(self.testprices)

                    profit = temp.Asset.iloc[-2]/1000000 - 1 # -2 because sometimes the -1 is nan due to no data 
                    gridline.append(l)
                    lowbound.append(buy)
                    highbound.append(sell)
                    profits.append(profit)
                    gridinterval.append(round((sell-buy)/(l-1),1))

            print("Grid line finished",l)


        
        df = pd.DataFrame({"Profit": profits,"Gridline": gridline, "Low":lowbound, "High":highbound,"Interval":gridinterval})
        df = df.sort_values(by = ["Profit"], ascending = [False])


        return df



def main():

    #Set up time period for the stock price
    timesize = 1

    #Using the date for today
    startday, endday = data_year_period(timesize)

    #Choosing the stock tick
    data = getstock("DIS", startday, endday)

    #Visualization
    draw(data)

    #Split data into training and testing data, 80% as training data and 20% as testing data
    ( train_data, test_data ) = splitdata(data)

    #Using training data to analyze the percentile of price change in 10 days. (The days number can be changed)
    TheStock = StockAnalysis(train_data) 
    TheStock.days_price_volitility(10,100) # In 10 days duration, using only last 100 from the above "train_data" 

    #Get the annual return and volitiliy
    (r,v) = TheStock.daily_return_volatility()
    (annual_r, annual_v) = TheStock.annualized_return_volatility()
    print("Daily Mean Return:",r)
    print("Daily Volatility:",v)
    print("Annual Mean Return:",annual_r)
    print("Annual Volitility:",annual_v)

    
    #Monete Carlo Simulation
    TheStock_simulate = MonteCarloStimulate(r,v)
    
    startprice = test_data.Close.iloc[0]

    ##Simple Monte carlo
    TheStock_simulate.simple_monte_carlo(startprice, T_years = 1, simulations_num = 5000)
    
    ##Geometric Brownian Motion
    TheStock_simulate.Geometric_monte_carlo(startprice, T_years = 1, simulations_num = 5000)

    #Grid Trade
    StockGrid = GridTradeStrategy(60, 120, 11, 10000)
    
    StockDf = StockGrid.run(test_data.Close)
    
    print(StockDf)
    StockGrid.plotresultdf()

    #Optimization Strategy by observating the top return parameters set

    opt = GridOptimization(gridline_range = (5,20), lowbuy_range = (60,70), highsell_range  = (115, 125), testprices = test_data.Close)
    dfresult = opt.optimize()
    print(dfresult)
    top100 = dfresult[:100]

    plt.scatter(top100.Gridline, top100.Profit, marker='o', color='red', s=100, alpha=0.8)
    plt.xlabel('Lines number')
    plt.ylabel('Profit')
    plt.title('Scatter Plot')
    plt.grid(True)
    plt.show()

    plt.scatter(top100.Interval, top100.Profit, marker='o', color='blue', s=100, alpha=0.8)
    plt.xlabel('Interval')
    plt.ylabel('Profit')
    plt.title('Scatter Plot')
    plt.grid(True)
    plt.show()

    plt.scatter(top100.Low, top100.Profit, marker='o', color='blue', s=100, alpha=0.8)
    plt.xlabel('Low')
    plt.ylabel('Profit')
    plt.title('Scatter Plot')
    plt.grid(True)
    plt.show()

    plt.scatter(top100.High, top100.Profit, marker='o', color='blue', s=100, alpha=0.8)
    plt.xlabel('High')
    plt.ylabel('Profit')
    plt.title('Scatter Plot')
    plt.grid(True)
    plt.show()
   
    return 0
    

if __name__ == '__main__':
    main()