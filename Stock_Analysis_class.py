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
    
    def days_price_volitility(self, in_days:int , n:int )-> list:

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
        percentile = np.percentile(result, 75)
        print("75th percentile:", percentile)

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
    
    def daily_return_volatility(self, n: int) -> Tuple[float, float]:

        # this method is for simple monte carlo used
        temp_close_price = self.close_price[-n:]
        returns = [price / temp_close_price[i - 1] - 1 for i, price in enumerate(temp_close_price)][1:]
        #mean of daily return
        daily_mean_return = round(np.mean(returns),6)
        # daily volatility
        daily_volatility = round(np.std(returns), 6)


        return daily_mean_return, daily_volatility
    
    def annualized_return_volatility(self, n: int) -> Tuple[float, float]:
        temp_close_price = self.close_price[-n:]
        #calculate log(pt/pt-1)
        returns = []
        for p in range(1,n):
            returns.append(np.log(temp_close_price[p]/temp_close_price[p-1]))
        #returns = np.log(temp_close_price[1:] / temp_close_price[:-1])
        
        # Calculate μ and σ
        mu = np.mean(returns)
        
        sigma = np.std(returns)

        # Assuming 252 trading days in a year for annualization
        trading_days_per_year = 252

        # Annualize μ and σ
        annualized_mu = mu * trading_days_per_year
        annualized_sigma = sigma * np.sqrt(trading_days_per_year)

        return annualized_mu, annualized_sigma


    
class MonteCarloStimulate:
    def __init__(self, dr, dv):
        self.daily_return = dr
        self.daily_volatility = dv
        self.annualized_return = (1 + dr)**252 - 1
        self.annualized_volatility = 252**0.5 * dv

    def simple_monte_carlo(self,last_price,T = 252, simulations_num = 1000):
        df = pd.DataFrame()
        last_price_list = []
        for x in range(simulations_num):
            count = 0
            price_list = []
            price = last_price * (1 + np.random.normal(0, self.daily_volatility))
            price_list.append(price)
            
            for y in range(T):
                if count == T-1:
                    break
                price = price_list[count]* (1 + np.random.normal(0, self.daily_volatility))
                price_list.append(price)
                count += 1

            df = pd.concat([df,pd.Series(price_list)],axis = 1)
            #df[x] = price_list
            last_price_list.append(price_list[-1])

            if x %500 ==0:
                print("No.",x,"Simulation")

        print("Finish Simulation!")
                
        fig = plt.figure()
        fig.suptitle("Monte Carlo Simulation: MSFT")
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
        



    def Geometric_monte_carlo(self, s0, mu, sigma, T_years = 1, dt = 1/ 252, simulate_num = 1000):

        print("Geometric Monte Carlo result")
        paths = []
        last_price_list = []

        for x in range(simulate_num):

            prices=[s0]
            time = 0

            while(time+dt <= T_years):
                prices.append(prices[-1]*np.exp((mu - 0.5*(sigma**2))*dt + sigma*np.random.normal(0, np.sqrt(dt))))
                time += dt
            if T_years - (time) > 0:
                prices.append(prices[-1]*np.exp((mu - 0.5*(sigma**2))*(T_years-time) + sigma*np.random.normal(0, np.sqrt(T_years-time))))

            paths.append(prices)
            last_price_list.append(round(prices[-1],2))


        fig = plt.figure()
        fig.suptitle("Monte Carlo Simulation: GBM with" +"mu: "+ str(round(mu,2)) +" sigma: " + str(round(sigma,2)) )
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
        passing_gridline = [ self.findstartposition(prices[0]) ]
        for i in range(1,len(df)):
            passing_gridline.append(self.findgridline(df.Close[i-1],df.Close[i]))
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
                     str(self.sellhighest) +", gridline#: " + str(self.linenum) + "\n Asset return:" +str(round(self.dfresult.Asset[-2]/self.capital - 1,4)*100) + "%" )
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

                    profit = temp.Asset[-2]/1000000 - 1 # -2 because sometimes the -1 is nan due to no data 
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

    timesize = 2

    startday, endday = data_year_period(timesize)
    data = getstock("DIS", startday, endday)
    
    #draw(data)

    TheStock = StockAnalysis(data[:-100]) # normally, using data [:-100] data has sliced. because the rest of data need to use for grid strategy
    TheStock.days_price_volitility(10,100)#  first argument is price change in days, second is how many days' data want to use in calculations 

    
    lastestprice = TheStock.close_price[-100] # should be training end data records
    (r,v) = TheStock.daily_return_volatility(100) # how many days' data want to involved in calcuation. Notice it should count date from sliced data.
    (annual_r, annual_v) = TheStock.annualized_return_volatility(100)
    print("Daily Mean Return:",r)
    print("Daily Volatility:",v)
    print("Annual Mean Return:",annual_r)
    print("Annual Volatility:",annual_v)
    TheStock_simulate = MonteCarloStimulate(r,v)

    ##simple monte carlo
    TheStock_simulate.simple_monte_carlo(lastestprice,T = 100, simulations_num = 5000)
    
    ##Geometric Brownian Motion
    #TheStock_simulate.Geometric_monte_carlo(lastestprice,annual_r,annual_v,T_years = 1/4)

    
    StockGrid = GridTradeStrategy(63,102, 40, 5000)
    
    StockDf = StockGrid.run(data.Close[-100:])# the sliced data is for grid strategy backtesting
    
    print(StockDf)
    StockGrid.plotresultdf()

    

    

    
    print("Holding Strategy")
    print("buy: ", round(data.Close[-100],2) )
    print("sell: ", round(data.Close[-1],2) )
    print(str(round((data.Close[-1]/data.Close[-100] - 1),4)*100 ) + "%")
    

    
    '''
    opt = GridOptimization((5,20),(10,16),(30,40),data.Close[-100:])
    dfresult = opt.optimize()
    print(dfresult)
    top100 = dfresult[:]

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
    '''

if __name__ == '__main__':
    main()