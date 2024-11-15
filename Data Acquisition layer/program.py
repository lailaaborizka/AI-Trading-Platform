from models.models import *
from price_action.price_module import * 
from news.news_module import * 
from helpers import *
from twitter.twitter_module import * 
from env import * 
from datetime import timedelta 
from models.models import dropTables , app
class priceActionStream():

    lastDate = None
    firstTime = True
    df_daily = df_hour_1 = df_min_5 = df_min_1 = None
    def __init__(self, ticker, starting_date = None, from_time = None , time_to = None , online = False ) -> None:
        self.ticker = ticker
        self.starting_date = starting_date
        self.from_time = from_time
        self.time_to = time_to
        self.online = online
        self.lastDate = starting_date

    def next(self, step_size = 1):

        if(self.online):
            pass
        else:

            try:
                if(self.firstTime):
                    self.df_daily = retrievePriceAction(self.ticker , "Daily" , from_time=self.starting_date , time_to=self.time_to)
                    self.df_hour_1 = retrievePriceAction(self.ticker , "Hourly" , from_time=self.starting_date , time_to=self.time_to)
                    self.df_min_5 = retrievePriceAction(self.ticker , "Min5" , from_time=self.starting_date , time_to=self.time_to)
                    self.df_min_1 = retrievePriceAction(self.ticker , "Min1" , from_time=self.starting_date , time_to=self.time_to)

                    df_daily_slice = retrievePriceAction(self.ticker , "Daily" , from_time=self.from_time , time_to=self.starting_date)
                    df_hour_1_slice = retrievePriceAction(self.ticker , "Hourly" , from_time=self.from_time , time_to=self.starting_date)
                    df_min_5_slice = retrievePriceAction(self.ticker , "Min5" , from_time=self.from_time , time_to=self.starting_date)
                    df_min_1_slice = retrievePriceAction(self.ticker , "Min1" , from_time=self.from_time , time_to=self.starting_date)

                    self.firstTime = False
                
                else:

                    df_daily_slice = self.df_daily[self.df_daily['timestamp'] <= self.lastDate + timedelta(seconds=step_size)]
                    df_hour_1_slice = self.df_hour_1[self.df_hour_1['timestamp'] <= self.lastDate + timedelta(seconds=step_size)]
                    df_min_5_slice = self.df_min_5[self.df_min_5['timestamp'] <= self.lastDate + timedelta(seconds=step_size)]
                    df_min_1_slice = self.df_min_1[self.df_min_1['timestamp'] <= self.lastDate + timedelta(seconds=step_size)]

                    self.df_daily = self.df_daily[self.df_daily['timestamp'] > self.lastDate + timedelta(seconds=step_size)]
                    self.df_hour_1 = self.df_hour_1[self.df_hour_1['timestamp'] > self.lastDate + timedelta(seconds=step_size)]
                    self.df_min_5 = self.df_min_5[self.df_min_5['timestamp'] > self.lastDate + timedelta(seconds=step_size)]
                    self.df_min_1 = self.df_min_1[self.df_min_1['timestamp'] > self.lastDate + timedelta(seconds=step_size)]

                    self.lastDate = self.lastDate + timedelta(seconds=step_size)
                print( self.lastDate)

                response = {}

                response = {**response , "Daily" : df_daily_slice} if len(df_daily_slice) !=0 else {**response}
                response = {**response , "Hourly" : df_hour_1_slice} if len(df_hour_1_slice) !=0 else {**response}
                response = {**response , "Min5" : df_min_5_slice} if len(df_min_5_slice) !=0 else {**response}
                response = {**response , "Min1" : df_min_1_slice} if len(df_min_1_slice) !=0 else {**response}

                if(len(response) == 0):
                    if(len(self.df_daily) != 0 or len(self.df_hour_1) != 0 or len(self.df_min_5) != 0 or len(self.df_min_1) != 0):
                        response = {"Interrupt" : True , "error" : False , "messgae" : "waiting for interval to catch" }
                    else:
                        response = {"Interrupt" : True , "no_more_data" : True , "error" : False  , 'message' : "there is no more data"}

            except  Exception as e:
                response = {"error" : True , "message" : str(e)}

            return response




# updateTweets(twitter_data)
# updateNewsTicker('CMCSA')
# df = retrieveTweets("INTC" , from_time=datetime.datetime(2023 , 3 , 5),time_to=datetime.datetime(2023 , 3 , 10))
# print(df)
# updatePriceAction(twitter_data)

# dropTables(app=app)
# df = retrieveNews(twitter_data)

# df = pd.read_csv("data_GOOG_prices_1min.csv")
# print(len(df))
# df = retrievePriceAction(twitter_data, "Min5"  ,from_time=datetime.datetime(2020 , 3 , 5))
# print(df)
df = retrieveTweets(twitter_data)
df.to_csv("tweets_updated.csv" , index=False)
# df = retrieveNews('AAPL')
# # print(df)
# df.to_csv("data_AAPL_daily.csv" , index=False)
# stream = priceActionStream("GOOG" , 
#                             starting_date=datetime.datetime(2023,3,8),
#                             from_time= datetime.datetime(2023 , 3 , 1),
#                             online=False)



# while(1):
#     print(stream.next(60*30*2))
#     print("____")
#     time.sleep(0.1)
