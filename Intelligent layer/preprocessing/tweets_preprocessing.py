'''
- THIS SCRIPT IS INTENDED TO PREPROCESS, DERIVE  AND RESAMPLE TWEETS PULLED FROM THE 
TWITTER API WITH THE ACTION PRICES PULLED FROM ALPHA VANTAGE.

- THIS SCRIPT USES THE 5MIN STOCK DATA INTERVAL FOR THE PURPOSE OF EXPERIMENTING 
WITH DIFFERENT INTERVALS THROUGHOUT THE PROJECT. 

- THIS SCRIPT ASSUMES THAT THE CSV FILES NEEDED WITH THE DATA ARE 
ALREADY AVAILABLE  IN THIS FOLDER AND PULLED VIA THE program.py SCRIPT THAT CAN 
BE FOUND IN THE 'Data Acquisition layer' 
'''
# our imports 
import pandas as pd
import re
from tqdm.notebook import tqdm
#from transformers import pipeline


class TweetsSentiment:
    def __init__(self, tweets,
                 stocks, 
                 ) -> None:
        
        self.stocks = stocks
        self.tweets = tweets

    def tweets_cleaner(self):
        '''
        This function uses regular expression to clean our tweets
        It also deals with handling duplicates
        '''
        all_tweets = self.tweets
        for i,s in enumerate(tqdm(all_tweets['Text'])):
            try :
                a = all_tweets.loc[i, 'Text']
                a = a.replace("#", "")
                a = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', a, flags=re.MULTILINE)
                a = re.sub('@\\w+ *', '', a, flags=re.MULTILINE)
                all_tweets.loc[i, 'Text'] = a

            except:
                all_tweets.drop(i)
                pass

        #handling duplicates
        print('tweet shape before droping duplicates', all_tweets.shape)
        duplicates_removed = all_tweets.shape[0]
        all_tweets = all_tweets.drop_duplicates(subset=['ID'])
        duplicates_removed -= all_tweets.shape[0]
        print('tweet shape after droping duplicates', all_tweets.shape)
        print('duplicates removed', duplicates_removed)

        self.tweets = all_tweets

        return
    
    def get_df_per_ticker(self, ticker_name):
        '''
        we want to run our experiment per tickers; 
        each ticker has a different time series distribution
        '''
        ticker_df = pd.read_csv('sentiment_scores.csv', encoding='utf-8')
        ticker_df = ticker_df[ticker_df['Ticker'] == ticker_name]

        #keep only needed columns 
        ticker_df = pd.DataFrame(ticker_df[['Text','CreatedAt','Ticker','sentiment_score']])
        
        # add an empty column in which sentiment would be added 
        #ticker_df["sentiment_score"] = ''

        stock_df = self.stocks[self.stocks['Ticker'] == ticker_name]
        stock_df['Date'] = pd.to_datetime(stock_df['timestamp'])
        return ticker_df,stock_df
    
    def distilbert(ticker_df):
        '''
        this function applies the distlibert model to get 
        the sentimet score.

        From our previously analysed models distilbert which 
        is a transformer based model gave us the most accurate
        sentiment when it was tested on labeled tweets datasets 
        '''
        #classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
        for index, row in tqdm(ticker_df.iterrows(), total=len(ticker_df)):
            text = row['Text']
            #score = classifier(text)
            #row["sentiment_score"] = score[0]['score']

        #we save our result to avoid rerunning the model everytime
        ticker_df.to_csv('sentiment_scores.csv')
        return
    
    def group_by_Min5(self , ticker_df):
        '''
        this function is intended to group the tweets by 5 minutes and get the mean score
        of sentiment
        '''
        
        ticker_df['Date'] = pd.to_datetime(ticker_df['CreatedAt']).dt.round('min')
        twitter_df = pd.DataFrame(ticker_df[['Date', 'Text', 'sentiment_score']])
        print("Original DataFrame:")
        print(twitter_df)

        twitter_df.set_index('Date', inplace=True)
        print("DataFrame with 'Date' as index:")
        print(twitter_df)

        twitter_df = twitter_df.resample("5T").mean()
        print("Resampled DataFrame:")
        print(twitter_df)


        return twitter_df
    
    def combine_datasets(self,ticker_df, stock_df, ticker_name, combine_flag = True):
        '''
        this function combines stock data with sentiment and tweets  
        '''
        ticker_df = pd.DataFrame(ticker_df)
        stock_df = pd.DataFrame(stock_df)
        #stock_df.set_index('Date')
        final_df = stock_df.join(ticker_df,how='left',on='Date',lsuffix='_stock', rsuffix='_ticker')
        final_df = final_df.dropna()
        final_df = final_df.drop(['timestamp'], axis=1)
        # if I just want the stock we will drop the sentiment
        if combine_flag == False:
            final_df = final_df.drop(columns=['sentiment_score'])
            final_df.to_csv('finaldf_without_sentiment_{}.csv'.format(ticker_name))
        else:    
            final_df.to_csv('finaldf_with_sentiment_{}.csv'.format(ticker_name))    
        return final_df
    
    def start_point(self):
        self.tweets = pd.read_csv('sentiment_scores.csv', encoding='utf-8')
        self.stocks = pd.read_csv('price_action_Min5.csv', encoding='utf-8')

        #prepare files needed for training
        tickers = self.tweets['Ticker'].dropna().unique()

        #self.tweets_cleaner()
        #self.distilbert(self.tweets)
        for ticker in tickers:
            print(ticker)
            ticker_df , stock_df = self.get_df_per_ticker(ticker)
            twitter_df = self.group_by_Min5(ticker_df)
            print(twitter_df,stock_df)
            self.combine_datasets(twitter_df,stock_df,ticker)
            self.combine_datasets(twitter_df,stock_df,ticker,combine_flag=False)

        return
    
test = TweetsSentiment(None, None)
print(test.start_point())
