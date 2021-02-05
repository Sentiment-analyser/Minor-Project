import tweepy,json,pandas
access_token="1356115757050937347-QCY7yAQu89RofRCbKkJxp3HMnQWnBQ"
access_token_secret="ULHsx0VzLBreR7aZ21u5SUFVTRPRdx9xXQ0XQeRVbBYT8"
consumer_key="jUZIb0X73d14NEWZHYMCHx2Vn"
consumer_secret="CGgQ9sOPHlUSpqufo39c63lRs4fN5JuSo8s5DmrVzd7WgO6FKF"
auth= tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
tweet_list=[]
class MyStreamListener(tweepy.StreamListener):
    def __init__(self,api=None):
        super(MyStreamListener,self).__init__()
        self.num_tweets=0
        self.file=open("tweet.txt","w")
    def on_status(self,status):
        tweet=status._json
        self.file.write(json.dumps(tweet)+ '\n')
        tweet_list.append(status)
        self.num_tweets+=1
        if self.num_tweets<1000:
            return True
        else:
            return False
        self.file.close()
l = MyStreamListener()
stream =tweepy.Stream(auth,l)
stream.filter(track=['depression','suicide','sad','#depression','#hopeless'])
tweets_data_path='tweet.txt'
tweets_data=[]
tweets_file=open(tweets_data_path,"r")
#read in tweets and store on list
for line in tweets_file[]:
    tweet=json.loads(line)
    tweets_data.append(tweet)
tweets_file.close()
print(tweets_data[0])

#data collection unit
