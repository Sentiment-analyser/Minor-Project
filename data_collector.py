import tweepy,json,pandas,csv
import urllib.request
from bs4 import BeautifulSoup



def search_tokens():
    url='https://www.thesaurus.com/browse/sad?s=t'  
    '''
    #IF you are willing to search for customised tags uncomment the code snippet, then enter a word and all related tags 
    #shall be automatically searched
    
    #By default all sad related tags are being searched
    
    n=input()
    url = 'https://www.thesaurus.com/browse/'+n+'?s=t'
    '''
    content = urllib.request.urlopen(url)
    read_content = content.read()         #data collection phase (data contains metadata and original data)
    
    #print(read_content)          #metadata check
    
    q,cnt=False,0
    # reference: <a font-weight="inherit" href="/browse/dismal" data-linkid="nn1ov4" class="css-1m14xsh eh475bn1">dismal<!-- --> </a>
    bs = BeautifulSoup(read_content, 'html.parser')
    h1 = bs.find_all('a', class_="css-1m14xsh eh475bn1")    #data filter phase
    t_d=[]
    for i in h1:
        t_d.append(i.text)
    return t_d




def raw_polished():     #raw data from Twitter is converted to a polished version where noise and other metadatas are removed
    ok=False
    l,ml,cnt=[],[],0
    with open('tweet.txt','r') as file:
        for line in file:
            for word in line.split():  
                if(word=='\"text\":' or word=='\"description\":'):
                    ok=True
                    w=''
                elif(word== '\"source\":'):
                    ok=False
                    l.append(w)
                elif(word=='\"translator_type\":'):
                    ok=False
                    l.append(w)
                    ml.append(l)
                    l=[]
                elif(ok):
                    w=w+word+' '
                    #print(w)      #debugging (check whether the variable is actually storing something)
    #print(ml)                     #debugging (check the final list for desired values) 
    with open('converted.csv','w') as f:
        write = csv.writer(f) 
        write.writerow(['POST','DESCRIPTION'])  #the different tags to choose from are present in the update log 2021.02.06
        write.writerows(ml)
    
    
    
def PHASE_1(READ=True):
    if(READ):                                           #since this process can be time consuming, data collection from Twitter everytime can
                                                        #be skipped by passing False as parameter 
        access_token="*"
        access_token_secret="*"
        consumer_key="*"
        consumer_secret="*"
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
        stream.filter(track=search_tokens())
        tweets_data_path='tweet.txt'
        tweets_data=[]
        tweets_file=open(tweets_data_path,"r")
        #read in tweets and store on list
        for line in tweets_file:
            tweet=json.loads(line)
            tweets_data.append(tweet)
        tweets_file.close()
        #print(tweets_data[0])         #printing the raw data collected freshly from Twitter via Tweepy
        '''read_file = pandas.read_csv ('tweet.txt')
        read_file.to_csv ('Untitled Folder/converted.csv', index=None)'''
        raw_polished()                #creates a CSV file after removing all the unnecessary data
    
def main():
    PHASE_1(False)      #data collection phase
    
if __name__ == "__main__":
    main()
    
#TO BE ADDED: 
#PREPROCESSING
