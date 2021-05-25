#For first module
import tweepy,json,pandas,csv,nltk,string,re  # for pattern matching, used for removing URLs, Usernames and Punctuations
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup


#For second module
from nltk.corpus import stopwords # for removing stopwords
from nltk.tokenize import RegexpTokenizer # for tokenizing
from nltk.stem import WordNetLemmatizer # for lemmatizing
from nltk.stem.porter import PorterStemmer # for stemming


#For third moodule
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics



#For fifth module
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score


def search_tokens():
    url='https://www.thesaurus.com/browse/happy?s=t'  
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
    t_d=['sad']
    for i in h1:
        #z='#'+i.text
        t_d.append(i.text)
    return t_d


def raw_polished():     #raw data from Twitter is converted to a polished version where noise and other metadatas are removed
    ok=False
    l,ml,cnt=[],[],0
    with open('Plus.txt','r') as file:
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
    with open('Plusconv.csv','w') as f:
        write = csv.writer(f) 
        write.writerow(['POST','DESCRIPTION'])  #the different tags to choose from are present in the update log 2021.02.06
        write.writerows(ml)
    
  
    
def PHASE_1(READ=True):
    if(READ):                                           #since this process can be time consuming, data collection from Twitter everytime can
                                                        #be skipped by passing False as parameter 
        access_token="1356115757050937347-hIaRlwZpgGafrGhaMHjruo72PUGf4v"
        access_token_secret="GstHAPyelXZlZtFsXu0ieAEQlX4BPUfZpnqjoYRCWrPrq"
        consumer_key="LUE8ITDfE08n3ERjWpXyx0ZWc"
        consumer_secret="ydOXjuc6RhYKogpJ42fKNodT8qV5NTXzoZFny5LAKFOo4VubqR"
        auth= tweepy.OAuthHandler(consumer_key,consumer_secret)
        auth.set_access_token(access_token,access_token_secret)
        tweet_list=[]
        class MyStreamListener(tweepy.StreamListener):
            def __init__(self,api=None):
                super(MyStreamListener,self).__init__()
                self.num_tweets=0
                self.file=open("Plus.txt","w")
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
        tweets_data_path='Plus.txt'
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

        
def remove_un(text):       #removing noise from the data
    
    free= re.sub(r'http\S+', '', text)
    free= re.sub(r'u0+','',free)
    free= re.sub(r'\\ud\S+', '', free)
    free= re.sub(r'\xa0\S+', '', free)
    free= re.sub(r'pic.\S+', '', free)
    free= re.sub('@[\w]+','',free)         #removing usernames
    free= "".join([char for char in free if char not in string.punctuation]) #removing punctuations
    return free 

def stop_lemm_stem(text):   #removing stopwords , lemmatizing and stemwords
    
    hs= (open('hindiswords.txt','r').readlines())
    free=[word for word in text if word not in stopwords.words('english')+stopwords.words('french')+hs]
    lemmatizer=WordNetLemmatizer()
    free=[lemmatizer.lemmatize(word) for word in free]
    stemmer=PorterStemmer()
    free=" ".join([stemmer.stem(word) for word in free])
    return free

    
def PHASE_2(PROCESS=True):
    if(nltk.download('stopwords') and nltk.download('wordnet')):
        print("prerequistes checked")
        data=pd.read_csv("Depression_Tweet.csv")
        data.shape 
        pd.set_option('display.max_colwidth',None)
        
        data['POST']=[str(tweet) for tweet in data['POST']]
        
        data['POST']=data['POST'].apply(lambda x:remove_un(x)) #list(data['TWEET'])
        
        tokenizer=RegexpTokenizer(r'\w+')
        data['POST']=data['POST'].apply(lambda x:tokenizer.tokenize(x.lower()))
        
        data['POST']=data['POST'].apply(lambda x:stop_lemm_stem(x))
        print(data) #to check the final contents of preprocessed data
        data.to_csv('preprocessed.csv')


        
def Train(data,labels,text_clf,tp):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    score = 'f1_macro'
    print("# Tuning hyper-parameters for %s" % score,end='\n\n')

    np.errstate(divide='ignore')
    
    clf = GridSearchCV(text_clf, tp, cv=10, scoring=score)
    clf.fit(x_train, y_train)

    print("Best parameters set found for the model:",end='\n\n')
    print(clf.best_params_,end='\n\n')
    print("Grid scores for development set:",end='\n\n')
    
    for mean, std, params in zip(clf.cv_results_['mean_test_score'], 
                             clf.cv_results_['std_test_score'], 
                             clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("\nClassification report:\n\nThe model is trained on the full development set.\nThe scores are computed on the full evaluation set.",end='\n\n')

    print(classification_report(y_test, clf.predict(x_test), digits=4),end='\n\n')
    return (clf,x_test,y_test)


def Test(Z):
    clf,x_test,y_test = Z[0] , Z[1] , Z[2]
    y_pred = clf.predict(x_test)
    prediction = pd.DataFrame({'Tweets (Test)': [i for i in x_test], 'Label (Predicted)': [j for j in y_pred]})
    prediction.head(10)     #checking the predicted values
    return y_pred
       
    
def PHASE_3(Imp=True):
    if(Imp):
        text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

        tuned_parameters = {
            'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': [1, 1e-1]      }
        df=pd.read_csv('Depression_Tweet_Preprocessed.csv')
        #df.head(20)    #To check if the the file is perfect and the extraction process is successful
        train,ch=(),'y'
        
        train=Train(list(df.POST.fillna(' ')),df['LABEL'],text_clf,tuned_parameters)  #returns clf , x_test , y_test
        
        test=Test(train)    #returns y_pred
        
        #ch=input("Do you want to compare the predicted and original results?")
        if(ch=='y'):
            comparison = pd.DataFrame({'Actual Label': train[2], 'Predicted Label': test})
            print(comparison)
            print('mean abs error: ', metrics.mean_absolute_error(train[2],test))
            print("R2 score = ", round(metrics.r2_score(train[2], test), 2))
            print('confusion matrix:\n ' ,confusion_matrix(train[2], test, labels=[1,0]))
        
        #print(train[1],train[2],test,sep='\n' )
        
        joblib.dump(train[0],'trained_GS_96.pk1')   #trained model with 96% accuracy
        Temp = {'x_test': train[1] , 'y_test':train[2] , 'y_pred':test }
        (pd.DataFrame(Temp)).to_csv('Buffer_df_M3.csv')
        #return Z   #returns clf,x_test,y_test,y_pred
    
    
def PHASE_4(newdata=True):             #checking any other dataset for classification
    if(newdata):
        clf=joblib.load('trained_GS_96.pk1')    
        check=pd.read_csv('preprocessed.csv')
        predicted_label=clf.predict(check.POST.fillna(' '))
        predicted_sentiment = list(map(lambda lis : 'Not Depressed' if (lis==1) else 'Depressed' , predicted_label) )
        #just to test which statement is termed to be depressed and which one is not 
        #for t,s in zip(check.POST.fillna(' '),predicted_sentiment):
        #    print('{}\nClassified as: {} tweet\n\n'.format(t,s))
        
    
def plot_roc(fpr, tpr):
    # calculate the fpr and tpr for all thresholds of the classification
    plt.plot(fpr, tpr, color='#0077b6', label='ROC')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    

def PHASE_5(disp=True):
    if(disp):
        clf=joblib.load('trained_GS_96.pk1')
        Z=pd.read_csv('Buffer_df_M3.csv')
        x_test,y_test,y_pred=Z.x_test,Z.y_test,Z.y_pred
        
        
        df=pd.read_csv('preprocessed.csv')
        text = " ".join(review for review in df.POST.fillna(' '))
        print ("\n\nThere are {} words in the combination of all TWEETS.".format(len(text)))
        
        wordcloud = WordCloud(background_color="white",colormap='cividis',random_state=42).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        text0 = " ".join(review for review in df.POST[df['LABEL']==0])
        print ("There are {} words in the combination of all DEPRESSED TWEETS.".format(len(text0)))
        wordcloud = WordCloud(background_color='white', colormap='winter', random_state=34).generate(text0)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        text1 = " ".join(str(review) for review in df.POST[df['LABEL']==1])
        print ("There are {} words in the combination of all NOT DEPRESSED TWEETS.".format(len(text1)))
        wordcloud = WordCloud(background_color="white", colormap='autumn', random_state=34).generate(text1) 
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        fpr, tpr, thresholds=roc_curve(y_test, y_pred)
        plot_roc(fpr, tpr)
                
        
        average_precision = average_precision_score(y_test, y_pred)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        disp = plot_precision_recall_curve(clf,x_test,y_test)
        print()
        disp.ax_.set_title('\nPrecision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision), size=18, pad='10')
    

def main():
    PHASE_1(False)      #data collection phase
    PHASE_2(False)      #data preprocessing
    PHASE_3(False)      #training and TFIDF
    PHASE_4()      #testing with a new dataset
    PHASE_5()      #visualising and comparison
    print('All complete')      #after training is completed PHASE 4 and PHASE 5 can run with any dataset present, remaining modules are optional
if __name__ == "__main__":
    main()
