import tweepy
import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from textblob import TextBlob
import matplotlib.pyplot as plt

#Google is 730,  GS is 250, MS is 90, Intel is 49

consumer_key= ''
consumer_secret= ''
access_token=''
access_token_secret=''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

public_tweets = api.search('Intel')



for tweet in public_tweets:    
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    


dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return


def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates),1))

    svr_lin = SVR(kernel= 'linear' , C=1e3)
    svr_poly = SVR(kernel= 'poly' , C=1e3, degree = 2)
    svr_rbf = SVR(kernel= 'rbf' , C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Machine')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('googl.csv')

predicted_price = predict_prices(dates, prices,8);
type(predicted_price);

#col = ['c1' , 'c2', 'c3']
df =pd.DataFrame(list(predicted_price));
print(df)

test = df.iloc[0];
accuracy = ((test - 1150) / 1150) * 100;
print("Accuracy:", 100 - accuracy);
