# Import pandas with alias
import pandas as pd

# Import the SMS data as a pandas DataFrame
df = pd.read_csv('clean_nus_sms.csv', index_col=0)

# Check data
#df.head()

# View DataFrame shape
#df.info()

#Remove Blank Messages
df = df.dropna()
len(df)

# Set all characters to lower case in Message
df["Message_clean"] = df["Message"].str.lower()
# View changes
#df.head()

# Remove punctiation from Message variable
df["Message_clean"] = df["Message_clean"].str.replace('[^\w\s]','', regex=True)
# View changes
#df.head()

# Import NLTK library
import nltk

#Not Going to use next lines. Leave Stopwords In Messages
# Import stopwords
# from nltk.corpus import stopwords

# View stopwords in english
# ", ".join(stopwords.words('english'))

# Import re library for regular expressions
import re

# Create a function to remove url from Message data
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Remove URLs from Message data
df["Message_clean"] = df["Message_clean"].apply(lambda text: remove_urls(text))
# View changes
#df.head()

# Create a function to remove HTML Tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Remove HTML from Message data
df["Message_clean"] = df["Message_clean"].apply(lambda text: remove_html(text))
# View changes
#df.head()

# Import word_tokenize from nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
# Tokenize cleaned messages
df['Message_tokenized'] = df.apply(lambda x: nltk.word_tokenize(x['Message_clean']), axis=1)
#df.head()

# Save the preprocessed DataFrame
df.to_csv('processed_clean_nus_sms.csv', header = True)

# Import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Magic function for plotting in notebook
%matplotlib inline

# Count the number of unique countries
df['country'].nunique()

# View message count by country
df['country'].value_counts()
#df.head()

# Correct values
df = df.replace({'country':{'SG':'Singapore', 
                            'USA':'United States',
                            'india':'India',
                            'INDIA':'India',
                            'srilanka':'Sri Lanka',
                            'UK':'United Kingdom',
                            'BARBADOS':'Barbados',
                            'jamaica':'Jamaica',
                            'MY':'Malaysia',
                            'unknown':'Unknown'}})
#count the real number of unique countries
df['country'].nunique()

# View message count by country
df['country'].value_counts()

# Number of messages per country
country_value_counts = df['country'].value_counts() 

# Number of messages per country for the top 10 most active countries
top_10_country_value_counts = country_value_counts.head(10) 

# Plot a bar chart using pandas built-in plotting apis
top_10_country_value_counts.plot.barh()

# Download twitter data and sentiment analysis model
nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# import the twitter data
from nltk.corpus import twitter_samples

# Instantiate positive tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
# Instantiate negative tweets
negative_tweets = twitter_samples.strings('negative_tweets.json')

# View first positive tweet
#print(positive_tweets[0])
# View number of positive tweets
#print('There are {} positive tweets.'.format(len(positive_tweets)))

# View first negative tweet
#print(negative_tweets[0])
# View number of negative tweets
#print('There are {} negative tweets.'.format(len(negative_tweets)))

# Create tokens from the positive tweets
pos_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
# Show the first tokenized tweet
#print(pos_tweet_tokens[0])

# Create tokens from the positive tweets
neg_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
# Show the first tokenized tweet
#print(neg_tweet_tokens[0])

# Define lists for preprocessed tokens
positive_cleaned_tweets_list = []
negative_cleaned_tweets_list = []

# Positive tokens
for tweet in pos_tweet_tokens:
    cleaned_tweet = []
    for token in tweet:
        # Remove URLs
        url_cleaned = remove_urls(token)
        # Remove HTML 
        html_cleaned = remove_html(url_cleaned)
        cleaned_tweet.append(html_cleaned)
    # Add to list
    positive_cleaned_tweets_list.append(cleaned_tweet)

# Negative tokens
for tweet in neg_tweet_tokens:
    cleaned_tweet = []
    for token in tweet:
        # Remove URLs
        url_cleaned = remove_urls(token)
        # Remove HTML 
        html_cleaned = remove_html(url_cleaned)
        cleaned_tweet.append(html_cleaned)
    # Add to list
    negative_cleaned_tweets_list.append(cleaned_tweet)

# Print preprocessed token lists
#print(positive_cleaned_tweets_list[:5])
#print(negative_cleaned_tweets_list[:5])

#create a list of all the tweets
list_of_all_tweets = positive_cleaned_tweets_list + negative_cleaned_tweets_list

#turn my list of lists into a flat list of tokens
all_tweet_tokens = [token for sublist in list_of_all_tweets for token in sublist]

#check how many words that is
len(all_tweet_tokens)

#create a frequency distribution of all the words.
all_tokens = nltk.FreqDist(token for token in all_tweet_tokens)

#inspect the result
#print(len(all_tokens))
#print(all_tokens.most_common(10))

#select the top 10,000 words to be our features
sentiment_features = [word for (word, freq) in all_tokens.most_common(10000)]

#check what this list looks like
sentiment_features[:5]