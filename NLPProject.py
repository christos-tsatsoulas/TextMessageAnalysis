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
df["Message_clean"] = df["Message_clean"].str.replace('[^\w\s]','')
# View changes
df.head()

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
df.head()

# Create a function to remove HTML Tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Remove HTML from Message data
df["Message_clean"] = df["Message_clean"].apply(lambda text: remove_html(text))
# View changes
df.head()

# Import word_tokenize from nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
# Tokenize cleaned messages
df['Message_tokenized'] = df.apply(lambda x: nltk.word_tokenize(x['Message_clean']), axis=1)
df.head()