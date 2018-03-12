
# News Mood
- In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

- Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: BBC, CBS, CNN, Fox, and New York times.

The first plot will be and/or feature the following:

- Be a scatter plot of sentiments of the last 100 tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
- Each plot point will reflect the compound sentiment of a tweet.
- Sort each plot point by its relative timestamp.
The second plot will be a bar plot visualizing the overall sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

- Pull last 100 tweets from each outlet.
- Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
- Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
- Export the data in the DataFrame into a CSV file.
- Save PNG images for each plot.
As final considerations:

Use the Matplotlib and Seaborn libraries.
- Include a written description of three observable trends based on the data.
- Include proper labeling of your plots, including plot titles (with date of analysis) and axes labels.
- Include an exported markdown version of your Notebook called  README.md in your GitHub repository.

# Three Observable Trends
- All news agencies have a diverse spread of sentiment for their tweets, not concentrating in negative or positive
- Overall, the BBC has the most negative sentiment of the reviewed news agencies, while the NYT is the most neutral
- News agencies overall lean towards more negative sentiment than positive sentiment


```python
# Dependencies
import tweepy
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(os.environ['consumer_key'], os.environ['consumer_secret'])
auth.set_access_token(os.environ['access_token'], os.environ['access_token_secret'])
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
os.environ['access_token']
```




    '969394233080627200-ifMdn3ZOYNawHe1dCRFOm5FkhkFZ9HT'




```python
# Target Account
target_user = ("@BBCWorld","@CBSNews", "@CNN", "@FoxNews", "@nytimes")

# Variables for holding sentiments
sentiments = []
oldest_tweet = None

# Loop through all target users
for target in target_user:
    # Counter
    counter = 1
# Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):
        
    # Get all tweets from home feed
        public_tweets = api.user_timeline(target, 
                                          count=20,
                                         max_id=oldest_tweet)
        
    # Loop through all tweets 
        for tweet in public_tweets:
            
        # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            
            
        # Add sentiments for each tweet into an array
            sentiments.append({"User": tweet["user"]["screen_name"],
                               "Date": tweet["created_at"], 
                               "Text": tweet["text"],
                               "Compound": results["compound"],
                               "Positive": results["pos"],
                               "Negative": results["neu"],
                               "Neutral": results["neg"],
                               "Tweets Ago": counter})
            
            # Add to counter 
            counter = counter + 1  
        oldest_tweet = int(tweet['id_str']) - 1
```


```python
# Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)

#save DF to CSV
sentiments_pd.to_csv("Sentimentspd.csv")

#sentiments_pd.set_index("User", inplace=True)
sentiments_pd.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>Mon Mar 12 01:01:22 +0000 2018</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Parliament marriage proposal MP weds in Austra...</td>
      <td>1</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>Mon Mar 12 00:39:38 +0000 2018</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>How slime turned around the fate of a glue fac...</td>
      <td>2</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>Mon Mar 12 00:17:08 +0000 2018</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Helicopter crashes in New York City's East Riv...</td>
      <td>3</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>Sun Mar 11 21:48:23 +0000 2018</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Putin ordered plane to be downed in 2014 https...</td>
      <td>4</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>Sun Mar 11 21:04:29 +0000 2018</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Martin Selmayr: The man at the heart of a Brus...</td>
      <td>5</td>
      <td>BBCWorld</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Media specific Dataframes
bbc_sentiments = sentiments_pd[(sentiments_pd["User"] == "BBCWorld")]
cbs_sentiments = sentiments_pd[(sentiments_pd["User"] == "CBSNews")]
cnn_sentiments = sentiments_pd[(sentiments_pd["User"] == "CNN")]
fox_sentiments = sentiments_pd[(sentiments_pd["User"] == "FoxNews")]
nyt_sentiments = sentiments_pd[(sentiments_pd["User"] == "nytimes")]
nyt_sentiments.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>400</th>
      <td>0.4019</td>
      <td>Wed Mar 07 15:18:03 +0000 2018</td>
      <td>0.617</td>
      <td>0.136</td>
      <td>0.247</td>
      <td>When her father's request for "compassionate r...</td>
      <td>1</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>401</th>
      <td>0.4215</td>
      <td>Wed Mar 07 15:11:06 +0000 2018</td>
      <td>0.872</td>
      <td>0.000</td>
      <td>0.128</td>
      <td>EU officials unveiled an array of tariffs that...</td>
      <td>2</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>402</th>
      <td>0.0000</td>
      <td>Wed Mar 07 15:00:37 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>3</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>403</th>
      <td>-0.0258</td>
      <td>Wed Mar 07 14:50:11 +0000 2018</td>
      <td>0.717</td>
      <td>0.163</td>
      <td>0.120</td>
      <td>Jimmy Kimmel struck back at President Trump af...</td>
      <td>4</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>404</th>
      <td>0.7430</td>
      <td>Wed Mar 07 14:40:09 +0000 2018</td>
      <td>0.741</td>
      <td>0.000</td>
      <td>0.259</td>
      <td>The United States Holocaust Memorial Museum ha...</td>
      <td>5</td>
      <td>nytimes</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
sns.set()
fig.suptitle("Sentiment Analysis of Media Tweets 3/5/18", fontsize=16, fontweight="bold")

ax.set_xlim(100, 0)
plt.yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("Tweets Ago")
ax.set_ylabel("Tweet Polarity")
plt.grid(True)

#Plot BBC
plt.scatter(bbc_sentiments["Tweets Ago"], bbc_sentiments["Compound"], 
                                        marker="o", 
                                        facecolors=sns.xkcd_rgb["light blue"], 
                                        edgecolors="black", 
                                        alpha=0.75, 
                                        label="BBC")
#Plot CBS
plt.scatter(cbs_sentiments["Tweets Ago"], cbs_sentiments["Compound"], 
                                        marker="o", 
                                        facecolors=sns.xkcd_rgb["grass green"], 
                                        edgecolors="black", 
                                        alpha=0.75, 
                                        label="CBS")
#Plot CNN
plt.scatter(cnn_sentiments["Tweets Ago"], cnn_sentiments["Compound"], 
                                        marker="o", 
                                        facecolors=sns.xkcd_rgb["red orange"], 
                                        edgecolors="black", 
                                        alpha=0.75, 
                                        label="CNN")
#Plot Fox
plt.scatter(fox_sentiments["Tweets Ago"], fox_sentiments["Compound"], 
                                        marker="o", 
                                        facecolors=sns.xkcd_rgb["electric blue"], 
                                        edgecolors="black", 
                                        alpha=0.75, 
                                        label="Fox")
#Plot NYT
plt.scatter(nyt_sentiments["Tweets Ago"], nyt_sentiments["Compound"], 
                                        marker="o", 
                                        facecolors=sns.xkcd_rgb["light yellow"], 
                                        edgecolors="black", 
                                        alpha=0.75, 
                                        label="New York Times")

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig.savefig("Tweetyplot.png")
plt.show()
```


![png](output_8_0.png)



```python
# Store all values
sentiments_comp = (bbc_sentiments["Compound"].mean(), 
                   cbs_sentiments["Compound"].mean(), 
                   cnn_sentiments["Compound"].mean(), 
                   fox_sentiments["Compound"].mean(),
                   nyt_sentiments["Compound"].mean())

# Format data
fig, ax = plt.subplots()
ind = np.arange(len(sentiments_comp))  
width = 1
bbc_bar = ax.bar(ind[0], sentiments_comp[0], width, color=sns.xkcd_rgb["light blue"])
cbs_bar = ax.bar(ind[1], sentiments_comp[1], width, color=sns.xkcd_rgb["grass green"])
cnn_bar = ax.bar(ind[2], sentiments_comp[2], width, color=sns.xkcd_rgb["red orange"])
fox_bar = ax.bar(ind[3], sentiments_comp[3], width, color=sns.xkcd_rgb["electric blue"])
nyt_bar = ax.bar(ind[4], sentiments_comp[4], width, color=sns.xkcd_rgb["light yellow"])


# Orient widths. Add labels, tick marks, etc. 
ax.set_ylabel('Tweety Polarity')
ax.set_title('Overall Media Sentiment Based on Twitter 3/5/18')
ax.set_xticks(ind)
ax.set_xticklabels(('BBC', 'CBS', 'CNN', 'Fox', 'NYT'))
ax.set_autoscaley_on(False)
ax.set_ylim([-0.2,0.15])
ax.grid(False)

# Save the Figure
fig.savefig("Tweetychart.png")

# Show the Figure
plt.show()
```


![png](output_9_0.png)

