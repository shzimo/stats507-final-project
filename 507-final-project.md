# Amazon Review Analysis Based on Beauty Products: Sentiment, Textual Features, and Product Quality

This notebook analyzes Amazon consumer reviews based on beauty products to explore how textual feedback correlates with perceived product quality. We use transformer-based models (BERT) for sentence embeddings and regression analysis to evaluate rating deviations.

## Project Scope: Beauty Product Reviews

This project focuses on customer reviews of beauty products from Amazon. We analyze how review text reflects perceived product quality through sentiment analysis, rating deviation modeling, and regression using transformer-based embeddings.


```python
# Neccessary Libraries
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from wordcloud import WordCloud
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
```

## Exploratory Data Analysis (EDA)

Before modeling, we explore basic properties of the dataset to understand rating distributions, review lengths, and patterns across categories. 



```python
# Load raw review data
dataset_beauty = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
print(dataset_beauty["full"][0])
```

    {'rating': 5.0, 'title': 'Such a lovely scent but not overpowering.', 'text': "This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!", 'images': [], 'asin': 'B00YQ6X8EO', 'parent_asin': 'B00YQ6X8EO', 'user_id': 'AGKHLEW2SOWHNMFQIJGBECAF7INQ', 'timestamp': 1588687728923, 'helpful_vote': 0, 'verified_purchase': True}



```python
# Convert the full split to a DataFrame
df_beauty = dataset_beauty['full'].to_pandas()
df_beauty.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>title</th>
      <th>text</th>
      <th>images</th>
      <th>asin</th>
      <th>parent_asin</th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>helpful_vote</th>
      <th>verified_purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>Such a lovely scent but not overpowering.</td>
      <td>This spray is really nice. It smells really go...</td>
      <td>[]</td>
      <td>B00YQ6X8EO</td>
      <td>B00YQ6X8EO</td>
      <td>AGKHLEW2SOWHNMFQIJGBECAF7INQ</td>
      <td>1588687728923</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>Works great but smells a little weird.</td>
      <td>This product does what I need it to do, I just...</td>
      <td>[]</td>
      <td>B081TJ8YS3</td>
      <td>B081TJ8YS3</td>
      <td>AGKHLEW2SOWHNMFQIJGBECAF7INQ</td>
      <td>1588615855070</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>Yes!</td>
      <td>Smells good, feels great!</td>
      <td>[]</td>
      <td>B07PNNCSP9</td>
      <td>B097R46CSY</td>
      <td>AE74DYR3QUGVPZJ3P7RFWBGIX7XQ</td>
      <td>1589665266052</td>
      <td>2</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>Synthetic feeling</td>
      <td>Felt synthetic</td>
      <td>[]</td>
      <td>B09JS339BZ</td>
      <td>B09JS339BZ</td>
      <td>AFQLNQNQYFWQZPJQZS6V3NZU4QBQ</td>
      <td>1643393630220</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>A+</td>
      <td>Love it</td>
      <td>[]</td>
      <td>B08BZ63GMJ</td>
      <td>B08BZ63GMJ</td>
      <td>AFQLNQNQYFWQZPJQZS6V3NZU4QBQ</td>
      <td>1609322563534</td>
      <td>0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_beauty = df_beauty[['asin', 'rating', 'title', 'text', 'helpful_vote', 'verified_purchase']]
df_beauty['text'] = df_beauty['text'].astype(str)
df_beauty['review_length'] = df_beauty['text'].apply(lambda x: len(x.split()))
```


```python
sample = df_beauty.iloc[0]
print(f"Rating: {sample['rating']}")
print(f"Title: {sample['title']}")
print(f"Text: {sample['text']}")
```

    Rating: 5.0
    Title: Such a lovely scent but not overpowering.
    Text: This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!



```python
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_beauty, x="verified_purchase", y="rating", palette="coolwarm")
plt.title("Boxplot of Star Ratings by Verified Purchase")
plt.xlabel("Verified Purchase")
plt.ylabel("Star Rating")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("rating_boxplot_verified.png")
plt.show()
```

    /var/folders/35/481h5z3934x7111xbgyyz4540000gn/T/ipykernel_14366/3795453360.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(data=df_beauty, x="verified_purchase", y="rating", palette="coolwarm")



    
![png](output_8_1.png)
    



```python
# Filter 5-star and 1-star reviews
five_star_reviews = df_beauty[df_beauty["rating"] == 5.0]["text"].dropna().astype(str)
one_star_reviews = df_beauty[df_beauty["rating"] == 1.0]["text"].dropna().astype(str)

# Join text
five_star_text = " ".join(five_star_reviews.tolist())
one_star_text = " ".join(one_star_reviews.tolist())

# Generate WordClouds
wc_five = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(five_star_text)
wc_one = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(one_star_text)

# Plot side-by-side
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(wc_five, interpolation='bilinear')
plt.axis("off")
plt.title("5-Star Review Word Cloud", fontsize=16)

plt.subplot(1, 2, 2)
plt.imshow(wc_one, interpolation='bilinear')
plt.axis("off")
plt.title("1-Star Review Word Cloud", fontsize=16)

plt.tight_layout()
plt.savefig("wordcloud_comparison.png")
plt.show()
```


    
![png](output_9_0.png)
    



```python
# Ratings
plt.figure(figsize=(6, 4))
sns.histplot(df_beauty['rating'], bins=5, color='lightcoral')
plt.title("Beauty Product Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```


    
![png](output_10_0.png)
    



```python
# Review length
plt.figure(figsize=(6, 4))
sns.histplot(df_beauty['review_length'], bins=50, color='steelblue')
plt.title("Review Length Distribution")
plt.xlabel("Words per Review")
plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    



```python
# Helpful votes
plt.figure(figsize=(6, 4))
sns.histplot(df_beauty['helpful_vote'], bins=50)
plt.title("Helpful Votes Distribution")
plt.xlabel("Helpful Votes")
plt.yscale("log")
plt.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    


## Merge with Product Metadata

We merge each review with its corresponding product's average rating (from metadata) to calculate rating deviation — the difference between a user's rating and the average rating of the product.



```python
print("Merging with metadata...")
meta_beauty = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True).to_pandas()
meta_beauty = meta_beauty[['parent_asin', 'average_rating']].dropna()

df_merge = pd.merge(df_beauty, meta_beauty, left_on='asin', right_on='parent_asin', how='left').dropna()
df_merge['rating_deviation'] = df_merge['rating'] - df_merge['average_rating']
df_merge = df_merge[df_merge['text'].str.len() > 20].reset_index(drop=True)

df_merge.head()
```

    Merging with metadata...





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>asin</th>
      <th>rating</th>
      <th>title</th>
      <th>text</th>
      <th>helpful_vote</th>
      <th>verified_purchase</th>
      <th>review_length</th>
      <th>parent_asin</th>
      <th>average_rating</th>
      <th>rating_deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B00YQ6X8EO</td>
      <td>5.0</td>
      <td>Such a lovely scent but not overpowering.</td>
      <td>This spray is really nice. It smells really go...</td>
      <td>0</td>
      <td>True</td>
      <td>61</td>
      <td>B00YQ6X8EO</td>
      <td>4.3</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B081TJ8YS3</td>
      <td>4.0</td>
      <td>Works great but smells a little weird.</td>
      <td>This product does what I need it to do, I just...</td>
      <td>1</td>
      <td>True</td>
      <td>47</td>
      <td>B081TJ8YS3</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B00R8DXL44</td>
      <td>4.0</td>
      <td>Pretty Color</td>
      <td>The polish was quiet thick and did not apply s...</td>
      <td>0</td>
      <td>True</td>
      <td>24</td>
      <td>B00R8DXL44</td>
      <td>3.8</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B099DRHW5V</td>
      <td>5.0</td>
      <td>Handy</td>
      <td>Great for many tasks.  I purchased these for m...</td>
      <td>0</td>
      <td>True</td>
      <td>22</td>
      <td>B099DRHW5V</td>
      <td>3.5</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B08P2DZB4X</td>
      <td>5.0</td>
      <td>Great for at home use and so easy to use!</td>
      <td>This is perfect for my between salon visits. I...</td>
      <td>0</td>
      <td>False</td>
      <td>72</td>
      <td>B08P2DZB4X</td>
      <td>3.8</td>
      <td>1.2</td>
    </tr>
  </tbody>
</table>
</div>



## Generate BERT Embeddings for Review Text

We use `sentence-transformers` to generate dense vector representations of each review. These embeddings capture semantic meaning for use in downstream modeling.



```python
df_sample = df_merge.sample(10000, random_state=42).reset_index(drop=True)

print("Generating BERT embeddings for 10,000 reviews...")
model_bert = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model_bert.encode(df_sample['text'].tolist(), batch_size=32, show_progress_bar=True)
```

    Generating BERT embeddings for 10,000 reviews...



    Batches:   0%|          | 0/313 [00:00<?, ?it/s]


## Fit Linear Regression to Predict Rating Deviation

We use the BERT embeddings as input features to predict rating deviation using a simple linear regression model.



```python
print("Fitting regression model...")
X = np.array(embeddings)
y = df_sample['rating_deviation'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("Regression R² score with BERT embeddings:", r2)
```

    Fitting regression model...
    Regression R² score with BERT embeddings: 0.3540727775143434


### Visualize Predictions
A scatterplot of predicted vs. actual values to see how well the model fit.


```python
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Rating Deviation")
plt.ylabel("Predicted Deviation")
plt.title("Predicted vs. Actual Rating Deviation")
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    


### Error Analysis
Check examples where the model was most wrong.


```python
df_sample['prediction'] = reg.predict(X)
df_sample['abs_error'] = np.abs(df_sample['prediction'] - df_sample['rating_deviation'])

# Top 5 highest error reviews
df_sample.sort_values('abs_error', ascending=False)[['text', 'rating', 'prediction', 'rating_deviation', 'abs_error']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>rating</th>
      <th>prediction</th>
      <th>rating_deviation</th>
      <th>abs_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9145</th>
      <td>Still trying it but not see too much effect</td>
      <td>5.0</td>
      <td>-2.403677</td>
      <td>2.3</td>
      <td>4.703677</td>
    </tr>
    <tr>
      <th>9019</th>
      <td>I couldn't closed it, so didn’t work.&lt;br /&gt;—-&gt;...</td>
      <td>4.0</td>
      <td>-3.335464</td>
      <td>0.6</td>
      <td>3.935464</td>
    </tr>
    <tr>
      <th>9808</th>
      <td>Waste of money. Very useless product.</td>
      <td>1.0</td>
      <td>-3.806041</td>
      <td>0.0</td>
      <td>3.806041</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>This hair ties were very useful as elastic for...</td>
      <td>1.0</td>
      <td>0.200914</td>
      <td>-3.6</td>
      <td>3.800914</td>
    </tr>
    <tr>
      <th>9826</th>
      <td>very ackward to use/hold, perhaps a little to ...</td>
      <td>1.0</td>
      <td>0.749705</td>
      <td>-3.0</td>
      <td>3.749705</td>
    </tr>
  </tbody>
</table>
</div>



### Compare Verified vs. Unverified Reviews
We check if verified reviews are more consistent with average product rating.


```python
sns.boxplot(data=df_sample, x='verified_purchase', y='rating_deviation')
plt.title("Rating Deviation by Verified Purchase Status")
plt.show()
```


    
![png](output_24_0.png)
    


#  Lasso Regression on BERT Embeddings


```python
X = np.array(embeddings)
y = df_sample['rating_deviation'].values

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Lasso
lasso = Lasso(alpha=0.1)  # Try alpha=0.01, 0.1, 1.0 as needed
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Lasso Regression R² score:", r2)
print("Lasso RMSE:", rmse)
```

    Lasso Regression R² score: -0.0007740937169409268
    Lasso RMSE: 1.391047602757293


    /opt/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(


Lasso yielded a negative R^2 value (-0.0008), suggesting that it underfit the data by overly penalizing model complexity.

# Ridge Regression on BERT Embeddings


```python
ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# Evaluation
print("Ridge Regression R^2 score:", r2_score(y_test, y_pred))
print("Ridge RMSE:", mean_squared_error(y_test, y_pred, squared=False))
```

    Ridge Regression R^2 score: 0.38200516346511026
    Ridge RMSE: 1.093116344738257


    /opt/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(


Ridge regression delivered the best performance, achieving an R^2 score of 0.382 and RMSE of 1.093. This suggests that BERT-derived semantic features explain over one-third of the variation in rating deviation, validating their relevance to perceived product quality. 


```python

```
