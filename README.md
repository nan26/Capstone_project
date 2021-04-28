# Classification and Product Recommendation System 

  E-commerce and retail companies are utilising the power of data and boosting sales by implementing recommender systems on their websites. The use cases of these systems have been steadily increasing in the last few years and it’s a great time to explore more into some of these machine learning techniques for recommendation systems.

  There is wide variety of online shopping platforms such as Shopee, Lazada, Zalora,etc that provides humongous amount of products. One of the biggest online shopping website is Amazon. Amazon sells morethan 12 million products, books, media, wine, and services source. Here I am using the amazon review dataset for Luxury Beauty products.The dataset can be downloaded from http://jmcauley.ucsd.edu/data/amazon/. I am considering the reviews and ratings given by the user to different products as well as his/her reviews about his/her experience with the product(s).

   In this project, I have classified the reviews using NLP techniques as 'Positive' or 'Negative' sentiments. Sentiment analysis is performed on review text to analyse the sentiment of user and therby used the polarity of the review to provide quality recommendations to the users. Moreover, I have also designed item-based collaborative filtering model using k-Nearest Neighbors to find top 10 most similar items.


**EXECUTIVE SUMMARY**

  Users need to go through humongous amount of products and reviews before finding the best one from this big catalogue.The need for a strong qualitative recommendation system is increasing day by day as we greatly depend on online platform these days. This can be done by analysing the sentiments of the users in the reviews to extract the qualitative feedback so as to feed it into the recommendation system. In this project, I have classified and analysed the sentiments of users and provided recommendations based on polarity.


**LOAD DATASET**

    with open("gdrive/My Drive/Capstone_project/data/Luxury_Beauty.json") as f:
     dataframe = pd.DataFrame([json.loads(l) for l in f.readlines()])
     dataframe.to_csv('gdrive/My Drive/Capstone_project/data/lb_reviews.csv', sep=',', index=False)

    with open("gdrive/My Drive/Capstone_project/data/meta_Luxury_Beauty.json") as f:
     dataframe = pd.DataFrame([json.loads(l) for l in f.readlines()])
    dataframe.to_csv('gdrive/My Drive/Capstone_project/data/meta_lb_reviews.csv', sep=',', index=False)

**SKLEARN MODELS**

    Logistic Regression with Count Vectorizer

    Logistic Regression with TFIDF

    Multinomial NB 

    Linear SVC


**NEURAL NETWORK MODELS**

    BiLSTM

    BERT using Ktrain Huggingface Transformers


**RECOMMENDATION SYSTEMS**

    Popularity Recommender System

    Popularity Recommender System Using Sentiment Analysis

    Content Based Recommendation

    Collaborative Recommendation System


**REFERENCES:**

    1) Text Classification https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb

    2) Content-Based Recommendation System https://medium.com/@bindhubalu/content-based-recommender-system-4db1b3de03e7

    3) Introduction to Recommender System https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26

    4) Classification Accuracy is Not Enough: More Performance Measures You Can Use https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/

    5) Building a Text Classification model using BiLSTM https://medium.com/analytics-vidhya/building-a-text-classification-model-using-bilstm-c0548ace26f2

