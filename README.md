# Recommendation Systems: Cold Start Problem Case Study

## Goal

Each user is potentially interested in watching one or more of the movies specified in `requests.json`, a file of users and movies. We used a combination of a matrix factorization model (ALS) and a cold start model (using existing ratings in `ratings.json` and movie metadata in `movies_metadata.csv`) to predict ratings for these movies and recommend movies that users would most likely want to see. 

## Methodology 

Using a combination of PySpark Dataframes, PySpark ML, NumPy, Pandas, and Sci-kit Learn, we first read in the `ratings.json` data and split into an 80% train set and a 20% test set. After some Spark dataframe cleaning and preparation, we initiated our `ratings_train` dataframe through PySpark ML's ALS model, fitting to the `ratings_train` set and yielding, via the transform method applied to the `requests` dataframe, movie recommendations for each combination of users and movies featured in `requests.json`. These initial recommendations were generated using matrix factorization, a collaborative filtering algorithm commonly used in recommendation systems that decomposes the existing user-item (movie fan-movie) interaction matrix in order to recommend new movies to fans we know a bit about already based on exhibited similarities to other users.

Collaborative Filtering simplified in a GIF:
<img src='https://upload.wikimedia.org/wikipedia/commons/5/52/Collaborative_filtering.gif'>


Whereas the ALS model uses prior explicitly stated information about what users have liked what movies to generate the most accurate predictions possible, occasionally we encounter a new user or a new movie for which we have no prior rating information. This is the cold start problem: being forced to generate a best initial guess of what movies to recommend to the user or what users to recommend the movie to even though we know (almost) nothing about them. In these cases we fall back on some pieces of info that would be input during user sign-up, such as age, gender, career, and zipcode. By running regression models (e.g. random forest regressors, gradient boosted regressors) on these, we are able to fill in the gaps left by our ALS model with educated guesses about what these new users might like. 
