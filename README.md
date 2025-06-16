# Recommendation-System-using-Collaborative-Filtering

COMPANY:CODTECH IT SOLUTIONS

NAME:KURUVA SHASHI KIRAN

INTERN ID:CT06DM211

DOMAIN:MACHINE LEARNING

DURATION:6 WEEKS

MENTOR:NEELA SANTHOSH KUMAR

The Python script codtech_task4.py implements a Movie Recommendation System. It leverages collaborative filtering techniques (User-Based and Item-Based) and Matrix Factorization (Non-negative Matrix Factorization - NMF) to provide personalized movie recommendations. The script also includes comprehensive data generation, exploration, model training, evaluation, and visualization components.

Here's a detailed breakdown of the script's functionality:

1. Setup and Library Imports:

Installs necessary packages like pandas, numpy, matplotlib, seaborn, and scikit-learn.
Includes robust error handling for scikit-learn import. If scikit-learn is not found, it attempts to install it. If installation fails, it falls back to custom, simplified implementations of train_test_split, mean_squared_error, mean_absolute_error, cosine_similarity, and a basic NMF class. This ensures the script can still run even without a fully functional scikit-learn environment, though with potentially less optimized performance for the custom implementations.
Sets a random seed for reproducibility.
2. Synthetic Data Generation:

A generate_movie_data function creates a synthetic dataset of movie ratings.
It generates n_users, n_movies, and n_ratings.
movies_df contains movie_id, title, genre, and year.
ratings_df includes user_id, movie_id, and rating, with some added genre bias to simulate real-world preferences.
Duplicate user-movie ratings are removed.
Prints dataset statistics such as the number of unique users, movies, total ratings, and data sparsity.
3. Data Exploration and Visualization:

Three histograms are generated using matplotlib and seaborn for data exploration:
Rating Distribution: Shows the frequency of different rating values (1-5).
Ratings per User: Displays the distribution of how many ratings each user has given.
Ratings per Movie: Illustrates how many ratings each movie has received.
4. Data Splitting and User-Item Matrix Creation:

The ratings_df is split into train_data and test_data using train_test_split (80% training, 20% testing).
A create_user_item_matrix function pivots the training data to create a user-item matrix, where rows represent users, columns represent movies, and values are their ratings. Missing ratings are filled with 0.
5. Recommendation System Implementations:

User-Based Collaborative Filtering (UserBasedCF):

Calculates the cosine similarity between users based on their shared movie ratings.
The predict method estimates a rating for a user-movie pair by finding the k most similar users who have rated the movie and taking a weighted average of their ratings.
The recommend method generates top-N movie recommendations for a given user by predicting ratings for unrated movies and sorting them.
Item-Based Collaborative Filtering (ItemBasedCF):

Calculates the cosine similarity between items (movies) based on how users have rated them.
The predict method estimates a rating for a user-movie pair by finding the k most similar movies that the user has already rated and taking a weighted average of those ratings.
The recommend method generates top-N movie recommendations for a given user similarly to User-Based CF, but using item similarities.
Matrix Factorization (MatrixFactorization):

Uses Non-negative Matrix Factorization (NMF) to decompose the user-item matrix into two lower-rank matrices: user features and item features.
The fit method trains the NMF model on the user-item matrix.
The predict method estimates a rating by taking the dot product of a user's feature vector and a movie's feature vector.
The recommend method generates top-N movie recommendations by calculating predicted ratings for all unrated movies for a user.
6. Training and Evaluation:

Each of the three recommendation models (UserBasedCF, ItemBasedCF, MatrixFactorization) is initialized and trained on the user_item_matrix.
An evaluate_model function calculates performance metrics (RMSE - Root Mean Squared Error and MAE - Mean Absolute Error) by comparing predicted ratings with actual ratings on a sample of the test data.
The evaluation results for all models are displayed in a table and visualized using bar plots for RMSE and MAE comparisons.
7. Recommendation Examples:

A random user ID is selected for demonstration.
The script prints the selected user's historical ratings.
It then generates and prints the top 5 movie recommendations for this sample user from each of the three implemented models, including the predicted rating and movie genre.
8. Additional Analysis:

Genre-based Analysis: Calculates and displays the average rating and count of ratings for each movie genre. These statistics are also visualized using bar plots.
Model Complexity Comparison: Provides a theoretical complexity analysis for each model, showing how their computational requirements scale with the number of users, items, and latent factors (for NMF).
9. Summary and Future Work:

A concluding summary highlights the key achievements of the script, such as implementing and evaluating different recommendation algorithms.
It suggests avenues for future experimentation, like trying different parameters or adding more sophisticated evaluation metrics.
