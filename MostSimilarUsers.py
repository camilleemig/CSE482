from operator import itemgetter
from math import sqrt
from statistics import mean
from MovieRatingData import MovieRatingData
from MostSimilarMovies import KnnRecommender

def find_cosine_similarity(ratings1, ratings2):
    d1 = 0
    d2 = 0
    dot = 0
    for movie in ratings1:
        rating1 = ratings1[movie]
        rating2 = ratings2[movie]
        dot += rating1*rating2
        d1 += rating1*rating1
        d2 += rating2*rating2
    return dot / (sqrt(d1)*sqrt(d2))


def find_predicted_ratings_for_data(user_1_data, similar_movies):
    """
    This method finds movies that other people who like the same stuff as you also like-
    doesn't take into account if you have any genres/movies that are related to that movie
    :param user_1_data:
    :return:
    """
    data = MovieRatingData()

    sorted_users = sorted(data.users_to_ratings.keys())
    test_users_movies = set(user_1_data.keys())
    test_users_movies_with_ratings = user_1_data
    # shared movies is stored as user_id -> set of shared movies
    shared_movies = {}
    for user in sorted_users:
        if len(test_users_movies) <= 30:
            min_movies = len(test_users_movies)
        else:
            min_movies = 1
        movies = test_users_movies & set(data.users_to_ratings[user].keys())
        if movies and len(movies) >= min_movies:
            shared_movies[user] = movies

    # similarities is stored as user_id -> user_similarity
    similarities = {}
    for user, movies in shared_movies.items():
        user1_movies = dict([(k, test_users_movies_with_ratings[k]) for k in movies])
        user2_movies = dict([(k, data.users_to_ratings[user][k]) for k in movies])
        similarity = find_cosine_similarity(user1_movies, user2_movies)
        similarities[user] = similarity

    # user averages is user_id -> average of all their movie_ratings
    user_averages = {}
    for user in sorted_users:
        ratings = list(data.users_to_ratings[user].values())
        average = sum(ratings)/len(ratings)
        user_averages[user] = average

    # predicted movie_ratings is movie name -> predicted rating
    predicted_ratings = {}
    test_ratings = list(test_users_movies_with_ratings.values())
    test_user_average = sum(test_ratings) / len(test_ratings)
    possible_movies = data.all_movies
    for movie in possible_movies:
        total_sim = 0
        total_sim_rated = 0
        for user in similarities:
            if movie not in data.users_to_ratings[user]:
                continue
            similarity = similarities[user]
            total_sim += similarity
            total_sim_rated += similarity*(data.users_to_ratings[user][movie] - user_averages[user])
        if total_sim == 0 or total_sim_rated == 0:
            continue
        predicted = total_sim_rated/total_sim
        predicted_ratings[movie] = test_user_average + predicted
    return sorted(predicted_ratings.items(), key=itemgetter(1), reverse=True)


data = MovieRatingData()
test_users = list(data.TEST_DATA.keys())
# average_off = []
# for user in test_users:
#     actual_data = data.TEST_DATA[test_users[0]]
#     predicted_data = dict(find_predicted_ratings_for_data(actual_data))
#     for movie, rating in actual_data.items():
#         predicted_rating = predicted_data[movie]
#         average_off.append(abs(rating-predicted_rating))
# print(mean(average_off))
rec = KnnRecommender()
similar_movies = {}
for movie in data.all_movies:
    similar_movies[movie] = set(rec.make_recommendations(movie))
average_num_recommended_and_rated = []
for user in test_users:
    actual_data = data.TEST_DATA[test_users[0]]
    predicted_data = dict(find_predicted_ratings_for_data(actual_data, similar_movies)).keys()
    average_num_recommended_and_rated.append(len(set(predicted_data) & set(actual_data.keys())))
print(mean(average_num_recommended_and_rated))