import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_user_similarity(user_item_matrix):
    """
    Calcola la similarità coseno tra gli utenti.

    Args:
        user_item_matrix (pd.DataFrame): Matrice user-item.

    Returns:
        pd.DataFrame: Matrice di similarità tra gli utenti.
    """
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return similarity_df

def recommend_movies_for_user(user_id, user_similarity_df, user_item_matrix, num_recommendations=5):
    """
    Genera raccomandazioni per un utente specifico basate sulla similarità con altri utenti.

    Args:
        user_id (int): ID dell'utente per cui fare raccomandazioni.
        user_similarity_df (pd.DataFrame): Matrice di similarità tra utenti.
        user_item_matrix (pd.DataFrame): Matrice user-item.
        num_recommendations (int): Numero di raccomandazioni da generare.

    Returns:
        list: Lista di tuple (movie_id, predicted_score) ordinate per punteggio.
    """
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]  # Escludi l'utente stesso
    recommended_movies = {}

    # Itera sugli utenti simili
    for similar_user in similar_users:
        user_ratings = user_item_matrix.loc[similar_user]
        for movie_id, rating in user_ratings.items():
            # Aggiungi il film alla lista se non è stato già visto dall'utente target
            if user_item_matrix.loc[user_id, movie_id] == 0 and rating > 3:
                recommended_movies[movie_id] = recommended_movies.get(movie_id, 0) + rating

    # Ordina i film per punteggio
    sorted_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    return sorted_movies[:num_recommendations]