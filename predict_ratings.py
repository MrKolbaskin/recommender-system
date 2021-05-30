import pandas as pd
from termcolor import colored
from tqdm import tqdm
import numpy as np
from collections import namedtuple
import pathlib


USERS_FILE = "static/svd_embedings/users_svd.pkl"
MOVIES_FILE = "static/svd_embedings/movies_svd.pkl"
NEW_USERS_FILE = pathlib.Path("static/svd_embedings/new_users_svd.pkl").resolve()
NEW_MOVIES_FILE = pathlib.Path("static/svd_embedings/new_movies_svd.pkl").resolve()
df_ratings = pd.read_csv("static/data/ratings_predict.csv")


Prediction = namedtuple("Prediction", ["uid", "iid", "est"])


class SVDModel:
    def __init__(self, users, movies):
        self.users = users
        self.movies = movies

    def predict(self, user_id: int, movie_id: int):
        return Prediction(
            user_id, movie_id, self.users[user_id] @ self.movies[movie_id]
        )


def prediction_to_dict(prediction):
    return {
        "Номер пользователя": prediction.uid,
        "Номер фильма": prediction.iid,
        "Предсказанный рейтинг": prediction.est,
    }


def predictions_to_df(predictions):
    return pd.DataFrame(
        [prediction_to_dict(prediction) for prediction in tqdm(predictions)]
    )


if __name__ == "__main__":
    try:
        print("LOAD SVD MODEL")
        users = np.load(USERS_FILE, allow_pickle=True)
        if NEW_USERS_FILE.exists():
            new_users = np.load(NEW_USERS_FILE, allow_pickle=True)
            users = np.vstack([users, new_users])
        movies = np.load(MOVIES_FILE, allow_pickle=True)
        if NEW_MOVIES_FILE.exists():
            new_movies = np.load(NEW_MOVIES_FILE, allow_pickle=True)
            movies = np.vstack([movies, new_movies])
        svd_model = SVDModel(users, movies)

        print("PREDICT")
        predictions = []
        for user_id in tqdm(range(len(users))):
            for movie_id in range(len(movies)):
                predictions.append(svd_model.predict(user_id, movie_id))

        print("TRANSLATE")
        df_predictions = predictions_to_df(predictions)

        df_predictions.to_csv("predictions.csv")
    except BaseException:
        print(colored("Что-то пошло не так!", "red"))
        exit()

    print(colored("Предсказания находятся в файле ratings_predict.csv", "green"))
