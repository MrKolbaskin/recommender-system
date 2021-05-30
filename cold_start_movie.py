import numpy as np
import pickle
import pandas as pd
from tensorflow import keras
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from langdetect import detect
from termcolor import colored
import pathlib

PATH_NEW_MOVIES = pathlib.Path("static/svd_embedings/new_movies_svd.pkl").resolve()
PATH_NEW_MOVIES_DF = pathlib.Path("static/data/new_movies.csv").resolve()
PATH_RUBERT_DP = pathlib.Path("static/models/model_rubert_dp.pkl").resolve()
PATH_MULTILINGUAL_DP = pathlib.Path("static/models/model_multilingual_dp.pkl").resolve()
PATH_RUBERT_2_SVD = pathlib.Path("static/models/model_rubert_nn.h5").resolve()
PATH_MULTILINGUAL_2_SVD = pathlib.Path(
    "static/models/model_multilingualbert_nn.h5"
).resolve()
NEW_MOVIES_PATH = "example/new_movies.csv"


movies = np.load("static/svd_embedings/movies_svd.pkl", allow_pickle=True)
count_movies = movies.shape[0]

if PATH_NEW_MOVIES.exists():
    main_new_movies_svd = np.load(PATH_NEW_MOVIES, allow_pickle=True)
    count_movies += main_new_movies_svd.shape[0]
else:
    main_new_movies_svd = None


rubert = pickle.load(open(PATH_RUBERT_DP, "rb"))
multilingual_bert = pickle.load(open(PATH_MULTILINGUAL_DP, "rb"))

rubert_2_svd = keras.models.load_model(PATH_RUBERT_2_SVD)
multilingual_2_svd = keras.models.load_model(PATH_MULTILINGUAL_2_SVD)


def div_to_sent(texts):
    res = []
    for text in texts:
        res.append(sent_tokenize(text))

    return res


def get_movie_ids(count: int, count_movies: int):
    result = []

    for tmp_id in range(count_movies + 1, count_movies + 1 + count):
        result.append(tmp_id)

    return result


def _add_svd_emb(svd_embs, embs):
    for ind, elem in enumerate(embs):
        elem["svd_emb"] = svd_embs[ind]


if __name__ == "__main__":
    new_movies_df = pd.read_csv(NEW_MOVIES_PATH)
    new_movie_ids = get_movie_ids(new_movies_df.shape[0], count_movies)
    new_movies_data = list(new_movies_df.title + ". " + new_movies_df.plot)

    for ind, elem in enumerate(new_movies_df.description):
        if elem:
            new_movies_data[ind] += " " + elem

    sent_data = div_to_sent(new_movies_data)

    print(colored("Построение векторных представлений сюжетов", "yellow"))
    rubert_embedings = []
    multilingual_embedings = []
    for ind, plot in tqdm(enumerate(sent_data)):
        if detect(plot) != "ru":
            _, _, _, _, _, _, bert_pooler_outputs = multilingual_bert(plot)
            multilingual_embedings.append(
                {"embeding": bert_pooler_outputs.mean(axis=0), "index": ind}
            )
        else:
            _, _, _, _, _, _, bert_pooler_outputs = rubert(plot)
            rubert_embedings.append(
                {"embeding": bert_pooler_outputs.mean(axis=0), "index": ind}
            )

    multilingual_emb_matrix = [elem["embeding"] for elem in multilingual_embedings]
    rubert_emb_matrix = [elem["embeding"] for elem in rubert_embedings]

    print(colored("Предсказание векторов SVD", "yellow"))
    multilingual_svd = multilingual_2_svd.predict(multilingual_emb_matrix)
    rubert_svd = rubert_2_svd.predict(rubert_emb_matrix)

    _add_svd_emb(multilingual_svd, multilingual_embedings)
    _add_svd_emb(rubert_svd, rubert_embedings)

    movie_embs = sorted(
        multilingual_embedings + rubert_embedings, key=lambda x: x["index"]
    )
    svd_new_movies = np.array([elem["svd_emb"] for elem in movie_embs])

    print(colored("Сохранение данных", "yellow"))
    if main_new_movies_svd is not None:
        with open(PATH_NEW_MOVIES, "wb") as f:
            pickle.dump(np.vstack([main_new_movies_svd, svd_new_movies]), f)
    else:
        with open(PATH_NEW_MOVIES, "wb") as f:
            pickle.dump(svd_new_movies, f)

    index_new_movies = [elem["index"] for elem in movie_embs]
    new_movies_df["movie_id"] = index_new_movies

    if PATH_NEW_MOVIES_DF.exists():
        main_new_users = pd.read_csv(PATH_NEW_MOVIES_DF)
        main_new_users = main_new_users.append(new_movies_df, ignore_index=True)
        main_new_users.to_csv(PATH_NEW_MOVIES_DF, index=False)
    else:
        new_movies_df.to_csv(PATH_NEW_MOVIES_DF, index=False)

    print(colored("Новые фильмы добавились!", "green"))
