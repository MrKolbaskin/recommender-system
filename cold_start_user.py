import json
import pickle
import pathlib
from typing import List

import numpy as np
import pandas as pd
from termcolor import colored
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

PATH_NEW_USERS = pathlib.Path("static/svd_embedings/new_users_svd.pkl").resolve()
PATH_USER_IDS = pathlib.Path("static/data/users_id.csv").resolve()
COUNT_NEIGHBOURS = 15
GENRES_LIST = [
    "Action",
    "Drama",
    "Horror",
    "Comedy",
    "Romance",
    "Adventure",
    "Animation",
    "Fantasy",
    "Thriller",
]

users = np.load("static/svd_embedings/users_svd.pkl", allow_pickle=True)

count_users = users.shape[0]

if PATH_NEW_USERS.exists():
    main_new_users_svd = np.load(PATH_NEW_USERS, allow_pickle=True)
    count_users += main_new_users_svd.shape[0]
else:
    main_new_users_svd = None


def get_user_ids(count: int, count_users: int):
    result = []

    for tmp_id in range(count_users + 1, count_users + 1 + count):
        result.append(tmp_id)

    return result


def from_list_2_dict(list_ratings: List[int]):
    res = {}
    for genre, rating in zip(GENRES_LIST, list_ratings):
        res[genre] = rating

    return res


def to_df(new_users: List[List[int]], new_user_ids: List[int]):
    list_new_users_with_id = []
    for ind, user in enumerate(new_users):
        tmp = from_list_2_dict(user)
        tmp["user_id"] = new_user_ids[ind]
        list_new_users_with_id.append(tmp)

    df_with_new_users = pd.DataFrame(list_new_users_with_id)
    if PATH_USER_IDS.exists():
        main_new_users = pd.read_csv(PATH_USER_IDS)
        main_new_users = main_new_users.append(df_with_new_users, ignore_index=True)
        main_new_users.to_csv(PATH_USER_IDS, index=False)
    else:
        df_with_new_users.to_csv(PATH_USER_IDS, index=False)


if __name__ == "__main__":
    try:
        new_users = json.load(open("example/new_users.json", "r"))
    except BaseException:
        print(colored("Неправильный формат данных", "red"))

    new_user_ids = get_user_ids(len(new_users), count_users)

    new_users_matrix = normalize(np.array(new_users), axis=1)
    users_matrix = normalize(np.load("static/embedings/user_emb.npy"), axis=1)

    print(colored("Предсказание векторов SVD", "yellow"))
    neighbors = cdist(new_users_matrix, users_matrix, metric="cosine").argsort(axis=1)[
        :, :COUNT_NEIGHBOURS
    ]

    svd_new_users = np.zeros((neighbors.shape[0], 100))

    for ind in range(neighbors.shape[0]):
        svd_new_users[ind] = users[neighbors[ind]].mean(axis=0)

    print(colored("Сохранение данных", "yellow"))
    if main_new_users_svd is not None:
        with open(PATH_NEW_USERS, "wb") as f:
            pickle.dump(np.vstack([main_new_users_svd, svd_new_users]), f)
    else:
        with open(PATH_NEW_USERS, "wb") as f:
            pickle.dump(svd_new_users, f)

    to_df(new_users, new_user_ids)

    print(colored("Новые пользователи добавились!", "green"))
