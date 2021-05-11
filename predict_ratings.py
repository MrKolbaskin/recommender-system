from surprise import dump
import pandas as pd
from termcolor import colored
from tqdm import tqdm


def prediction_to_dict(prediction):
    return {
        "UserId": prediction.uid,
        "MovieId": prediction.iid,
        "True rating": prediction.r_ui,
        "Prediction rating": prediction.est,
    }


def predictions_to_df(predictions):
    return pd.DataFrame(
        [prediction_to_dict(prediction) for prediction in tqdm(predictions)]
    )


if __name__ == "__main__":
    try:
        print("LOAD SVD MODEL")
        _, svd_model = dump.load("static/models/svd.model")
        df_ratings = pd.read_csv("static/data/ratings.csv")

        test_ratings = list(
            zip(
                df_ratings["userId"].to_list(),
                df_ratings["movieId"].to_list(),
                df_ratings["rating"].to_list(),
            )
        )

        print("PREDICT")
        predictions = svd_model.test(test_ratings)

        print("TRANSLATE")
        df_predictions = predictions_to_df(predictions)

        df_predictions.to_csv("predictions.csv")
    except BaseException:
        print(colored("Что-то пошло не так!", "red"))
        exit()

    print(colored("Предсказания находятся в файле predictions.csv", "green"))
