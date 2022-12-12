import pandas as pd

from helper import get_model, get_scores
import tensorflow._api.v2.compat.v1 as tf


def get_rec():
    """
    get recommendations for each user
    """
    tf.reset_default_graph()
    model = get_model()
    k = 20
    users = []
    recs = []
    rec_scores = []
    for user in range(model.num_users):
        scores, topk = get_scores(user, k, model)
        users.extend([user]*k)
        recs.extend(topk)
        rec_scores.extend([scores[item] for item in topk])

    rec_df = pd.DataFrame({
        'user': users,
        'rec': recs,
        'score': rec_scores
    })

    rec_df.to_csv('recs.csv', index=False)


if __name__ == "__main__":
    get_rec()
