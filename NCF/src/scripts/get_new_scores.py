import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf

from helper import get_model
from helper import get_scores as get_scores_per_user
from retrain import counterfactual2path


def get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
    """
    get scores of all items after retrained
    Args:
        idx: test number
        user_id: ID of user
        item_id: ID of item
        topk: the top-k items
        counterfactual: the counterfactual set
        predicted_scores: the predicted scores
        replacement: the replacement item
        item2scores: a dict for caching
        home_dir: the directory where trained models are stored

    Returns:
        a 2d array where each row is the scores of all items in one retrain.
    """
    key = counterfactual2path(user_id, counterfactual)
    if key in item2scores:
        return item2scores[key]
    if not Path(f'{home_dir}/{key}/').exists():
        print('missing', user_id, key)
        print(f'{home_dir}/{key}/')
        return None
    subfolders = sorted([f.path for f in os.scandir(f'{home_dir}/{key}/') if f.is_dir()])
    if len(subfolders) != 5:
        print('missing', user_id, key, len(subfolders))
        return None

    new_scores = []

    for i in range(5):
        tf.reset_default_graph()
        model = get_model(use_recs=True)
        path = f'{home_dir}/{counterfactual2path(user_id, counterfactual)}/{i}/'
        model.saver.restore(model.sess, path + 'model')
        print('begin scoring', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i, path)
        scores, _ = get_scores_per_user(user_id, 1, model)
        new_scores.append(scores)

    item2scores[key] = new_scores
    return new_scores


def get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
    """
        get the new scores of top-k items
        Args:
            idx: test number
            user_id: ID of user
            item_id: ID of item
            topk: the top-k items
            counterfactual: the counterfactual set
            predicted_scores: the predicted scores
            replacement: the replacement item
            item2scores: a dict for caching
            home_dir: the home directory, where trained models are stored

        Returns: a 2d array where each row is the scores of top-k items in one retrain.
    """
    scores = get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores,
                        home_dir)
    if scores is None:
        return None

    res = np.zeros((5, len(topk)))
    for i in range(5):
        res[i] = [scores[i][item] for item in topk]

    return res


def get_new_scores(algo, ks):
    """
        get new scores after retrained for the given values of k
        Args:
            algo: algorithm used to generate explanations
            ks: values of k to consider
    """
    input_files = [f"{algo}_{k}.csv" for k in ks]

    home_dir = 'D:/pretrain-ncf-test'

    item2scores = dict()

    for file in input_files:
        print('begin file', file)
        inputs = pd.read_csv(file)
        for row in inputs.itertuples():
            idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:7]
            if not isinstance(counterfactual, str):
                print('skip', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
                continue
            topk = literal_eval(topk)
            counterfactual = literal_eval(counterfactual)
            if isinstance(predicted_scores, str):
                predicted_scores = literal_eval(predicted_scores)
            else:
                predicted_scores = None
            print('begin idx', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)

            scores = get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement,
                                     item2scores, home_dir)
            if scores is None:
                print('bad scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
                continue
            assert len(scores) == 5

            for i in range(5):
                inputs.at[idx, f'actual_scores_{i}'] = str(list(scores[i]))
            s = np.mean(scores, axis=0)
            inputs.at[idx, f'actual_scores_avg'] = str(list(s))
            print('avg new scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, s)

        inputs.to_csv(file, index=False)


if __name__ == "__main__":
    get_new_scores(algo='accent', ks=[5, 10, 20])
