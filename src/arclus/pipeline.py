from typing import Collection, Union

import pandas
from tqdm import tqdm

from .evaluation import mndcg_score
from .models import RankingMethod
from .utils import load_assignments_with_numeric_relevance


def evaluate_ranking_method(
    method: RankingMethod,
    k: Union[int, Collection[int]],
) -> pandas.DataFrame:
    """
    Evaluate a ranking method with mnDCG@k.

    :param method:
        The ranking method.
    :param k: > 0
        (Potentially multiple) cut-off parameter(s).

    :return:
        A dataframe with columns ["claim_id", "k", "mnDCG"].
    """
    # Input normalization
    if isinstance(k, int):
        k = [k]
    # load assignments
    df = load_assignments_with_numeric_relevance()
    # keep only relevant columns
    df = df.loc[:, ["claim_id", "premise_id", "relevance", "premiseClusterID_groundTruth"]]
    # iterate over all claims
    result_data = []
    for query_claim_id, queries in tqdm(df.groupby(by="claim_id"), unit='claim'):
        # use other data for chosing parameters (=leave-one-out cross validation)
        training_data = df[df["claim_id"] != query_claim_id]
        for kk in k:
            # optimize parameters
            method.fit(
                training_data=training_data,
                k=kk,
            )
            # predict ranking
            predicted_ranking = method.rank(
                claim_id=query_claim_id,
                premise_ids=queries["premise_id"].to_numpy(dtype=str),
                k=kk,
            )
            # evaluate ranking
            score = mndcg_score(y_pred=predicted_ranking, data=queries, k=kk)
            result_data.append((query_claim_id, kk, score))
    return pandas.DataFrame(data=result_data, columns=["claim_id", "k", "mnDCG"])
