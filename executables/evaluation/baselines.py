import pandas

from arclus.models.baselines import RankingMethod
from arclus.utils import load_assignments_with_numeric_relevance


def main():
    algorithm = ...
    k = ...

    # load assignments
    df = load_assignments_with_numeric_relevance()
    # keep only relevant columns
    df = df.loc[:, ["claim_id", "premise_id", "relevance", "premiseClusterID_groundTruth"]]

    # Instantiate method
    method: RankingMethod

    # iterate over all claims
    result_data = []
    for claim_id, queries in df.groupby(by="claim_id"):
        # predict ranking
        predicted_ranking = method.rank(
            claim_id=claim_id,
            premise_ids=queries["premise_id"].to_numpy(dtype=str),
            k=k,
        )

        # evaluate ranking
        result_data.append((algorithm, claim_id, k, mndcg_score(predicted_ranking, queries)))
    result_df = pandas.DataFrame(data=result_data, columns=["algorithm", "claim_id", "k", "mnDCG"])
    result_df.to_csv(f"{algorithm}_results.csv")
    print(result_df['mnDCG@k'].mean())


if __name__ == '__main__':
    main()
