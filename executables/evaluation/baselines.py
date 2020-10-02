import pandas

from arclus.utils import load_assignments_with_numeric_relevance


def main():
    algorithm = ...
    k = ...

    # load assignments
    df = load_assignments_with_numeric_relevance()
    # keep only relevant columns
    df = df.loc[:, ["claim_id", "premise_id", "relevance", "premiseClusterID_groundTruth"]]

    # iterate over all claims
    result_data = []
    for claim_id, queries in df.groupby(by="claim_id"):
        # locate all premises which are assigned to the current claim
        premises_ids = queries["premise_id"].to_numpy(dtype=str)

        # predict ranking
        predicted_ranking = method.rank(
            claim_id,
            premise_ids,
        )

        # evaluate ranking
        result_data.append((algorithm, claim_id, k, mndcg_score(predicted_ranking, queries)))
    result_df = pandas.DataFrame(data=result_data, columns=["algorithm", "claim_id", "k", "mnDCG"])
    result_df.to_csv(f"{algorithm}_results.csv")
    print(result_df['mnDCG@k'].mean())


if __name__ == '__main__':
    main()
