"""Utility methods."""
import pandas


def truncate(x: str, limit: int) -> str:
    """Truncate a string to at most limit words."""
    return " ".join(x.split()[:limit])


def concat_premise_claims(
    premises_df: pandas.DataFrame,
    claims_df: pandas.DataFrame,
    assignment_df: pandas.DataFrame,
    max_premise_words: int,
    max_claim_words: int,
) -> pandas.Series:
    """
    Concatenate the texts of premise-claim pairs which are assigned.

    :param premises_df: columns: {"premise_id", "premise_text"}
        The dataframe of premises.
    :param claims_df: columns: {"claim_id", "claim_text"}
        The dataframe of claims.
    :param assignment_df: columns: {"claim_id", "premise_id"}
        The dataframe with assignment.
    :param max_premise_words:
        Truncate premises to at most max_premise_words words.
    :param max_claim_words:
        Truncate claims to at most max_claim_words words.

    :return:
        A series of concatenated texts.
    """
    # subset to relevant columns
    premises_df = premises_df.loc[:, ["premise_id", "premise_text"]]
    claims_df = claims_df.loc[:, ["claim_id", "claim_text"]]
    assignment_df = assignment_df.loc[:, ["premise_id", "claim_id"]]

    # join dataframes
    extended_assignment_df = pandas.merge(
        left=pandas.merge(
            left=assignment_df,
            right=premises_df,
            how="inner",
            on="premise_id",
        ),
        right=claims_df,
        how="inner",
        on="claim_id",
    )

    # truncate premises and claims, and concatenate them
    return extended_assignment_df["premise_text"].apply(
        truncate,
        args=(max_premise_words,)
    ) + ' ||| ' + extended_assignment_df["claim_text"].apply(
        truncate,
        args=(max_claim_words,)
    )
