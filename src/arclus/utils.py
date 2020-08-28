"""Utility methods."""
import random

import numpy as np
import pandas
import pandas as pd
import torch

from arclus.settings import PREP_ASSIGNMENTS_TEST


def is_blank(text: str) -> bool:
    """Return whether the text is blank."""
    return len(text.strip()) == 0


def set_random_seed(seed: int):
    """Set the random seed on numpy, torch, and python."""
    return (
        np.random.seed(seed=seed),
        torch.manual_seed(seed=seed),
        random.seed(seed),
    )


def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]
    return dict(items)


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


def load_assignments_with_numeric_relevance():
    # set the relevance to the according value (cf. paper)
    df_assignments = pd.read_csv(PREP_ASSIGNMENTS_TEST, sep=";")
    df_assignments['relevance'].loc[(df_assignments['relevance'] == "notRelevant")] = 0
    df_assignments['relevance'].loc[(df_assignments['relevance'] == "yesRelevant")] = 1
    df_assignments['relevance'].loc[(df_assignments['relevance'] == "yesVeryRelevant")] = 2
    return df_assignments
