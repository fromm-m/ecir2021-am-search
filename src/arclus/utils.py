"""Utility methods."""
import random
import string
from typing import Callable, Collection, List, Optional, Set, Type, TypeVar, Union

import numpy as np
import pandas
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


def format_numbers(numbers_string):
    return float(numbers_string.replace('.', ',', 1).replace('.', '').replace(',', '.'))


def load_assignments_with_numeric_relevance(csv_path: str = PREP_ASSIGNMENTS_TEST) -> pandas.DataFrame:
    """Load the dataframe with all informations."""
    # read data
    df = pandas.read_csv(csv_path, sep=";", thousands='.')

    # set the relevance to the according value (cf. paper)
    translation = {
        "notRelevant": 0,
        "yesRelevant": 1,
        "yesVeryRelevant": 2,
    }
    df["relevance"] = df["relevance"].map(translation.__getitem__)

    return df


T = TypeVar('T')


def get_all_subclasses(base_class: Type[T]) -> Set[Type[T]]:
    """Get a collection of all (recursive) subclasses of a given base class."""
    return set(base_class.__subclasses__()).union(s for c in base_class.__subclasses__() for s in get_all_subclasses(c))


def identity(x: T) -> T:
    return x


def get_subclass_by_name(
    base_class: Type[T],
    name: str,
    normalizer: Optional[Callable[[str], str]] = None,
    exclude: Optional[Union[Collection[Type[T]], Type[T]]] = None,
) -> Type[T]:
    """Get a subclass of a base-class by name.

    :param base_class:
        The base class.
    :param name:
        The name.
    :param normalizer:
        An optional name normalizer, e.g. str.lower
    :param exclude:
        An optional collection of subclasses to exclude.

    :return:
        The subclass with matching name.
    :raises ValueError:
        If no such subclass can be determined.
    """
    if normalizer is None:
        normalizer = identity
    if exclude is None:
        exclude = set()
    if isinstance(exclude, type):
        exclude = {exclude}
    norm_name = normalizer(name)
    for subclass in get_all_subclasses(base_class=base_class).difference(exclude):
        if normalizer(subclass.__name__) == norm_name:
            return subclass
    subclass_dict = {normalizer(c.__name__): c for c in get_all_subclasses(base_class=base_class)}
    raise ValueError(f'{base_class} does not have a subclass named {norm_name}. Subclasses: {subclass_dict}.')


def generate_random_words(num_words, max_word_length: int = 10) -> List[str]:
    """Generate a list of random words."""
    return [
        "".join(random.choices(string.ascii_letters, k=random.randrange(1, max_word_length)))
        for _ in range(num_words)
    ]


def argparse_bool(x):
    """Convert a command line arguments for a boolean value."""
    return str(x).lower() in {'true', '1', 'yes'}


def resolve_num_clusters(ratio: Optional[float], num_premises: int, k: int) -> int:
    if ratio is None:
        n_clusters = k
    else:
        n_clusters = int(round(ratio * num_premises))
        n_clusters = max(n_clusters, k)
    n_clusters = min(n_clusters, num_premises)
    return n_clusters
