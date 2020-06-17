import argparse
import json
import logging
import pathlib
from typing import Optional, Union

import pandas


def is_blank(text: str) -> bool:
    """Return whether the text is blank."""
    return len(text.strip()) == 0


class DatasetReader:
    """General class for reading datasets."""

    #: The root directory.
    root: pathlib.Path

    def __init__(
        self,
        root: Union[pathlib.Path, str],
        claim_id: int = 0,
        premise_id: int = 0,
    ):
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)
        self.root = root
        self.offset_claim_id = claim_id
        self.offset_premise_id = premise_id
        self._claims = []
        self._premises = []
        self._claim_premises = []
        self._read()

    def _add_claim(self, text: str) -> Optional[int]:
        """
        Add a claim to the dataset.

        :param text:
            The claim text.
        :return:
            The claim ID.
        """
        if is_blank(text=text):
            return None

        claim_id = len(self._claims) + self.offset_claim_id
        self._claims.append(dict(
            claim_text=text,
            claim_id=claim_id,
            source=self.name,
        ))
        return claim_id

    def _add_premise(self, text: str, stance: str) -> Optional[int]:
        """
        Add a premise to the dataset.

        :param text:
            The premise text.
        :return:
            The premise ID.
        """
        if is_blank(text=text):
            return None

        premise_id = len(self._premises) + self.offset_premise_id
        self._premises.append(dict(
            premise_text=text,
            premise_id=premise_id,
            stance=stance,
            source=self.name,
        ))
        return premise_id

    def _add_claim_premise(self, premise_id: Optional[int], claim_id: Optional[int]) -> None:
        """
        Add a link between premise_id and claim_id.

        If any of the IDs is None, no link is added.
        """
        if None not in {claim_id, premise_id}:
            self._claim_premises.append(dict(
                premise_id=premise_id,
                claim_id=claim_id,
            ))

    def __str__(self):
        return f'Dataset(name={self.name}, num_claims={len(self._claims)}, num_premises={len(self._premises)})'

    @property
    def max_claim_id(self) -> int:
        return self.offset_claim_id + len(self._claims)

    @property
    def max_premise_id(self) -> int:
        return self.offset_premise_id + len(self._premises)

    @property
    def premises(self) -> pandas.DataFrame:
        return pandas.DataFrame(self._premises)

    @property
    def claims(self) -> pandas.DataFrame:
        return pandas.DataFrame(self._claims)

    @property
    def claims_premises(self) -> pandas.DataFrame:
        return pandas.DataFrame(self._claim_premises)

    @property
    def name(self) -> str:
        """The name of the dataset."""
        raise NotImplementedError

    def _read(self):
        """Read the dataset."""
        raise NotImplementedError


class DebatePediaOrgReader(DatasetReader):
    """DebatePediaOrg dataset."""

    @property
    def name(self) -> str:
        return 'DebatePediaOrg'

    def _read(self):
        for index, file_path in enumerate(self.root.iterdir()):
            with file_path.open(mode='r', errors='ignore') as json_data:
                data = json.load(json_data)
                for claim_data in data:
                    if len(claim_data['pros']) + len(claim_data['cons']) == 0:
                        logging.warning('Skipping empty file')
                        continue
                    claim_text = claim_data['claimText']
                    claim_id = self._add_claim(text=claim_text)
                    for premise_data, stance in [(claim, stance) for stance in ('Pro', 'Con') for claim in
                                                 claim_data[stance.lower() + 's']]:
                        premise_text = premise_data['premiseText']
                        premise_id = self._add_premise(text=premise_text, stance=stance)
                        self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


class DebateOrgReader(DatasetReader):
    """DebateOrg dataset."""

    @property
    def name(self) -> str:
        return 'DebateOrg'

    def _read(self):
        with self.root.open(mode='r', errors='ignore') as json_data:
            data = json.load(json_data)
            for claim_data in data:
                if len(claim_data['pros']) + len(claim_data['cons']) == 0:
                    logging.warning('Skipping empty file')
                    continue
                claim_text = claim_data['title']
                claim_id = self._add_claim(claim_text)
                for premise_data, stance in [(claim, stance) for stance in ('Pro', 'Con') for claim in
                                             claim_data[stance.lower() + 's']]:
                    premise_text = premise_data['text']
                    premise_id = self._add_premise(text=premise_text, stance=stance)
                    self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


class DebateWiseReader(DatasetReader):
    """DebateWise dataset."""

    @property
    def name(self) -> str:
        return 'debatewise'

    def _read(self):
        for index, file_path in enumerate(self.root.iterdir()):
            with file_path.open(mode='r', errors='ignore') as json_data:
                data = json.load(json_data)
                if len(data['ArgumentList']) == 0:
                    logging.warning('Skipping empty file')
                    continue
                claim_text = data['MetaData']['Title']
                claim_id = self._add_claim(text=claim_text)
                for premise_data in data['ArgumentList']:
                    premise_text = premise_data['Argument']['Premise'][0]
                    premise_id = self._add_premise(text=premise_text,
                                                   stance=premise_data['Argument']['PremiseStance'][0])
                    self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


class IDebateOrgReader(DatasetReader):
    """iDebateOrg dataset."""

    @property
    def name(self) -> str:
        return 'iDebateOrg'

    def _read(self):
        for index, file_path in enumerate(self.root.iterdir()):
            with file_path.open(mode='r', errors='ignore') as json_data:
                data = json.load(json_data)
                if len(data['pros']) + len(data['cons']) == 0:
                    logging.warning('Skipping empty file')
                    continue
                claim_text = data['title']
                claim_id = self._add_claim(claim_text)
                for premise_data_pro in (data['pros']):
                    premise_text = premise_data_pro['text point pro claim']
                    premise_id = self._add_premise(text=premise_text, stance="Pro")
                    self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)
                for premise_data_con in (data['cons']):
                    premise_text = premise_data_con['text point con claim']
                    premise_id = self._add_premise(text=premise_text, stance="Con")
                    self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


def remove_duplicates(
    premises: pandas.DataFrame,
    claims: pandas.DataFrame,
    assignments: pandas.DataFrame
) -> [pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Remove duplicate premises and claims (w.r.t. text).
    Update assignments:
    - ids that belong to a duplicate have to be updated to the remaining id.
    - then, duplicate assignments are removed

    :param premises:
        The premises.
    :param claims:
        The claims.
    :param assignments:
        The assignments.
    :return:
        The unique premises, claims and assignments.
    """
    # extend assignments to have the premise and the claim text in df
    ass_extended = pandas.merge(assignments, premises, how='inner', on="premise_id")
    ass_extended = pandas.merge(ass_extended, claims, how='inner', on="claim_id")
    # drop duplicates in claims and premises (first occurence is kept)
    claims_df = claims.drop_duplicates(subset=["claim_text"])
    premises_df = premises.drop_duplicates(subset=["premise_text"])

    # extend assignments again by the now unique claim and premise text
    ass_extended = pandas.merge(ass_extended, claims_df, how='inner', on="claim_text")
    ass_extended = pandas.merge(ass_extended, premises_df, how='inner', on="premise_text")

    # the newly added claim and premise ids are now the ids of the remaining ones
    ass_extended = ass_extended[["premise_id_y", "claim_id_y"]]
    # rename
    ass_extended = ass_extended.rename(columns={"claim_id_y": "claim_id", "premise_id_y": "premise_id"})
    # now drop all duplicate assignments
    assignments_df = ass_extended.drop_duplicates(subset=["claim_id", "premise_id"])
    return premises_df, claims_df, assignments_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input Directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output Directory")
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    c_id, p_id = 0, 0
    datasets = []
    for name, dataset_root, cls in [
        ('DebatePediaOrg', input_dir / "debatePediaOrg" / "json_files", DebatePediaOrgReader),
        ('DebateOrg', input_dir / "debateOrg" / "debates.json", DebateOrgReader),
        ('DebateWise', input_dir / "debatewise", DebateWiseReader),
        ('iDebateOrg', input_dir / "iDebateOrg" / "json_files", IDebateOrgReader)
    ]:
        logging.info(f'Read {name} from {dataset_root}.')
        reader = cls(
            root=dataset_root,
            claim_id=c_id,
            premise_id=p_id,
        )
        logging.info(f'Read dataset: {str(reader)}')
        c_id, p_id = reader.max_claim_id, reader.max_premise_id
        datasets.append(reader)

    premises_df = pandas.concat([dataset.premises for dataset in datasets])
    claims_df = pandas.concat([dataset.claims for dataset in datasets])
    assignments_df = pandas.concat([dataset.claims_premises for dataset in datasets])

    print("Claims total: ", len(claims_df))
    print("Premises total: ", len(premises_df))
    print("Assignments total: ", len(assignments_df))

    premises_df, claims_df, assignments_df = remove_duplicates(premises=premises_df, claims=claims_df,
                                                               assignments=assignments_df)

    print("Claims total without duplicates: ", len(claims_df))
    print("Premises total without duplicates: ", len(premises_df))
    print("Assignments total without duplicates: ", len(assignments_df))

    for df, name in [
        (premises_df, 'premises.csv'),
        (claims_df, 'claims.csv'),
        (assignments_df, 'assignments.csv'),
    ]:
        df.to_csv(output_dir / name)


if __name__ == '__main__':
    main()
