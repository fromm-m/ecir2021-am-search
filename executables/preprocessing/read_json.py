import argparse
import logging
import pathlib

import pandas
from arclus.preprocessing.dataset_reader import DebateWiseReader, DebateOrgReader, DebatePediaOrgReader, IDebateOrgReader, remove_duplicates


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
