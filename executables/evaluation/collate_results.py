import argparse
import pathlib

import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_directory', default='./results', type=str)
    args = parser.parse_args()
    input_root = pathlib.Path(args.result_directory).expanduser().absolute()
    files = list(input_root.glob('*.csv'))
    if len(files) == 0:
        print('no input files')
        quit(-1)
    df = pandas.concat(
        objs=[pandas.read_csv(file_path) for file_path in files],
        ignore_index=True,
    )
    summary = df.groupby(by=['method', 'similarity', 'k']).agg(dict(mnDCG='mean'))
    print(summary.unstack())
