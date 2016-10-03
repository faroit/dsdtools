from __future__ import print_function
import dsdtools
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Source Separation based on Common Fate Model')

    parser.add_argument(
        'mat_file',
        type=str,
        help='Input matfile'
    )

    parser.add_argument(
        'pandas_file',
        type=str,
        help='Output as pandas pickle'
    )

    args = parser.parse_args()

    # initiate dsdtools
    dsd = dsdtools.DB(evaluation=True)

    dsd.evaluator.data.import_mat(args.mat_file)
    df = dsd.evaluator.data.df
    dsd.evaluator.data.to_pickle(args.pandas_file)
