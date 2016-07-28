from __future__ import print_function
import dsdtools
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Source Separation based on Common Fate Model')

    parser.add_argument(
        'estimate_dirs',
        type=str,
        nargs='+',
        help='Estimate folders'
    )

    args = parser.parse_args()

    # initiate dsdtools
    dsd = dsdtools.DB(evaluation=True)

    dsd.evaluate(estimates_dirs=args.estimate_dirs, parallel=True)
    dsd.evaluator.data.to_pickle("results.pandas")
