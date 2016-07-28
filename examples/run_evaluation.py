from __future__ import print_function
import dsdtools


# initiate dsdtools
dsd = dsdtools.DB(evaluation=True)

dsd.evaluate(estimates_dirs='./Estimates')

print(dsd.evaluator.df.df)
import ipdb; ipdb.set_trace()
