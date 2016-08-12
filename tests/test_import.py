import pytest
import dsdtools
import dsdtools.evaluate as ev
import numpy as np


@pytest.fixture(params=['data/DSD100subset'])
def dsd(request):
    return dsdtools.DB(root_dir=request.param, evaluation=True)


def _compare_dataframes(df_ref, df_imp):
    # test the numerical fields
    imp_sub = df_imp.select_dtypes(include=['float64'])
    ref_sub = df_ref.select_dtypes(include=['float64'])
    num_comp = (np.all(np.isclose(imp_sub.as_matrix(), ref_sub.as_matrix())))
    # it may fail due to different ordering of rows
    if not num_comp:
        # try the misorder row test
        test_index = [imp_sub.index[2],
                      imp_sub.index[3],
                      imp_sub.index[0],
                      imp_sub.index[1]]
        imp_sub = imp_sub.reindex(test_index)
        imp_sub = imp_sub.select_dtypes(include=['float64'])
        num_comp = (np.all(np.isclose(imp_sub.as_matrix(),
                                      ref_sub.as_matrix())))
    assert num_comp
    # test the non-number fields
    imp_sub = imp_sub.select_dtypes(exclude=['float64'])
    ref_sub = ref_sub.select_dtypes(exclude=['float64'])
    df_comp = imp_sub.equals(ref_sub)
    # it may fail due to non-equal indexes
    if not df_comp:
        imp_sub = imp_sub.reset_index()
        del imp_sub['index']
        df_comp = imp_sub.equals(ref_sub)
    assert df_comp


def test_mat_import(dsd):
    # run the baseline evaluation
    dsd.evaluate(
        estimates_dirs='tests/data/'
    )
    # load the mat file
    mat_data = ev.Data(dsd.evaluator.data.columns)
    mat_data.import_mat('tests/data/results.mat')
    # compare the methods
    # make sure they have the same columns
    assert set(mat_data.df.columns) == set(dsd.evaluator.data.df.columns)
    # estimate_dir and sample will be different; remove them from both
    del mat_data.df['estimate_dir']
    del mat_data.df['sample']
    del dsd.evaluator.data.df['estimate_dir']
    del dsd.evaluator.data.df['sample']
    # reorder the columns so they are the same
    mat_data.df = mat_data.df[dsd.evaluator.data.df.columns]
    # prune the rows that won't match
    mat_data.df = mat_data.df[np.invert(np.isclose(mat_data.df['SDR'], 0.0))]
    mat_data.df = mat_data.df[mat_data.df['subset'] == 'Test']
    # compare the dataframes
    _compare_dataframes(dsd.evaluator.data.df, mat_data.df)


def test_json_import(dsd):
    # run the baseline evaluation
    dsd.evaluate(
        estimates_dirs='tests/data/'
    )
    # load the json file
    json_data = ev.Data(dsd.evaluator.data.columns)
    json_data.import_json('tests/data/evaluation.json')
    # compare the methods
    # make sure they have the same columns
    assert set(json_data.df.columns) == set(dsd.evaluator.data.df.columns)
    # estimate_dir will be different; remove it from both
    del json_data.df['estimate_dir']
    del dsd.evaluator.data.df['estimate_dir']
    # reorder the columns so they are the same
    json_data.df = json_data.df[dsd.evaluator.data.df.columns]
    # compare the dataframes
    _compare_dataframes(dsd.evaluator.data.df, json_data.df)
