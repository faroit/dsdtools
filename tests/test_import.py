import pytest
import dsdtools
import dsdtools.evaluate as ev
import numpy as np


@pytest.fixture(params=['data/DSD100subset'])
def dsd(request):
    return dsdtools.DB(root_dir=request.param, evaluation=True)


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
    # test the numerical fields
    mat_sub = mat_data.df.select_dtypes(include=['float64'])
    dsd_sub = dsd.evaluator.data.df.select_dtypes(include=['float64'])
    num_comp = (np.all(np.isclose(mat_sub.as_matrix(), dsd_sub.as_matrix())))
    # it will likely fail due to different ordering of rows
    if not num_comp:
        # try the misorder row test
        test_index = [mat_sub.index[2],
                      mat_sub.index[3],
                      mat_sub.index[0],
                      mat_sub.index[1]]
        mat_data.df = mat_data.df.reindex(test_index)
        mat_sub = mat_data.df.select_dtypes(include=['float64'])
        num_comp = (np.all(np.isclose(mat_sub.as_matrix(),
                                      dsd_sub.as_matrix())))
    assert num_comp
    # test the non-number fields
    mat_sub = mat_data.df.select_dtypes(exclude=['float64'])
    dsd_sub = dsd.evaluator.data.df.select_dtypes(exclude=['float64'])
    mat_sub = mat_sub.reset_index()
    del mat_sub['index']
    df_comp = mat_sub.equals(dsd_sub)
    assert df_comp


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
    # test the numerical fields
    json_sub = json_data.df.select_dtypes(include=['float64'])
    dsd_sub = dsd.evaluator.data.df.select_dtypes(include=['float64'])
    num_comp = (np.all(np.isclose(json_sub.as_matrix(), dsd_sub.as_matrix())))
    # it may fail due to different ordering of rows
    if not num_comp:
        # try the misorder row test
        test_index = [json_sub.index[2],
                      json_sub.index[3],
                      json_sub.index[0],
                      json_sub.index[1]]
        json_data.df = json_data.df.reindex(test_index)
        json_sub = json_data.df.select_dtypes(include=['float64'])
        num_comp = (np.all(np.isclose(json_sub.as_matrix(),
                                      dsd_sub.as_matrix())))
    assert num_comp
    # test the non-number fields
    json_sub = json_data.df.select_dtypes(exclude=['float64'])
    dsd_sub = dsd.evaluator.data.df.select_dtypes(exclude=['float64'])
    df_comp = json_sub.equals(dsd_sub)
    if not df_comp:
        json_sub = json_sub.reset_index()
        del json_sub['index']
    assert json_sub.equals(dsd_sub)
