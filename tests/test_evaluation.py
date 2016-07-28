import pytest
import dsdtools


def user_function1(track):
    '''Pass'''

    # return any number of targets
    estimates = {
        'vocals': track.audio,
        'drums': track.audio,
    }
    return estimates


def user_function2(track):
    '''Pass'''

    # return any number of targets
    estimates = {
        'drums': track.audio,
        'vocals': track.audio,
    }
    return estimates


@pytest.mark.parametrize(
    "method",
    [
        'mir_eval',
        pytest.mark.xfail('not_a_function', raises=ValueError)
    ]
)
def test_evaluate(method):

    dsd = dsdtools.DB(root_dir="data/DSD100subset", evaluation=True)

    # process dsd but do not save the results
    assert dsd.run(
        user_function=user_function1,
        ids=55,
        evaluate=True
    )

    assert dsd.run(
        user_function=user_function2,
        ids=1,
        evaluate=True,
    )
