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


def test_evaluate():

    dsd = dsdtools.DB(root_dir="data/DSD100subset", evaluation=True)

    # process dsd but do not save the results
    result = dsd.run(
        user_function=user_function1,
        ids=55,
        evaluate=True
    )
    assert result[0][0].any()
