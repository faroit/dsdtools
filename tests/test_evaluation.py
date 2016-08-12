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


def user_function3(track):
    '''Pass'''

    # return incorrect shaped targets
    estimates = {
        'vocals': track.audio[2:],
        'accompaniment': track.audio[2:],
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


def test_verbosity():

    dsd = dsdtools.DB(root_dir="data/DSD100subset", evaluation=True)

    track = dsd.load_dsd_tracks(ids=55)[0]

    # test with verbose set to True (for coveralls)
    results = dsd.evaluator.evaluate_track(track,
                                           user_function1(track),
                                           verbose=True)
    assert results


def test_eval_failure():

    dsd = dsdtools.DB(root_dir="data/DSD100subset", evaluation=True)

    track = dsd.load_dsd_tracks(ids=55)[0]

    # test with a failing evaluation
    results = dsd.evaluator.evaluate_track(track,
                                           user_function3(track))
    assert len(results) == 0
