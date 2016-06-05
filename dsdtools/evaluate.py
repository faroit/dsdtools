from __future__ import print_function
import numpy as np
import mir_eval
import utils


class BSSeval(object):
    def __init__(self, method, collect=True):
        methods = ("mir_eval")
        if method not in methods:
            raise ValueError("method must be in %s" % ','.join(methods))

        self.method = method
        self.df = utils.DF_writer([
            'track_id',
            'track_name',
            'target_name',
            'estimate_name',
            'SDR',
            'ISR',
            'SIR',
            'SAR',
        ])

    def evaluate_track(self, track, user_estimates, estimates_dir=None):
        audio_estimates = []
        audio_reference = []

        # make sure to always build the list in the same order
        # therefore track.targets is an OrderedDict
        labels_references = []  # save the list of targets to be evaluated
        for key, target in list(track.targets.items()):
            try:
                # try to fetch the audio from the user_results of a given key
                estimate = user_estimates[key]
                # add the audio to the list of estimates
                audio_estimates.append(estimate)
                # add the audio to the list of references
                audio_reference.append(track.targets[key].audio)
                # append this target name to the list of labels
                labels_references.append(target)
            except KeyError:
                # ignore wrong key and continue
                continue

        if audio_estimates and audio_reference:
            audio_estimates = np.array(audio_estimates)
            audio_reference = np.array(audio_reference)
            if audio_estimates.shape == audio_reference.shape:
                SDR, ISR, SIR, SAR = self.evaluate(
                    audio_estimates, audio_reference
                )

                for i, target in enumerate(labels_references):
                    print(target.name)
                    print(SDR[i])

                    self.df.append(
                        track_id=0,
                        track_name=track.name,
                        target_name=target.name,
                        estimate_dir=estimates_dir,
                        SDR=SDR[i],
                        ISR=ISR[i],
                        SIR=SIR[i],
                        SAR=SAR[i],
                    )

    def evaluate(self, estimates, originals, verbose=True):
        """Universal BSS evaluate frontend for several evaluators

        Parameters
        ----------
        originals : np.ndarray, shape=(nsrc, nsampl, nchan)
            array containing true reference sources
        estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
            array containing estimated sources

        Returns
        -------
        SDR : np.ndarray, shape=(nsrc,)
            vector of Signal to Distortion Ratios (SDR)
        ISR : np.ndarray, shape=(nsrc,)
            vector of Source to Spatial Distortion Image (ISR)
        SIR : np.ndarray, shape=(nsrc,)
            vector of Source to Interference Ratios (SIR)
        SAR : np.ndarray, shape=(nsrc,)
            vector of Sources to Artifacts Ratios (SAR)
        """
        print("Evaluating with %s" % self.method)

        if self.method == "mir_eval":
            mono_estimates = np.mean(estimates, axis=0).T
            mono_originals = np.mean(originals, axis=0).T
            SDR, SIR, SAR, perm = mir_eval.separation.bss_eval_sources(
                mono_estimates,
                mono_originals,
            )

            ISR = np.empty(SDR.shape)
            ISR[:] = np.NAN

        if verbose:
            print("SDR: ", str(SDR))
            print("ISR: ", str(ISR))
            print("SIR: ", str(SIR))
            print("SAR: ", str(SAR))

        return SDR, ISR, SIR, SAR
