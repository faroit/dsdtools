from __future__ import print_function
import numpy as np
import mir_eval
from . import utils


class BSSeval(object):
    def __init__(
        self,
        collect=True,
        window=30*44100,
        hop=15*44100,
    ):
        self.data = utils.DF_writer([
            'track_id',
            'track_name',
            'target_name',
            'estimate_dir',
            'SDR',
            'ISR',
            'SIR',
            'SAR',
            'sample'
        ])
        self.window = window
        self.hop = hop

    def evaluate_track(self, track, user_estimates, estimates_dir=None):
        audio_estimates = []
        audio_reference = []

        # make sure to always build the list in the same order
        # therefore track.targets is an OrderedDict
        targets = []  # save the list of targets to be evaluated
        for key, target in list(track.targets.items()):
            try:
                # try to fetch the audio from the user_results of a given key
                estimate = user_estimates[key]
                # add the audio to the list of estimates
                audio_estimates.append(estimate)
                # add the audio to the list of references
                audio_reference.append(track.targets[key].audio)
                # append this target name to the list of labels
                targets.append(target)
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
                # iterate over all targets
                for i, target in enumerate(targets):
                    # iterate over all frames
                    for k in range(len(SDR[i])):
                        self.data.append(
                            track_id=int(track.id),
                            track_name=track.filename,
                            target_name=target.name,
                            estimate_dir=estimates_dir,
                            SDR=SDR[i, k],
                            ISR=ISR[i, k],
                            SIR=SIR[i, k],
                            SAR=SAR[i, k],
                            sample=k * self.hop
                        )

    def evaluate(self, estimates, references, verbose=True):
        """BSS_EVAL images evaluation using mir_eval.separation module

        Parameters
        ----------
        references : np.ndarray, shape=(nsrc, nsampl, nchan)
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

        sdr, isr, sir, sar, _ = mir_eval.separation.bss_eval_images_framewise(
            estimates,
            references,
            compute_permutation=False,
            window=self.window,
            hop=self.hop,
        )

        if verbose:
            print("SDR: ", str(sdr))
            print("ISR: ", str(isr))
            print("SIR: ", str(sir))
            print("SAR: ", str(sar))

        return sdr, isr, sir, sar
