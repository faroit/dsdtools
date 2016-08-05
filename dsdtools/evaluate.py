from __future__ import print_function
import numpy as np
import mir_eval
import pandas as pd
import json
from os import path as op


class Data(object):
    def __init__(self, columns):
        self.df = pd.DataFrame(columns=columns)
        self.columns = columns

    def row2series(self, **row_data):
        if set(self.columns) == set(row_data):
            return pd.Series(row_data)

    def append(self, series):
            self.df = self.df.append(series, ignore_index=True)

    def to_pickle(self, filename):
        self.df.to_pickle(filename)


class BSSeval(object):
    def __init__(
        self,
        window=30*44100,
        hop=15*44100,
    ):
        self.data = Data([
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

    # def plot_results(self, measures=['SDR', 'ISR', 'SIR', 'SAR']):
    #     figure, ax = plt.subplots(1, len(measures))
    #     for i, measure in enumerate(measures):
    #         sns.boxplot(
    #             "target_name",
    #             measure,
    #             hue='estimate_dir',
    #             data=self.data.df,
    #             showmeans=True,
    #             showfliers=False,
    #             palette=sns.color_palette('muted'),
    #             ax=ax[i],
    #             meanline=True,
    #         )
    #     return figure

    def evaluate_track(
        self,
        track,
        user_estimates,
        estimates_dir=None,
        verbose=False
    ):
        audio_estimates = []
        audio_reference = []

        rows = []

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

            try:
                SDR, ISR, SIR, SAR = self.evaluate(
                    audio_estimates, audio_reference
                )
                # iterate over all targets
                for i, target in enumerate(targets):
                    # iterate over all frames
                    if verbose:
                        print(target)
                        print("SDR: ", str(SDR[i]))
                        print("ISR: ", str(ISR[i]))
                        print("SIR: ", str(SIR[i]))
                        print("SAR: ", str(SAR[i]))

                    for k in range(len(SDR[i])):
                        rows.append(
                            self.data.row2series(
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
                        )

            except ValueError:
                pass

            if estimates_dir and rows:
                try:
                    # save the dataframe corresponding to this estimate as json
                    temp_frame = pd.DataFrame()
                    for row in rows:
                        temp_frame = temp_frame.append(row, ignore_index=True)
                    json_string = json.dumps(json.loads(temp_frame.to_json()),
                                             sort_keys=True,
                                             indent=4,
                                             separators=(',', ':'))
                    track_dir = op.join(
                        estimates_dir,
                        track.subset,
                        track.filename
                    )
                    with open(track_dir + '/evaluation.json', 'w') as f:
                        f.write(json_string)
                except (ValueError, IOError):
                    pass

        return rows

    def evaluate(self, estimates, references):
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
            references,
            estimates,
            compute_permutation=False,
            window=self.window,
            hop=self.hop,
        )

        return sdr, isr, sir, sar
