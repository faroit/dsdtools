from __future__ import print_function
import numpy as np
import mir_eval
import pandas as pd
import json
from os import path as op
import scipy


class Data(object):
    def __init__(self, custom_column_keys=None):
        self.columns = [
            'track_id',
            'track_name',
            'target_name',
            'estimate_dir',
            'estimate_name',
            'SDR',
            'ISR',
            'SIR',
            'SAR',
            'sample',
            'subset'
        ]

        self.custom_column_keys = custom_column_keys

        if custom_column_keys is not None:
            self.columns.extend(self.custom_column_keys)

        self.df = pd.DataFrame(columns=self.columns)

    def row2series(self, **row_data):
        return pd.Series(row_data)

    def append(self, series):
        self.df = self.df.append(series, ignore_index=True)

    def to_pickle(self, filename):
        self.df.to_pickle(filename)

    def import_mat(self, filename, estimate_name='', custom_column_dict=None):
        mat = scipy.io.loadmat(filename)
        mdata = mat['result']
        ndata = {n.title(): mdata[n][0, 0] for n in mdata.dtype.names}
        s = []

        # some checks
        if self.custom_column_keys is not None:
            if custom_column_dict is None:
                raise(ValueError("Custom value data need to be submitted"))

            if set(custom_column_dict.keys()) != set(self.custom_column_keys):
                raise(
                    ValueError("Custom data for each custom data key \
                                needs to be provided")
                )

        for subset, subset_data in ndata.items():
            data = subset_data['results']
            for track in range(data.shape[1]):
                tdata = data[0, track][0, 0]
                for target in [
                    'vocals', 'drums', 'other', 'bass', 'accompaniment'
                ]:
                    frames = len(tdata[target][0]['sdr'][0][0])

                    has_nan_values = []
                    for metric in ['sdr', 'isr', 'sir', 'sar']:
                        score = tdata[target][0][metric][0][0]
                        has_nan_values.append(np.all(np.isnan(score)))

                    if any(has_nan_values):
                        # skip target
                        continue
                    else:
                        for frame in range(frames):
                            split_name = tdata['name'][0].split(' - ')
                            series = self.row2series(
                                track_id=int(split_name[0]),
                                track_name=tdata['name'][0],
                                target_name=target,
                                estimate_dir=filename,
                                estimate_name=estimate_name,
                                SDR=tdata[target][0]['sdr'][0][0][frame],
                                ISR=tdata[target][0]['isr'][0][0][frame],
                                SIR=tdata[target][0]['sir'][0][0][frame],
                                SAR=tdata[target][0]['sar'][0][0][frame],
                                sample=frame,
                                subset=subset
                            )
                            if custom_column_dict is not None:
                                custom_data_s = pd.Series(custom_column_dict)
                                series.append(custom_data_s)

                            s.append(series)
        self.append(s)

    def import_json(self, filename):
        with open(filename, 'r') as f:
            df = pd.read_json(f.read())
        # play with it a bit to get the columns and rows in the right order
        df = df.sort_index(ascending=True)
        base_cols = [
            'track_id',
            'track_name',
            'target_name',
            'estimate_dir',
            'estimate_name',
            'SDR',
            'ISR',
            'SIR',
            'SAR'
        ]
        new_cols = set(df.columns.tolist()).difference(set(base_cols))
        cols = base_cols + list(new_cols)
        df = df[cols]
        self.df = pd.concat([self.df, df], ignore_index=True)


class BSSeval(object):
    def __init__(
        self,
        window=30*44100,
        hop=15*44100,
        save_json=True
    ):
        self.data = Data()
        self.save_json = save_json
        self.window = window
        self.hop = hop

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

        if 'estimates_name' in user_estimates:
            estimates_name = user_estimates['estimates_name']
        else:
            estimates_name = op.basename(estimates_dir)

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
                                estimate_name=estimates_name,
                                SDR=SDR[i, k],
                                ISR=ISR[i, k],
                                SIR=SIR[i, k],
                                SAR=SAR[i, k],
                                sample=k * self.hop,
                                subset=track.subset
                            )
                        )

            except ValueError:
                pass

            if estimates_dir and rows and self.save_json:
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
                    with open(track_dir + '/evaluation.json', 'w+') as f:
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
