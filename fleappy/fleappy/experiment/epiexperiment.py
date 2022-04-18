from itertools import chain, tee
import logging
import natsort as ns
import numpy as np
from math import ceil
import os
from pathlib import Path
from pims.bioformats import BioformatsReader

from skimage.transform import downscale_local_mean, resize
from skimage.io import imsave, imread

from scipy.ndimage.filters import percentile_filter

import fleappy.imgregistration.dsreg as reg_module
from fleappy.experiment.baseexperiment import BaseExperiment
from fleappy.metadata.epimetadata import EpiMetadata
from fleappy.roimanager import imagejroi
from fleappy.filter import fermi
from fleappy.analysis.epiorientation import EpiOrientationAnalysis
from fleappy.analysis.epispontaneous import EpiSpontaneousAnalysis

BATCH_SIZE = 2000


class EpiExperiment(BaseExperiment):
    RESOLUTION = 1000/172  # microns per pixel for 4x4 binning
    __slots__ = ['files', 'animal_id']

    def __init__(self, path: str, expt_id: str, file_dir: str, animal_id: str = None, remote_path=None, **kwargs):
        """Constructor for Epi Experiment

        Args:
            path (str): Base path for expeirment files
            expt_id (str): Folder for spike2 dta
            file_dir (str): Folder for tif files
            animal_id (str, optional): Animal ID. Defaults to None.
            remote_path ([type], optional): Remote location to load data from. Defaults to None.
        """
        global BATCH_SIZE
        BaseExperiment.__init__(self, animal_id=animal_id, **kwargs)
        self.files = {'path': path, 'directory': file_dir, 'fnames': {}, 'remote': remote_path}
        self.metadata = EpiMetadata(path=path, expt_id=expt_id, **kwargs)
        self.metadata.load_stims()
        self.animal_id = animal_id

        self.files['pattern_noise'] = kwargs['pattern_noise'] if 'pattern_noise' in kwargs else None

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        for key in chain.from_iterable(getattr(cls, '__slots__', []) for cls in EpiExperiment.__mro__):
            str_ret = str_ret + f'{key}:{getattr(self, key)}{os.linesep}'

        return str_ret

    def add_analysis(self, field: str, **kwargs):
        """Add an analysis to experiment.

        This function adds an analysis to the experiment. Currently it automatically adds an analysis of orientation of
        the stimulus type is driftingGrating.

        TODO:
            * Let users overwrite the default analysis

        Args:
            analysis_id (str): Name for analysis set.
            field (str): Field to use analysis
        """
        analysis = None

        if self.metadata.stim_type() == 'driftingGrating':
            analysis = EpiOrientationAnalysis(self, field, **kwargs)
        elif self.metadata.stim_type() == 'blackScreen':
            analysis = EpiSpontaneousAnalysis(self, field, **kwargs)

        if analysis is not None:
            logging.info('Creating Analysis %s', analysis.id)
            self.analysis[analysis.id] = analysis

    def update_remote_path(self, remote_path) -> None:
        if isinstance(remote_path, str):
            self.files['remote'] = Path(remote_path)
        elif isinstance(remote_path, Path) and remote_path.exists():
            self.files['remote'] = remote_path
        else:
            FileNotFoundError()

    def update_path(self, path):
        super().update_path(path)
        self.files['path'] = path

        self.load_files(override=True)
        self.load_roi()

    def load_files(self, override=False) -> None:
        """Collect Bioformat tif.

            N.B. Bioformat tifs handles multiple tif files so only the first needs to be loaded.
            TODO:
                *Handle the case where bioformats tifs are not being used.
        """
        # Load rawF
        if 'rawF' not in self.files['fnames'].keys() or override:
            file_path = self._tif_path()
            files_in_dir = ns.natsorted(list(file_path.glob('*.ome.tif')), alg=ns.PATH)
            if len(files_in_dir) > 0:
                files_in_dir = [files_in_dir[0]]
            else:
                logging.info('OME file format not found using .tif')
                files_in_dir = ns.natsorted(list(file_path.glob('*.tif')), alg=ns.PATH)

            logging.debug(f'Added in files {files_in_dir}')
            self.files['fnames']['rawF'] = files_in_dir

        # load registered

        file_in_dir = ns.natsorted(list(self._tif_path('registered').glob('Registered_tseries.tif')), alg=ns.PATH)
        if len(file_in_dir) == 0:
            file_in_dir = ns.natsorted(
                list(self._tif_path('registered', remote=True).glob('Registered_tseries.tif')), alg=ns.PATH)

        if len(file_in_dir) > 0:
            logging.info(f'Added {str(file_in_dir)}')
            self.files['fnames']['registered'] = file_in_dir

        for ftype in ['baseline', 'dff']:
            file_in_dir = ns.natsorted(list(self._tif_path(ftype).glob('*.tif')), alg=ns.PATH)
            if len(file_in_dir) == 0:
                file_in_dir = ns.natsorted(list(self._tif_path(ftype, remote=True).glob('*.tif')), alg=ns.PATH)
            for fid in file_in_dir:
                logging.info(f'Added {str(fid)}')
                self.files['fnames'][fid.stem] = [fid]

    def load_roi(self):
        """Load a ImageJ ROI file into epi imaging experiment.

        ImageJ ROI file loaded from the tif file directory
        """
        file_path = self._roi_path()
        roi_files = ns.natsorted(list(file_path.glob('*.roi')), alg=ns.PATH)
        if len(roi_files) > 0:
            logging.debug('Addiing roi from %s', roi_files[0])
            self.files['mask'] = roi_files[0].absolute()
        else:
            logging.warning('No ROI found in %s', file_path.absolute)

    def noise_correct(self, noise_path, filename='denoised_tseries.tif', overwrite=False):

        target_file_path = Path(os.path.join(
            str(self._tif_path().absolute()), 'denoised'))
        target_tif_file = Path(os.path.join(target_file_path, filename))

        if not overwrite and target_tif_file.exists():
            logging.info(
                f'{self.animal_id}:Found previous denoise images and using {str(target_tif_file)}')
        else:
            pattern_noise = imread(noise_path)
            if len(pattern_noise.shape) == 3:
                pattern_noise = np.mean(pattern_noise, axis=0)

            if not Path(target_file_path).exists():
                Path(target_file_path).mkdir(parents=True, exist_ok=True)
            elif target_tif_file.exists():
                logging.info('Removing previous registered file')
                os.remove(target_tif_file)

            freader = self._access_files(series_type='rawF')
            total_frames = freader.sizes['t']
            for frame_idx in range(0, total_frames, BATCH_SIZE):
                frames = self.get_frames(
                    frame_idx, frame_idx+BATCH_SIZE, scaling_factor=(1, 1, 1), file_reader=freader)
                frames = frames - np.broadcast_to(pattern_noise, frames.shape)
                frames[frames < 0] = 0
                imsave(str(target_tif_file), frames.astype(
                    'uint16'), plugin='tifffile', append=True, bigtiff=True, check_contrast=False)
            freader.close()
        self.files['fnames']['denoised'] = [target_tif_file]

    def register_images(self, template_path=None, series_type='rawF', overwrite=False,
                        filename='Registered_tseries.tif', noise_path=None, register=None):
        """[summary]

        Args:
            template_path ([type], optional): [description]. Defaults to None.
            series_type: 
            overwrite (bool, optional): [description]. Defaults to False.
            filename:
            noise_path
            register: 
        """

        target_file_path = self._tif_path('registered', remote=False)

        scaling_factor = (1, 1, 1)

        if not Path(target_file_path).exists():
            remote_file_path = self._tif_path('registered', remote=True)
            if remote_file_path.exists():
                target_file_path = remote_file_path
            else:
                Path(target_file_path).mkdir(parents=True, exist_ok=True)

        logging.debug('Using %s as registered', str(target_file_path.absolute()))

        target_tif_file = Path(os.path.join(target_file_path, filename))
        target_tspec_file = Path(os.path.join(target_file_path, 'Registered_tseries.tspec'))

        self._clean_register_files(target_tif_file, overwrite, target_tspec_file)
        if target_tif_file.exists():
            self.files['fnames']['registered'] = [target_tif_file]
        elif target_tspec_file.exists():
            freader = self._access_files(series_type=series_type)
            tspec = reg_module.load(target_tspec_file)
            total_frames = freader.sizes['t']
            for frame_idx in range(0, total_frames):
                frame = self.get_frames(
                    frame_idx, frame_idx+1, scaling_factor=(1, 1, 1), file_reader=freader)
                registered_frame = reg_module.transform(
                    frame, tspec[frame_idx, :])
                imsave(str(target_tif_file), registered_frame.astype(
                    'uint16'), plugin='tifffile', append=True, bigtiff=True, check_contrast=False)
            freader.close()
        else:
            freader = self._access_files(series_type=series_type)
            tspec = np.zeros((freader.sizes['t'], 2), dtype=np.int8)
            total_frames = freader.sizes['t']

            # Create Dark Noise Mask
            noise_path = self.files['pattern_noise'] if noise_path is None else noise_path
            if noise_path is not None:

                pattern_noise = imread(noise_path)
                if len(pattern_noise.shape) == 3:
                    pattern_noise = np.mean(pattern_noise, axis=0)

            if template_path is None:
                template = np.mean(self.get_frames(
                    0, 1000, scaling_factor=scaling_factor, file_reader=freader), axis=0)
                if noise_path is not None and template.shape == pattern_noise.shape:
                    template = template-pattern_noise
                    template[template < 0] = 0
                else:
                    noise_path = None
            else:
                template = imread(template_path)

            imsave(os.path.join(target_file_path, 'template.tif'),
                   template.astype('uint16'), check_contrast=False)

            intermediate_template = np.mean(self.get_frames(
                0, 1000, scaling_factor=(1, 1, 1), file_reader=freader), axis=0)
            intermediate_template = np.expand_dims(intermediate_template, axis=0)

            if register is not None:
                transform_temp = reg_module.register(template, intermediate_template, {
                    'maxmovement': 1/10, 'downsamplerates': [1/4, 1/2, 1]})
            else:
                transform_temp = np.zeros((1, 2))

            template = reg_module.transform(intermediate_template, transform_temp)
            template = np.mean(template, axis=0)
            intermediate_template = None
            transform_temp = None

            for frame_idx in range(0, total_frames, BATCH_SIZE):
                logging.debug('%s: starting batch %i of %i',
                              self.animal_id,
                              int(frame_idx/BATCH_SIZE + 1),
                              int(ceil(total_frames/BATCH_SIZE) + 1))

                end_idx = frame_idx + BATCH_SIZE if frame_idx + \
                    BATCH_SIZE < total_frames else total_frames

                frame_batch = self.get_frames(
                    frame_idx, end_idx, scaling_factor=(1, 1, 1), file_reader=freader)

                if noise_path is not None:
                    for fidx, frame in enumerate(frame_batch):
                        if frame.shape == pattern_noise.shape:
                            frame_batch[fidx, :, :] = frame-pattern_noise
                    frame_batch[frame_batch < 0] = 0
                if register is not None:
                    tspec[frame_idx:end_idx, :] = reg_module.register(
                        template, frame_batch, {'maxmovement': 1/10, 'downsamplerates': [1/4, 1/2, 1]})

                registered = reg_module.transform(frame_batch, tspec[frame_idx:end_idx, :])
                imsave(str(target_tif_file), registered.astype('uint16'),
                       plugin='tifffile', check_contrast=False, append=True, bigtiff=True)
            reg_module.save(tspec, target=target_tspec_file)
            freader.close()
        self.files['fnames']['registered'] = [target_tif_file]

    def percentile_baseline_series(
            self, percentile, window, scaling_factor=(10, 1, 1),
            series_type='rawF', target_name=None, overwrite=False, batch_process=True):
        """Baselines the

        [description]

        TODO:
            * Speed up this function. It is currently horribly inefficient

        Args:
            percentile ([type]): [description]
            window ([type]): [description]
            series_type (str, optional): Defaults to 'rawF'. [description]
        """

        frame_window = int(np.round(window * self.metadata.frame_rate()/scaling_factor[0]))
        file_frame_window = int(scaling_factor[0]*frame_window)

        target_file_path = Path(os.path.join(str(self._tif_path(remote=False).absolute()), 'baseline'))
        if not Path(target_file_path).exists():
            Path(target_file_path).mkdir(parents=True, exist_ok=True)

        if target_name is None:
            target_file_name = f'baseline_{scaling_factor[0]}_{scaling_factor[1]}_{scaling_factor[2]}_p{percentile}_w{window}'
        else:
            target_file_name = target_name
        target_tif_file = os.path.join(target_file_path, target_file_name + '.tif')

        self._clean_register_files(Path(target_tif_file), overwrite)

        if not Path(target_tif_file).exists():

            freader = self._access_files(series_type=series_type)
            total_frames = freader.sizes['t']
            use_batch_size = BATCH_SIZE if batch_process else total_frames
            for batch_idx, frame_idx in enumerate(range(0, total_frames, use_batch_size)):
                logging.debug(
                    f'{self.animal_id}: Loading Time Series batch {batch_idx+1} of batch {ceil(total_frames/use_batch_size)}')

                file_start_idx = frame_idx - file_frame_window if frame_idx > file_frame_window else 0
                file_end_idx = frame_idx + use_batch_size + file_frame_window if frame_idx + \
                    use_batch_size + file_frame_window < total_frames else total_frames

                batch_series = self.get_frames(
                    start=file_start_idx, stop=file_end_idx, scaling_factor=scaling_factor, file_reader=freader)

                batch_start_idx = int(np.round(frame_idx-file_start_idx)/scaling_factor[0])
                batch_end_idx = int(batch_start_idx + use_batch_size/scaling_factor[0])
                if batch_end_idx > batch_series.shape[0]:
                    batch_end_idx = batch_series.shape[0]

                baseline_dx = percentile_filter(batch_series, percentile, size=(frame_window, 1, 1))
                baseline = np.empty(
                    ((batch_end_idx - batch_start_idx) * scaling_factor[0],
                        batch_series.shape[1],
                        batch_series.shape[2]))
                for idx in range(0, scaling_factor[0]):
                    baseline[idx::scaling_factor[0], :, :] = baseline_dx[batch_start_idx:batch_end_idx, :, :]

                imsave(str(target_tif_file), baseline.astype(
                    'uint16'), plugin='tifffile', check_contrast=False, append=True, bigtiff=True)
            freader.close()
        self.files['fnames'][target_file_name] = [Path(os.path.join(target_file_path, target_file_name+'.tif'))]

    def create_dff(self, baseline_name=None, target_name=None, raw_series=None, overwrite=False, batch_process=True):
        if raw_series is None:
            raw_series = 'registered' if 'registered' in self.files['fnames'].keys(
            ) else 'rawF'
        target_file_path = Path(os.path.join(
            str(self._tif_path().absolute()), 'dff'))

        if baseline_name is None:
            baselines = ns.natsorted(
                [fname for fname in self.files['fnames'].keys() if 'baseline' in fname])
            if len(baselines) > 1:
                logging.warning(
                    f'More than one baseline found using {baselines[0]}')
            baseline_name = baselines[0]

        if target_name is None:
            target_file_name = baseline_name.replace('baseline', 'dff')
        else:
            target_file_name = target_name

        if not Path(target_file_path).exists():
            Path(target_file_path).mkdir(parents=True, exist_ok=True)
        target_tif_file = Path(os.path.join(
            target_file_path, target_file_name+'.tif'))

        self._clean_register_files(target_tif_file, overwrite)

        if not target_tif_file.exists():

            raw_freader = self._access_files(series_type=raw_series)
            bl_freader = self._access_files(series_type=baseline_name)
            total_frames = raw_freader.sizes['t']

            use_batch_size = BATCH_SIZE if batch_process else total_frames

            scaling_factor = scaling_factor = (1, int(
                raw_freader.sizes['y']/bl_freader.sizes['y']), int(raw_freader.sizes['x']/bl_freader.sizes['x']))

            for frame_idx in range(0, total_frames, use_batch_size):
                logging.debug(
                    f'{self.animal_id}: starting batch {frame_idx/use_batch_size+1} of {ceil(total_frames/use_batch_size)}')
                stop_idx = frame_idx+use_batch_size if frame_idx + \
                    use_batch_size < total_frames else total_frames

                baseline = self.get_frames(start=frame_idx, stop=stop_idx, file_reader=bl_freader)
                rawF = self.get_frames(start=frame_idx, stop=stop_idx,
                                       file_reader=raw_freader, scaling_factor=scaling_factor)

                dff = (rawF-baseline)/baseline

                imsave(str(target_tif_file), dff.astype('double'),
                       plugin='tifffile', append=True, bigtiff=True)
            raw_freader.close()
            bl_freader.close()
        self.files['fnames'][target_file_name] = [target_tif_file]

    # Accessor Methods

    def pixel_frame_size(self, series_type='rawF'):
        freader = self._access_files(series_type=series_type)
        frame_size = (freader.sizes['y'], freader.sizes['x'])
        freader.close()

        return frame_size

    def resolution(self, series_type='rawF'):
        original_frame_size = self.pixel_frame_size(series_type='rawF')
        frame_size = self.pixel_frame_size(series_type=series_type)
        original_resolution = self.metadata.imaging['resolution']

        return tuple([old/new * res for old, new, res in zip(original_frame_size, frame_size, original_resolution)])

    def roi(self, series_type=None):
        """Load and returns the appropriate mask for the given series

        [description]


        Returns:
            [type]: Boolean numpy array
        """

        if series_type is None:
            series_type = 'rawF'

        if series_type not in self.files['fnames'].keys():
            raise FileNotFoundError('Unknown series type')

        if 'mask' in self.files.keys():
            roi = imagejroi.roiread(str(self.files['mask'].absolute()))

            original_frame_size = self.pixel_frame_size(series_type='rawF')
            mask = imagejroi.to_array(roi, framesize=original_frame_size)
            frame_size = self.pixel_frame_size(series_type=series_type)
            mask = resize(mask, frame_size)
            mask = mask.astype('bool')
            logging.debug('Converting frame of size %s to %s', str(original_frame_size), str(frame_size))
        else:
            frame_size = self.pixel_frame_size(series_type=series_type)
            mask = np.ones(frame_size, dtype=bool)

        return mask

    def get_frames(self, start: int, stop: int, scaling_factor=(1, 1, 1),
                   file_reader=None, series_type='rawF', chunk_size=2000, **kwargs) -> np.array:
        """Load frames as a numpy array [start,stop)

        Args:
            start (int): Starting frame to load (inclusive).
            stop (int): Ending frame to load (non-inclusive).
            scaling_factor (tuple, optional): Defaults to (1,1,1). Downscaling factor in (t, y, x)

        Returns:
            np.array: Numpy array of frames from start to stop

        TODO:
            * Handle case of non-bioformats
            * This implementation will almost certainly break if the scaling factor for t isn't the same as the chunk size
        """

        if file_reader is None:
            file_reader = self._access_files(series_type=series_type)

        if stop == -1:
            stop = file_reader.sizes['t']
        elif stop > file_reader.sizes['t']:
            stop = file_reader.sizes['t']
        logging.debug(f'{start}:{stop} - {file_reader.sizes}')

        logging.debug(ceil((stop-start)))
        total_frames = ceil((stop-start)/scaling_factor[0])

        if total_frames < 0:
            logging.error(f'{start}:{stop} - {total_frames} is a NEGATIVE number!')

        y = ceil(file_reader.sizes['y']/scaling_factor[1])
        x = ceil(file_reader.sizes['x']/scaling_factor[2])
        logging.debug((total_frames, y, x))
        try:
            frame_stack = np.empty((total_frames, y, x))
        except:
            logging.error(f'{start}:{stop} - {total_frames} is a NEGATIVE number!')

        frame_start_list = np.append(
            np.arange(start, stop, chunk_size), stop)
        if len(self.files['fnames'][series_type]) > 0:
            for start_frame, stop_frame in EpiExperiment._pairwise(frame_start_list):
                logging.debug(
                    f'Loading {start_frame}:{stop_frame} frame to {int((start_frame-start)/scaling_factor[0])}:{ceil((stop_frame-start)/scaling_factor[0])}')
                frame_stack[int((start_frame-start)/scaling_factor[0]):ceil((stop_frame-start)/scaling_factor[0]),
                            :, :] = downscale_local_mean(np.array(file_reader[start_frame:stop_frame]), scaling_factor)

        else:
            raise Exception('Unknown file type')

        return frame_stack

    def get_trial_responses(
            self, prepad=1, postpad=1, dff=False, series_type='rawF', prestim_bl=False, **kwargs) -> tuple:
        """[summary]

        Args:
            prepad (int, optional): [description]. Defaults to 1.
            postpad (int, optional): [description]. Defaults to 1.
            dff (bool, optional): [description]. Defaults to False.
            mask (bool, optional): [description]. Defaults to True.
            series_type (str, optional): [description]. Defaults to 'rawF'.
            prestim_bl (bool, optional): [description]. Defaults to False.

        Returns:
            tuple: Trials Responses (stims, trials, time, x, y), Trial Masks (pretim, stim, poststim)
        """
        frames = int(np.round(
            float(self.metadata.stim['stimDuration']) * self.metadata.frame_rate()))

        pre_stim = int(np.round(prepad * self.metadata.frame_rate()))
        post_stim = int(np.round(postpad*self.metadata.frame_rate()))

        total_triggers = self.metadata.num_stims()
        total_trials = self.metadata.num_trials()
        freader = self._access_files(series_type=series_type)

        x_size = int(np.round(freader.sizes['y']))
        y_size = int(np.round(freader.sizes['x']))

        trial_responses = np.empty(
            [total_triggers, total_trials, pre_stim+frames+post_stim, x_size, y_size], dtype='float64')
        logging.info('[%s] Processing Stimuli...', self.animal_id)
        for stim_index, stim_id in enumerate(np.sort(np.unique(self.metadata.stim['triggers']['id']))):
            logging.debug('[%s] Process Stim # %i', self.animal_id, stim_id)
            for trial, timestamp in enumerate(
                    [x['time'] for x in self.metadata.stim['triggers'] if x['id'] == stim_id]):
                if trial >= self.metadata.num_trials():
                    break
                start_frame, _ = self.metadata.find_frame_idx(timestamp)
                logging.debug('[%s] Process Trial # %i %i to %i', self.animal_id,
                              trial, start_frame-pre_stim, start_frame+frames+post_stim)
                stim_response = self.get_frames(start_frame-pre_stim, start_frame+frames+post_stim,
                                                file_reader=freader, scaling_factor=(1, 1, 1))

                if pre_stim != 0 and (dff or prestim_bl):
                    prestim_response = np.mean(
                        stim_response[0:pre_stim, :, :], axis=0)
                    prestim_response = np.expand_dims(prestim_response, 0)
                    prestim_response = np.tile(
                        prestim_response, (stim_response.shape[0], 1, 1))
                    if dff:
                        trial_responses[stim_index, trial, :, :, :] = (
                            stim_response-prestim_response) / prestim_response
                    else:
                        trial_responses[stim_index, trial, :,
                                        :, :] = stim_response-prestim_response
                else:
                    trial_responses[stim_index, trial, :, :, :] = stim_response

        freader.close()
        trial_masks = np.zeros((trial_responses.shape[2], 3), dtype=bool)
        trial_masks[0:pre_stim, 0] = 1
        trial_masks[pre_stim:pre_stim+frames, 1] = 1
        trial_masks[pre_stim+frames:pre_stim+frames+post_stim, 2] = 1
        return trial_responses, trial_masks

    def get_avg_img(self, **kwargs):
        kwargs['start'] = kwargs.get('start', 0)
        kwargs['stop'] = kwargs.get('stop', 2000)
        kwargs['series_type'] = kwargs.get('series_type', 'rawF')
        kwargs['roi'] = kwargs.get('roi', False)

        img = np.mean(self.get_frames(**kwargs), axis=0)

        if kwargs['roi']:
            img[~self.roi(series_type=kwargs['series_type'])] = np.nan

        return img

    # Private

    def _clean_register_files(self, target_tif_file, overwrite, target_tspec_file=None):
        if target_tif_file.exists():
            try:
                self.files['fnames']['tmp'] = [target_tif_file]
                reg_freader = self._access_files('tmp')
                raw_freader = self._access_files('rawF')

                reg_size = reg_freader.sizes['t']
                raw_size = raw_freader.sizes['t']

                reg_freader.close()
                raw_freader.close()
                self.files['fnames']['tmp'] = None
            except Exception as err:
                logging.error('[%s] Could not process %s: %s', self.animal_id, target_tif_file, repr(err))
                overwrite = True
            if overwrite or reg_size < raw_size:
                logging.info('[%s] Removing previous file %s', self.animal_id, target_tif_file)
                os.remove(target_tif_file)
                if target_tspec_file is not None and target_tspec_file.exists():
                    os.remove(target_tspec_file)

    def _tif_path(self, tif_type=None, remote=False):

        root_path = self.files['remote'] if remote and self.files['remote'] is not None else self.files['path']

        if tif_type is None:
            return Path(root_path, self.files['directory'])
        elif tif_type in ['registered', 'baseline', 'dff']:
            return Path(root_path, self.files['directory'], tif_type)
        else:
            raise TypeError('Unknown file type %s looking in %s', tif_type, root_path)

    def _roi_path(self):
        return Path(self.files['path'], self.files['directory'])

    def _access_files(self, series_type=None):

        if series_type is None or series_type is 'rawF':
            return BioformatsReader(
                str(self.files['fnames']['rawF'][0]),
                series=0, java_memory=os.getenv("MAX_JAVA_HEAP"))
        elif series_type in list(self.files['fnames'].keys()) and len(self.files['fnames'][series_type]) == 1:
            return BioformatsReader(
                str(self.files['fnames'][series_type][0]),
                series=0, java_memory=os.getenv("MAX_JAVA_HEAP"))
        else:
            raise TypeError('Unknown file type')

    @staticmethod
    def _pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
