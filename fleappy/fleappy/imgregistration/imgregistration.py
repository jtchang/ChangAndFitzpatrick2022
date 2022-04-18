# -*- coding: utf-8 -*-
""" Image Registration Module.

This module provides a class for calculating and applying transformations to images during image registration. 

Example:
    Create an object and apply registration:
    ::code-block
    
        $ imgreg = ImageRegistration(reg_module=dftreg)
        $ imreg.register(filepath, seriesname)

Returns:
    [type]: [description]
"""
from fleappy.imgregistration.scanphase import fix_scan_phase
import logging
from pathlib import Path
import math

import imageio
import numpy as np
from numpy.lib.ufunclike import fix
import skimage.io as io
from collections import defaultdict
from fleappy.tiffread import scanimage
from . import phasecrosscorrelation

logger = logging.getLogger(__name__)


class ImageRegistration:
    """ Class for handling image registration.

    Attributes:
        reg_module (module): Module of registration functions. Must include register, transform, join, save, load, 
            create_template methods.
        files (dict): Dictionary of time series groups and files.
        transform (dict): Collection of transformations needed to register time series in files.
        template (numpy.ndarray): Template image to register to.
        reference (str): Reference time series to register all other time series to.
        directory (dict): Dictionary of paths for time series.
    """

    __slots__ = ['reg_module', 'files', 'transform', 'template', 'reference', 'directory', 'registration_settings']

    def __init__(self, reg_module=phasecrosscorrelation, template=None, registration_settings={}) -> None:
        for attr in ['register', 'transform', 'join', 'save', 'load', 'create_template']:
            assert hasattr(
                reg_module, attr), f'{reg_module.__name__} must have attribute {attr}'
        self.reg_module = reg_module
        self.files = {}
        self.transform = {}
        self.template = template
        self.reference = None
        self.directory = {}
        self.registration_settings = registration_settings

    def register(self, directory: str, seriesname: str, referenceseries=None, chunksize=2000) -> None:
        """ Register a collection of files.

        Given a directory and a series name, collect all the files with that series name and register them. This
        function automatically will parse scanimage 2015 headers to get channel and volumetric information, splitting
        registered images into individual files. To minimize memory usage, files are processed in chunks of frames that
        are each written to file. Registered images will be saved in a subdirectory of the image directory following 
        the pattern:

        *./<seriesname>/<piezo slice #>/stack_c<channel #>_<file #>.tif*

        Todo:
            * Multiple Channel Support

        Args:
            directory (str): Directory with images to register
            seriesname (str): Name of the series of data. Should follow format '<series>*.tif'
            referenceseries (str, optional): Defaults to None and will use the current series. Name of the series
                to register to.
            chunksize (int, optional): Defaults to 2000. Number of frames to include in the output file. Increasing
                this number may result in larger RAM usage.

        Returns:
            None
        """
        reg_settings = self.registration_settings

        # Collect Files
        files = _collect_files(directory, seriesname)
        self.files[seriesname] = [str(x.absolute()) for x in files]
        self.directory[seriesname] = directory
        logging.info('Search %s and found %s files', directory, len(files))
        self.transform[seriesname] = defaultdict(lambda: None)

        # Load first file, parse headers, figure out how to load files to meet the chunk size
        logging.info('Loading...%s', files[0].name)
        tif_stack, header = scanimage.read_si_tiffstack(
            files[0], header_only=False)
        frames_per_file = scanimage.frames_per_file(header)
        num_slices = scanimage.piezo_slices(header)
        channels = scanimage.channels(header)
        num_channels = len(channels)
        total_approximate_frames = len(files) * frames_per_file
        batch_chunk_size = chunksize * num_slices * num_channels
        logging.info('\nAligning using %s\n %i Slices \n %i Channels\n %i Frame Batch Size \n %i Total Frames Expected',
                     self.reg_module.__name__, num_slices, num_channels, batch_chunk_size, total_approximate_frames)

        # Prepare Attributes
        _, shape_y, shape_x = tif_stack.shape
        if self.template is None:
            self.template = np.empty(
                (shape_y, shape_x, num_slices), dtype=np.float)
            self.template[:] = np.nan
            self.reference = seriesname

        intermediate_template = np.empty(
            (num_slices, shape_y, shape_x), dtype=np.float)
        for slice_id in range(num_slices):
            target_path = Path(directory + '/' + seriesname +
                               '/Registered/slice' + str(slice_id + 1))
            if not target_path.exists():
                target_path.mkdir(parents=True)

        for batch_num in range(math.ceil(total_approximate_frames / batch_chunk_size)):
             # Start Loading Files
            batch_start_index = batch_num * batch_chunk_size
            batch_stop_index = (batch_num + 1) * batch_chunk_size - 1
            if batch_stop_index > total_approximate_frames-1:
                batch_stop_index = total_approximate_frames-1
            logging.info('Batch %i to %i', batch_start_index, batch_stop_index)
            file_start = math.floor((batch_start_index)/frames_per_file)
            if len(tif_stack) != 0:
                file_start = file_start + 1
            file_end = math.floor(batch_stop_index/frames_per_file)

            for file_num in range(file_start, file_end+1, 1):
                if file_num > len(files):
                    break
                logging.info('Loading... %s', files[file_num].name)
                file_stack, _ = scanimage.read_si_tiffstack(files[file_num], header_only=False)
                if len(tif_stack) != 0:
                    tif_stack = np.concatenate((tif_stack, file_stack), axis=0)
                else:
                    tif_stack = file_stack

            tif_start_idx = batch_start_index % frames_per_file
            tif_stop_idx = tif_start_idx+batch_chunk_size

            if tif_stop_idx > tif_stack.shape[0]:
                tif_stop_idx = tif_stack.shape[0]

            register_stack = tif_stack[tif_start_idx:tif_stop_idx, :, :]
            tif_stack = tif_stack[tif_stop_idx+1:, :, :]
            logger.info('Loaded Frames for Batch Register: %i: %i of %i',
                        tif_start_idx, tif_stop_idx - 1, register_stack.shape[0])

            # Register tif stack based on channel and piezo slice
            channel = 1
            for slice_id in range(num_slices):
                slice_start_idx = slice_id * num_channels
                # Either generate the template or use the previous template to generate an intermediate template
                if batch_num == 0:
                    scan_offset, projection = self.reg_module.create_template(register_stack
                                                                              [slice_start_idx::num_slices * num_channels, :, :],
                                                                              scanphase=reg_settings.get('scanphase', False))
                    if np.isnan(self.template[:, :, slice_id]).any():
                        self.template[:, :, slice_id] = projection
                        intermediate_template[slice_id, :, :] = self.template[:, :, slice_id].astype('int16')
                        logger.info('Creating New Template for %i', slice_id+1)
                    else:
                        transform_spec = self.reg_module.register(np.squeeze(self.template[:, :, slice_id]),
                                                                  projection[np.newaxis, :, :],
                                                                  **reg_settings)
                        intermediate_template[slice_id, :, :] = self.reg_module.transform(projection[np.newaxis, :, :].astype('int16'),
                                                                                          transform_spec)
                        logger.info('Using Old Template for %i', slice_id+1)
                if reg_settings.get('scanphase', False):
                    scan_offset, _ = fix_scan_phase(np.mean(register_stack[slice_start_idx:: num_slices *
                                                                           num_channels, 50:-50, :], axis=0))
                    logging.info('Using %f scan phase correction', scan_offset)
                logging.info('Registering slice %i for batch number %i', slice_id+1, batch_num+1)
                transform_spec = self.reg_module.register(np.squeeze(intermediate_template[slice_id, :, :]),
                                                          register_stack[slice_start_idx:: num_slices *
                                                                         num_channels, :, :],
                                                          scan_offset=scan_offset,
                                                          **reg_settings)
                logging.info('Transforming slice %i for batch number %i', slice_id+1, batch_num+1)
                corrected = self.reg_module.transform(register_stack[slice_start_idx:: num_slices * num_channels, :, :],
                                                      transform_spec)

                # Write the file and save the transform
                filename = 'stack_c{0}_{1}.tif'.format(channel, batch_num+1)
                logger.info('writing to File: %s', filename)
                target_path = Path(
                    directory + '/' + seriesname + '/Registered/slice' + str(slice_id + 1))
                _write_tiff(corrected, target_path.joinpath(filename))

                self.transform[seriesname][slice_id] = self.reg_module.join(
                    self.transform[seriesname][slice_id], transform_spec)
                logging.info(self.transform[seriesname][slice_id].shape)

        # Save the transform and template
        for slice_id in range(num_slices):
            target_path = Path(directory + '/' + seriesname +
                               '/Registered/slice' + str(slice_id + 1))
            _write_tiff(np.squeeze(self.template[:, :, slice_id]).astype('int16'),
                        target_path.joinpath('MasterTemplate.tif'))
            _write_tiff(intermediate_template[slice_id, :, :].astype('int16'),
                        target_path.joinpath('IntermediateTemplate.tif'))
            self.reg_module.save(
                self.transform[seriesname][slice_id], target_path.joinpath('transform.tspec'))
            scanimage.to_json(header, target_path.joinpath('header.json'))
        return None

    def batch_register(self, directory: str, serieslist: list, referenceseries: str = None) -> None:
        """Batch registration of multiple time series.


        TODO:
            * Batch register

        Args:
             directory (str): Directory with images to register
            serieslist (list): List of the names of the time series to register. Should follow format '<series>*.tif'
            referenceseries (str, optional): Defaults to None and will use the current series. Name of the series
                to register to.

        Raises:
            NotImplementedError: [description]

        Returns:
            None: [description]
        """

        raise NotImplementedError('Batch register is not yet implemented')


def _collect_files(pathname, seriesname) -> list:
    pth = Path(pathname)
    assert pth.exists(), 'Unknown Path!'

    tiffs = pth.glob('{0}*.tif'.format(seriesname))

    return list(tiffs)


def _write_tiff(img_stack, path_name) -> None:
    if isinstance(path_name, str):
        assert path_name.endswith('.tif'), 'Please specify a .tif filename'
        path_to_write = Path(path_name)
    elif isinstance(path_name, Path):
        assert(path_name.suffix == '.tif'), 'Please specify a .tif'
        path_to_write = path_name
    else:
        raise 'Unknown file type!'

    if len(img_stack.shape) < 3:
        img_stack = img_stack[np.newaxis, :, :]

    imageio.mimwrite(path_to_write, img_stack)
