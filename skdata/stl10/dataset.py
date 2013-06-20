"""
STL-10 classification dataset

Data available from and described at:
http://www.stanford.edu/~acoates/stl10/

If you use this dataset, please cite
Adam Coates, Honglak Lee, Andrew Y. Ng "An Analysis of Single Layer Networks in Unsupervised Feature Learning" AISTATS, 2011.

Dataset downloadable here:
http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz

"""

# Authors: Alex Susemihl (alexsusemihl@gmail.com)
# License: BSD 3 clause

import os
import cPickle
from scipy.io import loadmat
import logging
import shutil

import numpy as np

from ..data_home import get_data_home
from ..utils.download_and_extract import download_and_extract

logger = logging.getLogger(__name__)

URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz'

CLASSNAMES  = [u'airplane', u'bird', u'car', u'cat', u'deer', u'dog', u'horse',
          u'monkey', u'ship', u'truck']

class STL10(object):
    """

    meta[i] is dict with keys:
        id: int identifier of this example
        label: int in range(10)
        split: 'train' or 'test'

    meta_const is dict with keys:
        image:
            shape: 32, 32, 3
            dtype: 'uint8'


    """

    DOWNLOAD_IF_MISSING = True  # the value when accessing .meta

    def __init__(self):
        self.meta_const = dict(
                image = dict(
                    shape = (32, 32, 3),
                    dtype = 'uint8',
                    )
                )
        self.descr = dict(
                n_classes = 10,
                )

    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=self.DOWNLOAD_IF_MISSING)
            self._meta = self.build_meta()
            return self._meta
    meta = property(__get_meta)

    def home(self, *names):
        return os.path.join(get_data_home(), 'STL10', *names)

    def fetch(self, download_if_missing):
        if os.path.isdir(self.home('STL10_data')):
            return

        if not os.path.isdir(self.home()):
            if download_if_missing:
                os.makedirs(self.home('STL10_data'))
            else:
                raise IOError(self.home())

        download_and_extract(URL, self.home('STL10_data'))

    def clean_up(self):
        logger.info('recursively erasing %s' % self.home())
        if os.path.isdir(self.home()):
            shutil.rmtree(self.home())

    def build_meta(self):
        try:
            self._pixels
        except AttributeError:
            # load data into class attributes _pixels and _labels
	    #the first 5000 examples are labeled, all the following aren't
            pixels = np.zeros((113000, 96, 96, 3), dtype='uint8')
	    labels = np.zeros(13000, dtype='int32')
            fnames = ['train.mat','test.mat','unlabeled.mat']

            # load train and validation data
	    n_loaded = 0
            for i, fname in enumerate(fnames):
		    data = self.load_from_mat(filename)
		    assert data['X'].dtype == np.uint8
		    def futz(X):
			    return X.reshape((X.shape[0], 96, 96,3),order='F')
		    if fname!='unlabeled.mat':
			    labels[n_loaded:n_loaded + data['X'].shape[0]] = data['y']
		    pixels[n_loaded:n_loaded + data['X'].shape[0]] = futz(data['X'])
		    n_loaded += data['X'].shape[0]
            
	    STL10._pixels = pixels
	    STL10._unlabeled_pixels = unlabeled_pixels
            STL10._labels = labels

            # -- mark memory as read-only to prevent accidental modification
            pixels.flags['WRITEABLE'] = False
            labels.flags['WRITEABLE'] = False

        meta = [dict(
                    id=i,
                    split='train' if i < 5000 else ('test' if i<13000 else 'unlabeled'),
                    label=LABELS[l] if i < 13000 else None)
                for i,l in enumerate(self._labels)]
        return meta

    def load_from_mat(self, basename):
        fname= self.home('STL10_data', basename)
	logger.info('loading file %s' % fname)
        data = loadmat(fname)
        return data

