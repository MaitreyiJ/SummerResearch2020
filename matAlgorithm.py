from spikeforest2 import sorters
from spikeforest2 import processing
import hither_sf as hither
import kachery as ka
import json
import numpy as np
from spikeinterface import widgets as sw
from typing import List, Union, Dict
from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor
from spikeforest2_utils.autoextractors.mdaextractors.mdaextractors import MdaRecordingExtractor
import spikeinterface.comparison as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import spikeinterface.toolkit as st
import spikeinterface.extractors as se
from spikeforest2_utils import writemda32
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import PLSRegression
from mpl_toolkits import mplot3d


#so we are done with the pipeline, now we need to compare this algorithm to ground truth
#We will be comparing the features of each of our waveforms(waveform) with the mean_cluster forms
#for spike classification
#time = np.linspace(0, wave_form.shape[1]/sampleRate, wave_form.shape[1])*1000
#cluster_mean = wave_form[cluster==0, :].mean(axis=0)

#Adding the code for comparision of spike sorting data with sorting extractor
from scipy.io import loadmat
x = loadmat('resultfile.mat')
sortingPipeline = se.NumpySortingExtractor()
labels=x['idx']
times_reference=x['spike_idxs']
sampleRate=x['Fs']
sortingPipeline.set_times_labels(times=times_reference[0], labels=labels)
sortingPipeline.set_sampling_frequency(sampling_frequency=sampleRate)


print('Unit ids = {}'.format(sortingPipeline.get_unit_ids()))
st = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1 = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
w_rs_gt = sw.plot_rasters(sortingPipeline,sampling_frequency=sampleRate)




