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


#HOW TO USE THE LOADMAT FUNCTION TO LOAD"RESULTFILE.MAT FROM UNIT CODE INTO SPIKEINTERFACE

#Adding the code for comparision of spike sorting data with sorting extractor
from scipy.io import loadmat
x = loadmat('resultfile.mat')
sortingPipeline = se.NumpySortingExtractor() #initiating the sorting extractor object(refer spikeinterface tut on details aboyt sorting extractor)
labels=x['idx'] #(Spike labels)
times_reference=x['spike_idxs'] #(times where spikes are detected)
sampleRate=x['Fs']  #(sampling frequency)
sortingPipeline.set_times_labels(times=times_reference[0], labels=labels)   #(FEED THIS DATA FROM RESULT FILE TO SORTING EXTRACTOR)
sortingPipeline.set_sampling_frequency(sampling_frequency=sampleRate)


print('Unit ids = {}'.format(sortingPipeline.get_unit_ids()))#(VERIFY DATA CORRECTLY STORED INTO SORTING EXTRACTOR)
st = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1 = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
w_rs_gt = sw.plot_rasters(sortingPipeline,sampling_frequency=sampleRate)




