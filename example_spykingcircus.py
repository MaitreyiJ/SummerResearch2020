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
import spikeinterface.sorters as ss
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
import spikeforest_widgets as sw2

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
times_reference=x['spike_min_idxs']
Fs=x['Fs']
timeseries=x['mcs_data']
geometry=x['geom']

x2 = loadmat('script(2)_idxs.mat')
labels2=x2['idx']
x2r=loadmat('script(2)_minidxs.mat')
times_reference2=x2r['spike_min_idxs']

x3 = loadmat('script(3)_idxs.mat')
labels3=x3['idx']
x3r=loadmat('script(3)_minidxs.mat')
times_reference3=x3r['spike_min_idxs']





#concatenating the data 
times_reference_joined=np.concatenate((times_reference[0], times_reference2[0],times_reference3[0]))

labels_joined=np.concatenate((labels, labels2,labels3), axis=0)


print(times_reference_joined.shape)
print(labels_joined.shape)


sortingPipeline.set_times_labels(times=times_reference_joined, labels=labels_joined)

sortingPipeline.set_sampling_frequency(sampling_frequency=Fs)









print('Unit ids = {}'.format(sortingPipeline.get_unit_ids()))
st = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1 = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
w_rs_gt = sw.plot_rasters(sortingPipeline,sampling_frequency=Fs)

plt.show()











#Correctly loading the recordings-loading the json file

import os
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '/home/maitreyi/spikeforest_recordings/recordings/SYNTH_MEAREC_TETRODE/synth_mearec_tetrode_noise10_K10_C4/synth_mearec_tetrode_noise10_K10_C4.json')
with open(file_path, 'r') as fi:
    spec = json.load(fi)


#Step 2-Parsing the jason file for input data
name = spec['name']
print("The name of study is ", name)
print('##########################################################')


studySetName=spec['studySetName']
print("The name of the study set is ", studySetName)


#Just experimenting with the first recording
recordingsArray=spec['recordings']
recordingZero=recordingsArray[0]

#Getting the recording and sorting path
recordingPath=recordingZero['directory']
sortingPath=recordingZero['firingsTrue']

#Getting other input information
sampleRate=recordingZero['sampleRateHz']
numChannels=recordingZero['sampleRateHz']
durationSec=recordingZero['durationSec']
numTrueUnits=recordingZero['numTrueUnits']
spikeSign=recordingZero['spikeSign']


#Downloading the recording objects

ka.set_config(fr='default_readonly')
print(f'Downloading recording: {recordingZero["studyName"]}/{recordingZero["name"]}')
ka.load_file(recordingZero['directory'] + '/raw.mda')
ka.load_file(recordingZero['directory'] + '/params.json')
ka.load_file(recordingZero['directory'] + '/geom.csv')
ka.load_file(recordingZero['directory'] + '/firings_true.mda')

#Attaching the results
recordingZero['results'] = dict()

#Tryting to plot the recordings
recordingInput = AutoRecordingExtractor(dict(path=recordingPath), download=True)
w_ts = sw.plot_timeseries(recordingInput)
w_ts.figure.suptitle("Recording by group")
w_ts.ax.set_ylabel("Channel_ids")






#We will also try to plot the rastor plot for the ground truth
gtOutput=AutoSortingExtractor(sortingPath)
#We need to change the indices of  the ground truth output 
w_rs_gt = sw.plot_rasters(gtOutput,sampling_frequency=sampleRate)

#Splitting up the data 

recordingSplit = se.NumpyRecordingExtractor(timeseries=timeseries, geom=geometry, sampling_frequency=Fs)









#Spike-Sorting
#trying to run MS4 through spike interface
#sorting_MS4 = ss.run_tridesclous(recordingSplit, output_folder='tmp_MS4')



#spykingcircus
with ka.config(fr='default_readonly'):
    #with hither.config(cache='default_readwrite'):
        with hither.config(container='default'):
            result_spyKingCircus = sorters.spykingcircus.run(
                recording_path=recordingPath,
                sorting_out=hither.File()
            )


#Mountainsort
with ka.config(fr='default_readonly'):
    #with hither.config(cache='default_readwrite'):
        with hither.config(container='default'):
            result_MS4 = sorters.mountainsort4.run(
                recording_path=recordingPath,
                sorting_out=hither.File()
            )



#Aggregating the output of the sorters
sorting_MS4=AutoSortingExtractor(result_MS4.outputs.sorting_out._path)
sorting_SP=AutoSortingExtractor(result_spyKingCircus.outputs.sorting_out._path)

#Comparing to ground truth-confusion matrix 
comp_MATLAB = sc.compare_sorter_to_ground_truth(gtOutput, sortingPipeline,sampling_frequency=sampleRate,delta_time=3,match_score=0.5,chance_score=0.1,well_detected_score=0.1,exhaustive_gt=True)
w_comp_MATLAB = sw.plot_confusion_matrix(comp_MATLAB, count_text=True)
plt.show()

comp_MS4 = sc.compare_sorter_to_ground_truth(gtOutput, sorting_MS4,sampling_frequency=sampleRate,delta_time=3,match_score=0.5,chance_score=0.1,well_detected_score=0.1,exhaustive_gt=True)
w_comp_MS4 = sw.plot_confusion_matrix(comp_MS4, count_text=True)
plt.show()


comp_SP = sc.compare_sorter_to_ground_truth(gtOutput, sorting_SP,sampling_frequency=sampleRate,delta_time=3,match_score=0.5,chance_score=0.1,well_detected_score=0.1,exhaustive_gt=True)
w_comp_SP = sw.plot_confusion_matrix(comp_SP, count_text=True)
plt.show()



#Computing some metrics for benchmarking-agreement matrix
sw.plot_agreement_matrix(comp_MATLAB, ordered=True,count_text=True)
perf_MATLAB = comp_MATLAB.get_performance()
plt.show()
print(perf_MATLAB)

#comparing the sorting algos
#We will try to compare all the three sorters
mcmp=sc.compare_multiple_sorters(sorting_list=[sortingPipeline,sorting_MS4,sorting_SP],
                               name_list=['Our','MS4','SP'],verbose=True)
                                 
sw.plot_multicomp_graph(mcmp)
plt.show()


#Pairwise

cmp_MS4_Our=sc.compare_two_sorters(sorting1=sorting_MS4,sorting2=sortingPipeline,sorting1_name='MS4',sorting2_name='Our')
sw.plot_agreement_matrix(cmp_MS4_Our, ordered=True,count_text=True)
plt.show()


cmp_SP_Our=sc.compare_two_sorters(sorting1=sorting_SP,sorting2=sortingPipeline,sorting1_name='SP',sorting2_name='Our')
sw.plot_agreement_matrix(cmp_SP_Our, ordered=True,count_text=True)
plt.show()

















   


























