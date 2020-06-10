#!/usr/bin/env python

from spikeforest2 import sorters
from spikeforest2 import processing
import hither_sf as hither
import kachery as ka
import json
import numpy as np
from spikeinterface import widgets as sw
from typing import List, Union, Dict
from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor
import spikeinterface.comparison as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





from scipy.signal import butter, lfilter



import spikeinterface.toolkit as st


#Correctly loading the recordings-loading the json file

import os
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '/Users/Maitreyi/CareerDevelopmentoversumeer/SummerResearch2020/spikeforest2/spikeforest_recordings/recordings/SYNTH_MEAREC_TETRODE/synth_mearec_tetrode_noise10_K10_C4/synth_mearec_tetrode_noise10_K10_C4.json')
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
w_rs_gt = sw.plot_rasters(gtOutput,sampling_frequency=sampleRate)

#Also the unit spikes for ground truth
w_wf_gt = sw.plot_unit_waveforms(recordingInput,gtOutput, max_spikes_per_unit=100)


# Summarize recordings
recordingZero['results']['computed-info'] = processing.compute_recording_info.run(
                _label=f'compute-recording-info:{recordingZero["studyName"]}/{recordingZero["name"]}',
                recording_path=recordingPath,
                json_out=hither.File()
            )
recordingZero['results']['true-units-info'] = processing.compute_units_info.run(
                _label=f'compute-units-info:{recordingZero["studyName"]}/{recordingZero["name"]}',
                recording_path=recordingPath,
                sorting_path=sortingPath,
                json_out=hither.File()
            )


#Spike-Sorting


with ka.config(fr='default_readonly'):
    #with hither.config(cache='default_readwrite'):
        with hither.config(container='default'):
            sorting_result = sorters.spykingcircus.run(
                _label=f'algorithm:{recordingZero["studyName"]}/{recordingZero["name"]}',
                recording_path=recordingPath,
                sorting_out=hither.File()
            )


recordingZero['results']['sorting-SpyKigCircus'] = sorting_result
recordingZero['results']['comparison-with-truth-spyKingCircus']= processing.compare_with_truth.run(
                        _label=f'comparison-with-truth -algorithm:{recordingZero["studyName"]}/{recordingZero["name"]}',
                        sorting_path=sorting_result.outputs.sorting_out,
                        sorting_true_path=sortingPath,
                        json_out=hither.File()
                    )
recordingZero['results']['units-info-spyKingCircus'] = processing.compute_units_info.run(
                        _label=f'units-info-algorithm:{recordingZero["studyName"]}/{recordingZero["name"]}',
                        recording_path=recordingPath,
                        sorting_path=sorting_result.outputs.sorting_out,
                        json_out=hither.File()
                        
                    )


#Trying to slot in our algorithm for benchmarking
#Threshold application and spike detection


#Step 1-Filtering the data






















#Processing the results for someting meaningful-Rastor Plots

sortingOutput = AutoSortingExtractor(sorting_result.outputs.sorting_out._path)#So now we successfully got the object
w_rs = sw.plot_rasters(sortingOutput,sampling_frequency=sampleRate)






#Unit waveforms-of spike sorting results 
w_wf = sw.plot_unit_waveforms(recordingInput, sortingOutput, max_spikes_per_unit=100)


   
   
#Assembling all the results
print('')
print('=======================================================')
print('Assembling results...')
print(f'Assembling recording: {recordingZero["studyName"]}/{recordingZero["name"]}')
recordingZero['summary'] = dict(
            plots=dict(),
            computed_info=ka.load_object(recordingZero['results']['computed-info'].outputs.json_out._path),
            true_units_info=ka.store_file(recordingZero['results']['true-units-info'].outputs.json_out._path)
        )


#This is using spike forest comaprision tools
print(f'Assembling sorting: SpyKingCircus {recordingZero["studyName"]}/{recordingZero["name"]}')
sorting_result = recordingZero['results']['sorting-SpyKigCircus']
comparison_result = recordingZero['results']['comparison-with-truth-spyKingCircus']
units_info_result = recordingZero['results']['units-info-spyKingCircus']


#Unable to convert the spike forest comparsion 

#Directly using the spike interface comparision widgets-plotting the confusion matrix
comp_spyKingCircus = sc.compare_sorter_to_ground_truth(gtOutput, sortingOutput,sampling_frequency=sampleRate,exhaustive_gt=True)
w_comp_spyKingCircus = sw.plot_confusion_matrix(comp_spyKingCircus, count_text=False)



######
#Computing some metrics for benchmarking-agreement matrix
sw.plot_agreement_matrix(comp_spyKingCircus, ordered=True)
perf_spyKingCircus = comp_spyKingCircus.get_performance()
print(perf_spyKingCircus)



#Querrying for the well detected units
print(comp_spyKingCircus.get_well_detected_units())
print(comp_spyKingCircus.get_false_positive_units())



#Now we will be plotting quality metrics-SNR against accuracy


snrs = st.validation.compute_snrs(gtOutput, recordingInput, save_as_property=True)

w_perf = sw.plot_sorting_performance(comp_spyKingCircus, property_name='snr', metric='accuracy')


