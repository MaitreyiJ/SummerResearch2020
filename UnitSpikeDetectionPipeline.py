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

def k_means(data, num_clus=3, steps=200):

    # Convert data to Numpy array
    cluster_data = np.array(data)

    # Initialize by randomly selecting points in the data
    center_init = np.random.randint(0, cluster_data.shape[0], num_clus)

    # Create a list with center coordinates
    center_init = cluster_data[center_init, :]

    # Repeat clustering  x times
    for _ in range(steps):

        # Calculate distance of each data point to cluster center
        distance = []
        for center in center_init:
            tmp_distance = np.sqrt(np.sum((cluster_data - center)**2, axis=1))

            # Adding smalle random noise to the data to avoid matching distances to centroids
            tmp_distance = tmp_distance + np.abs(np.random.randn(len(tmp_distance))*0.0001)
            distance.append(tmp_distance)

        # Assign each point to cluster based on minimum distance
        _, cluster = np.where(np.transpose(distance == np.min(distance, axis=0)))

        # Find center of mass for each cluster
        center_init = []
        for i in range(num_clus):    
            center_init.append(cluster_data[cluster == i, :].mean(axis=0).tolist())

    return cluster, center_init, distance

def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):

    # Calculate threshold based on MDA-combining two methods 
    thresh = np.mean(np.abs(data)) *tf

    # Find positions wherere the threshold is crossed
    pos = np.where(data > thresh)[0]
    pos = pos[pos > spike_window]

    # Extract potential spikes and align them to the maximum
    spike_samp = []
    wave_form = np.empty([1, spike_window*2])
    for i in pos:
        if i < data.shape[0] - (spike_window+1):
            # Data from position where threshold is crossed to end of window
            tmp_waveform = data[i:i+spike_window*2]

            # Check if data in window is below upper threshold (artifact rejection)
            if np.max(tmp_waveform) < max_thresh:
                # Find sample with maximum data point in window
                tmp_samp = np.argmax(tmp_waveform) +i

                # Re-center window on maximum sample and shift it by offset
                tmp_waveform = data[tmp_samp-(spike_window-offset):tmp_samp+(spike_window+offset)]

                # Append data
                spike_samp = np.append(spike_samp, tmp_samp)
                wave_form = np.append(wave_form, tmp_waveform.reshape(1, spike_window*2), axis=0)

   

    return spike_samp, wave_form






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































#Spike Detection

#Code for filtering-Bandpass filter 

recobj = dict(
    raw=recordingZero['directory'] + '/raw.mda',
    params=ka.load_object(recordingZero['directory'] + '/params.json'),
    geom=np.genfromtxt(ka.load_file(recordingZero['directory'] + '/geom.csv'), delimiter=',').tolist()
)


# bandpass filter

rx = AutoRecordingExtractor(recobj)
rx2 = st.preprocessing.bandpass_filter(recording=rx, freq_min=300, freq_max=3000, freq_wid=1000)


    
R = AutoRecordingExtractor(rx2)

X = R.get_traces()


#Getting some information about the traces









    
sig = X.copy()




sig=sig.ravel()
spike_samp, wave_form = get_spikes(sig, spike_window=50, tf=8, offset=20)



print(f'The size of the array is = {wave_form.size}')
print(f'Num. reference events = {len(wave_form)}')
print(f'The dimensions of the array is  = {wave_form.ndim}')
print(f'The shape of the array is  = {wave_form.shape}')




print(f'The size of the spike sample array is = {spike_samp.size}')
print(f'Num. reference events in spike sample array is  = {len(spike_samp)}')
print(f'The dimensions of the  spike sample array is  = {spike_samp.ndim}')
print(f'The shape of the  spike sample array is  = {spike_samp.shape}')




fig, ax = plt.subplots(figsize=(15, 5))

for i in range(100):
    spike = np.random.randint(0, wave_form.shape[0])
    ax.plot(wave_form[spike, :])

ax.set_xlim([0, 90])
ax.set_xlabel('# sample', fontsize=20)
ax.set_ylabel('amplitude [uV]', fontsize=20)
ax.set_title('spike waveforms', fontsize=23)
plt.show()


#Here we will be performing PCA for feature extraction and spike alignment
import sklearn as sk
from sklearn.decomposition import PCA

# Apply min-max scaling
scaler= sk.preprocessing.MinMaxScaler()
dataset_scaled = scaler.fit_transform(wave_form)

# Do PCA
pca = PCA(n_components=12)
pca_result = pca.fit_transform(dataset_scaled)
#Attempting to get some information about pca_result
print(f'The size of the pca array is = {pca_result.size}')

print(f'The dimensions of the  pca array is  = {pca_result.ndim}')
print(f'The shape of the  pca result array is  = {pca_result.shape}')



#We will now compare the PCS scores of the ground truth with this sorter output
pca_scores = st.postprocessing.compute_unit_pca_scores(recordingInput, gtOutput, n_comp=12, verbose=False)

for pc in pca_scores:
    print(pc.shape)

fig, ax = plt.subplots() #PCA scores of ground truth
ax.plot(pca_scores[0][:, 0], pca_scores[0][:, 1], 'r*')
ax.plot(pca_scores[2][:, 0], pca_scores[2][:, 1], 'b*')
plt.show()

num_clus = 6
cluster, centers, distance = k_means(pca_result, num_clus)

# Plot the result
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=cluster)
ax[0].set_xlabel('1st principal component', fontsize=20)
ax[0].set_ylabel('2nd principal component', fontsize=20)
ax[0].set_title('clustered data', fontsize=23)

time = np.linspace(0, wave_form.shape[1]/sampleRate, wave_form.shape[1])*1000
for i in range(num_clus):
    cluster_mean = wave_form[cluster==i, :].mean(axis=0)
    cluster_std = wave_form[cluster==i, :].std(axis=0)

    ax[1].plot(time, cluster_mean, label='Cluster {}'.format(i))
    ax[1].fill_between(time, cluster_mean-cluster_std, cluster_mean+cluster_std, alpha=0.15)

ax[1].set_title('average waveforms', fontsize=23)
ax[1].set_xlim([0, time[-1]])
ax[1].set_xlabel('time [ms]', fontsize=20)
ax[1].set_ylabel('amplitude [uV]', fontsize=20)

plt.legend()
plt.show()




    

