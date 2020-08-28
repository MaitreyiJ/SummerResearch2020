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
#CREDITS TO ORIGINAL IMLEMENTATION:
#https://towardsdatascience.com/whos-talking-using-k-means-clustering-to-sort-neural-events-in-python-e7a8a76f316
#Forward declaration

def extract_snippets(X, *, reference_frames, snippet_len):
    if isinstance(snippet_len, (tuple, list, np.ndarray)):
        snippet_len_before = snippet_len[0]
        snippet_len_after = snippet_len[1]
    else:
        snippet_len_before = int((snippet_len + 1) / 2)
        snippet_len_after = snippet_len - snippet_len_before

    num_snippets = len(reference_frames)
    num_channels = X.shape[0]
    num_frames = X.shape[1]
    snippet_len_total = snippet_len_before + snippet_len_after
    # snippets = []
    snippets = np.zeros((num_snippets, num_channels, snippet_len_total))
    #TODO extract all waveforms in a chunk
    # pad_first = False
    # pad_last = False
    # pad_samples_first = 0
    # pad_samples_last = 0
    # snippet_idxs = np.array([], dtype=int)
    for i in range(num_snippets):
        snippet_chunk = np.zeros((num_channels, snippet_len_total))
        if (0 <= reference_frames[i]) and (reference_frames[i] < num_frames):
            snippet_range = np.array(
                [int(reference_frames[i]) - snippet_len_before, int(reference_frames[i]) + snippet_len_after])
            snippet_buffer = np.array([0, snippet_len_total])
            # The following handles the out-of-bounds cases
            if snippet_range[0] < 0:
                snippet_buffer[0] -= snippet_range[0]
                snippet_range[0] -= snippet_range[0]
            if snippet_range[1] >= num_frames:
                snippet_buffer[1] -= snippet_range[1] - num_frames
                snippet_range[1] -= snippet_range[1] - num_frames
            snippet_chunk[:, snippet_buffer[0]:snippet_buffer[1]] = X[:, snippet_range[0]:snippet_range[1]]
        snippets[i] = snippet_chunk
    return snippets

def knn_denoise(X, X_reference, *, k, ncomp):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neighbors import NearestNeighbors
    # Xb = X * window

    print('PCA...')
    npca = np.minimum(300, X.shape[0])
    u, s, vh = np.linalg.svd(X)
    features = u[:, 0:npca] * s[0:npca]
    components = vh[0:npca, :]
    # s = s[0:npca]

    print('Nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    features2 = np.zeros(features.shape, dtype=features.dtype)
    for j in range(X.shape[0]):
        print(f'{j+1} of {X.shape[0]}')
        inds0 = np.squeeze(indices[j, :])
        inds0 = inds0[1:]
        # Xbneighbors = Xb[inds0, :]
        f_neighbors = features[inds0, :]
        pls = PLSRegression(n_components=ncomp)
        # pls.fit(Xbneighbors.T, Xb[j, :].T)
        pls.fit(f_neighbors.T, features[j, :].T)
        features2[j, :] = pls.predict(f_neighbors.T).T
        # X2[j, :] = pls.predict(Xbneighbors.T).T
    print(features2.shape)
    print(components.shape)
    X2 = features2 @ components
    return X2


def detect_on_channel(data,*,detect_threshold,detect_interval,detect_sign,margin=0):
    # Adjust the data to accommodate the detect_sign
    # After this adjustment, we only need to look for positive peaks
    if detect_sign<0:
        data=data*(-1)
    elif detect_sign==0:
        data=np.abs(data)
    elif detect_sign>0:
        pass

    data=data.ravel()
        
    #An event at timepoint t is flagged if the following two criteria are met:
    # 1. The value at t is greater than the detection threshold (detect_threshold)
    # 2. The value at t is greater than the value at any other timepoint within plus or minus <detect_interval> samples
    
    # First split the data into segments of size detect_interval (don't worry about timepoints left over, we assume we have padding)
    N=len(data)
    S2=int(np.floor(N/detect_interval))
    N2=S2*detect_interval
    data2=np.reshape(data[0:N2],(S2,detect_interval))
    
    # Find the maximum on each segment (these are the initial candidates)
    max_inds2=np.argmax(data2,axis=1)
    max_inds=max_inds2+detect_interval*np.arange(0,S2)
    max_vals=data[max_inds]
    
    # The following two tests compare the values of the candidates with the values of the neighbor candidates
    # If they are too close together, then discard the one that is smaller by setting its value to -1
    # Actually, this doesn't strictly satisfy the above criteria but it is close
    # TODO: fix the subtlety
    max_vals[ np.where((max_inds[0:-1]>=max_inds[1:]-detect_interval) & (max_vals[0:-1]<max_vals[1:]))[0] ]=-1
    max_vals[1+np.array( np.where((max_inds[1:]<=max_inds[0:-1]+detect_interval) & (max_vals[1:]<=max_vals[0:-1]))[0] )]=-1
    
    # Finally we use only the candidates that satisfy the detect_threshold condition
    times=max_inds[ np.where(max_vals>=detect_threshold)[0] ]
    if margin>0:
        times=times[np.where((times>=margin)&(times<N-margin))[0]]

    return times



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


#A SAMPLE IMPLEMENTAION OF SPIKE SORTING PIPELINE FROM PYTHON-DISCARDED DUE TO POOR RESULTS IN BENCHMARKING-ONLY FOR UNDERSTANDING PURPOSES

#Correctly loading the recordings-loading the json file

import os
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '/Users/Maitreyi/CareerDevelopmentoversumeer/SummerResearch2020/spikeforest2/spikeforest_recordings/recordings/PAIRED_CRCNS_HC1/paired_crcns/paired_crcns.json')
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



gtOutput=AutoSortingExtractor(sortingPath)
#Only getting a part of the recording
#gtOutput.add_epoch(epoch_name="first_half", start_frame=0, end_frame=recordingInput.get_num_frames()/2) #set

#subsorting = gtOutput.get_epoch("first_half")
#w_rs_gt = sw.plot_rasters(gtOutput,sampling_frequency=sampleRate)

#w_wf_gt = sw.plot_unit_waveforms(recordingInput,gtOutput, max_spikes_per_unit=100)

#We will also try to plot the rastor plot for the ground truth



#We will be trying to get a sub set of the recording
#recording4 = se.SubRecordingExtractor(parent_recording=recordingInput, channel_ids=[2, 3, 4, 5])
#Plotting a segment of recording
#w_ts = sw.plot_timeseries(recording4)
w_ts.figure.suptitle("Recording by group")
w_ts.ax.set_ylabel("Channel_ids")



#Spike Detection

#Code for filtering-Bandpass filter 

recobj = dict(
    raw=recordingZero['directory'] + '/raw.mda',
    params=ka.load_object(recordingZero['directory'] + '/params.json'),
    geom=np.genfromtxt(ka.load_file(recordingZero['directory'] + '/geom.csv'), delimiter=',').tolist()
)


# bandpass filter

rx = AutoRecordingExtractor(recobj)#recobj is our recording data(raw voltage waveforms)
rx2 = st.preprocessing.bandpass_filter(recording=rx, freq_min=300, freq_max=3000, freq_wid=1000)
#rx3=st.preprocessing.bandpass_filter(recording=recording4,freq_min=300, freq_max=3000, freq_wid=1000)
#Just a few segments of the recording

detect_threshold = 5  #As obtained in literature 
detect_interval = 200
detect_interval_reference = 10
detect_sign = -1
num_events = 1000
snippet_len = (200, 200)
window_frac = 0.3
num_passes = 20
npca = 100
max_t = 30000 * 100
k = 20
ncomp = 4
    
R = AutoRecordingExtractor(rx2)

X = R.get_traces() #getting the snippets of data
    
sig = X.copy()
if detect_sign < 0:
    sig = -sig
elif detect_sign == 0:
    sig = np.abs(sig)
sig = np.max(sig, axis=0)
noise_level = np.median(np.abs(sig)) / 0.6745  # median absolute deviation (MAD)
times_reference = detect_on_channel(sig, detect_threshold=noise_level*detect_threshold, detect_interval=detect_interval_reference, detect_sign=1, margin=1000)
times_reference = times_reference[times_reference <= max_t]
print(f'The size of the array is = {times_reference.size}')
print(f'Num. reference events = {len(times_reference)}')
print(f'The dimensions of the array is  = {times_reference.ndim}')
print(f'The shape of the array is  = {times_reference.shape}')


#print(f'The first time stamp of the array is  = {times_reference[0]}')

snippets_reference = extract_snippets(X, reference_frames=times_reference, snippet_len=snippet_len)

tt = np.linspace(-1, 1, snippets_reference.shape[2])
window0 = np.exp(-tt**2/(2*window_frac**2))
for j in range(snippets_reference.shape[0]):
    for m in range(snippets_reference.shape[1]):
        snippets_reference[j, m, :] = snippets_reference[j, m, :] * window0
A_snippets_reference = snippets_reference.reshape(snippets_reference.shape[0], snippets_reference.shape[1] * snippets_reference.shape[2])

print(f'The size of the snippets array is = {A_snippets_reference.size}')
print(f'Num. reference events in spnippets array = {len(A_snippets_reference)}')
print(f'The dimensions of the  snippets array is  = {A_snippets_reference.ndim}')
print(f'The shape of the  snippets array is  = {A_snippets_reference.shape}')

print(f'The size of the snippets array is = {snippets_reference.size}')
print(f'Num. reference events in spnippets array = {len(snippets_reference)}')
print(f'The dimensions of the  snippets array is  = {snippets_reference.ndim}')
print(f'The shape of the  snippets array is  = {snippets_reference.shape}')


print('PCA...')
import sklearn as sk
from sklearn.decomposition import PCA

# Apply min-max scaling
scaler= sk.preprocessing.MinMaxScaler()
dataset_scaled = scaler.fit_transform(A_snippets_reference)

# Do PCA
pca = PCA(n_components=12)
pca_result = pca.fit_transform(dataset_scaled)

# Plot the 1st principal component aginst the 2nd and use the 3rd for color
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
ax.set_xlabel('1st principal component', fontsize=20)
ax.set_ylabel('2nd principal component', fontsize=20)
ax.set_title('first 3 principal components', fontsize=23)

fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


#How many clusters do we need



num_clus = 4
cluster, centers, distance = k_means(pca_result, num_clus)

# Plot the result
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=cluster)
ax[0].set_xlabel('1st principal component', fontsize=20)
ax[0].set_ylabel('2nd principal component', fontsize=20)
ax[0].set_title('clustered data', fontsize=23)

time = np.linspace(0, A_snippets_reference.shape[1]/sampleRate, A_snippets_reference.shape[1])*1000
for i in range(num_clus):
    cluster_mean = A_snippets_reference[cluster==i, :].mean(axis=0)
    cluster_std = A_snippets_reference[cluster==i, :].std(axis=0)

    ax[1].plot(time, cluster_mean, label='Cluster {}'.format(i))
    ax[1].fill_between(time, cluster_mean-cluster_std, cluster_mean+cluster_std, alpha=0.15)

ax[1].set_title('average waveforms', fontsize=23)
ax[1].set_xlim([0, time[-1]])
ax[1].set_xlabel('time [ms]', fontsize=20)
ax[1].set_ylabel('amplitude [uV]', fontsize=20)

plt.legend()
plt.show()


print(f'The size of the snippets array is = {snippets_reference.size}')
print(f'Num. reference events in spnippets array = {len(snippets_reference)}')
print(f'The dimensions of the  snippets array is  = {snippets_reference.ndim}')
print(f'The shape of the  snippets array is  = {snippets_reference.shape}')



#so we are done with the pipeline, now we need to compare this algorithm to ground truth
#We will be comparing the features of each of our waveforms(waveform) with the mean_cluster forms
#for spike classification
#time = np.linspace(0, wave_form.shape[1]/sampleRate, wave_form.shape[1])*1000
#cluster_mean = wave_form[cluster==0, :].mean(axis=0)

#Adding the code for comparision of spike sorting data with sorting extractor

sortingPipeline = se.NumpySortingExtractor()
sortingPipeline.set_times_labels(times=times_reference, labels=cluster)
sortingPipeline.set_sampling_frequency(sampling_frequency=sampleRate)

print('Unit ids = {}'.format(sortingPipeline.get_unit_ids()))
st = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1 = sortingPipeline.get_unit_spike_train(unit_id=1)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))

#We are also going to be setting up the unit spike features associated with each waveform
ID1_features = A_snippets_reference[cluster==1, :]
sortingPipeline.set_unit_spike_features(unit_id=1, feature_name='unitId1',
                                 value=ID1_features)
print("Spike feature names: " + str(sortingPipeline.get_unit_spike_feature_names(unit_id=1)))

#Comparing sorter with ground truth
cmp_gt_SP = sc.compare_sorter_to_ground_truth(gtOutput, sortingPipeline, exhaustive_gt=True)
sw.plot_agreement_matrix(cmp_gt_SP, ordered=True)

#Some more comparision metrics
perf = cmp_gt_SP.get_performance()
#print('well_detected', cmp_gt_SP.get_well_detected_units(well_detected_score=0))
print (perf)
#We will try to get the SNR and firing rates

#firing_rates = st.validation.compute_firing_rates(sortingPipeline, duration_in_frames=recordingInput.get_num_frames())


#Raster plots

w_rs_gt = sw.plot_rasters(sortingPipeline,sampling_frequency=sampleRate)

w_wf_gt = sw.plot_unit_waveforms(recordingInput,sortingPipeline, max_spikes_per_unit=100)

