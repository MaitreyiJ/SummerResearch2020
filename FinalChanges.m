% Spike sorting with an EDM-based Autoencoder
%addpath(genpath('./LIB')); 

%filename = '../data/data0012.mcd';
%lib_path = './LIB/MCS/Matlab-Import-Filter\Matlab_Interface\nsMCDLibrary64.dll';

%DEBUG = 0;
%if (DEBUG)
    %[metadata, mcs_data] = get_mcs_mat(lib_path, filename);
%    load ../data/mcs_data.mat
%end

mcs_data = load('simulation_1.mat');
mcs_data=mcs_data.data;
mcs_data=mcs_data(1,1:35000)
mcs_data = mcs_data./1e-6; % convert to microvolts
    
Fs = 24000;
ch_IDs = [1];

n_samp = size(mcs_data,2);
t_samp = 1/Fs;
t = 0:t_samp:(t_samp*n_samp)-t_samp;

ap_width_t_max = 6e-3; % max width of action potential
ap_width_s_max = ap_width_t_max/t_samp;

ap_width_t_typ = 3e-3; % typical width of action potential
ap_width_s_typ = ap_width_t_typ/t_samp;


%% Select a channel to analyze and extract all channels
% ap_threshold based spike sorting
PLOT_ALL_CHANNEL_SPIKES = 0;
if (PLOT_ALL_CHANNEL_SPIKES)
    spikeTimes = {};
    for c_num = 1:1;
        fprintf('%i\n',c_num);        
        spikes = [];

        %Filter data
        channel_data = mcs_data(c_num,:);
        filt_ch_data = aps_filter(channel_data, Fs);
        max_peak = max(filt_ch_data);
        min_peak = min(filt_ch_data);        
        
        spike_posth_idx = aps_get_thresh_idxs(filt_ch_data, 1, 3.5);
        spike_negth_idx = aps_get_thresh_idxs(filt_ch_data, 0, 3.5);
        spike_idxs = [spike_posth_idx spike_negth_idx];
        

        search_window = ap_width_s_typ;
        [spike_min_idxs, spike_max_idxs] = aps_get_spike_idxs(filt_ch_data, spike_idxs, search_window);

        % Get time series window of spikes for plotting
        [spike_aligned_min] = aps_get_spikes(filt_ch_data, spike_min_idxs, search_window);
        [spike_aligned_max] = aps_get_spikes(filt_ch_data, spike_max_idxs, search_window);
        
        ch_spikes = zeros(1,length(filt_ch_data));
        ch_spikes(spike_min_idxs) = min_peak;
        ch_spikes(spike_max_idxs) = max_peak;
        
        fig = figure(20);
        clf(fig);    
        subplot(3,1,1);        
        plot(filt_ch_data,'k');
        hold on;
        plot(ch_spikes,class1_color);
        axis tight;
        subplot(3,1,2);
        plot(spike_aligned_min');
        hold on;
        axis tight;
        subplot(3,1,3);        
        plot(spike_aligned_max');
        hold on;
        axis tight;
        
        pause;
%         fig = figure(7);
%         clf(fig);
%         plot(filt_ch_data,'k');
%         hold on;
%         plot(ch_spikes,class1_color);
%         pause;
%         spike_idx = spike_max_idxs(i);
%         subplot(3,1,1);
%         ap_trace(spike_idx) = ap_thresh;        
    end
end



%% Get all spikes in an individual channel and align
EXTRACT_ALIGN_INDIV = 1;
if (EXTRACT_ALIGN_INDIV)
%     c_num = 16;
    c_num = 1; %good
    channel_data = mcs_data(c_num,:);
    % Preprocess data by filtering for noise and filter out LFPs
    filt_ch_data = aps_filter(channel_data, Fs);
    max_peak = max(filt_ch_data);
    min_peak = min(filt_ch_data);
        
    % ap_threshold data and get spike indices
%     spike_posth_idx = aps_get_thresh_idxs(filt_ch_data, 1, 3.5);
    spike_negth_idx = aps_get_thresh_idxs(filt_ch_data, 0, 2.8);%changed the threhold computation
    noise_level_median = mad(filt_ch_data,1)/ 0.6745 ; % median absolute deviation (MAD)
    ap_thresh_median = 3.5*noise_level_median;
    
    %also the mean absolute deviation 
    noise_level_mean=mad(filt_ch_data)/0.6745;
    ap_thresh_mean = 3*noise_level_mean;

%     spike_idxs = [spike_posth_idx spike_negth_idx];
    spike_negth_idx = spike_negth_idx(2:end-1); % remove some at start of file due to EDM
    spike_idxs = spike_negth_idx;
    n_spikes = length(spike_idxs);
    
    search_window = ap_width_s_typ;
    [spike_min_idxs, spike_max_idxs] = aps_get_spike_idxs(filt_ch_data, spike_idxs, search_window);
    
    % Get time series window of spikes for plotting
    [spike_mins] = aps_get_spikes(filt_ch_data, spike_min_idxs, search_window);
    [spike_maxs] = aps_get_spikes(filt_ch_data, spike_max_idxs, search_window);
    
    % Build preliminary classes based on a manually chosen threshold 
    % (only works when spikes can be really easily separated)
    max_vals = max(spike_mins');
    min_vals = min(spike_mins');
    pos_thresh = 10;
    neg_thresh = -13;
    
    class1_pth_idxs = find(max_vals > pos_thresh);
    class1_nth_idxs = find(min_vals < neg_thresh);
%     class1_idxs = class1_pth_idxs;
%     class1_idxs = intersect(class1_pth_idxs, class1_nth_idxs);
    class1_idxs = intersect(class1_pth_idxs, class1_nth_idxs);
    
    class2_th1_idxs = find(min_vals > -18);
    class2_th2_idxs = find(min_vals > -12);
    class2_idxs = setdiff(class2_th1_idxs , class2_th2_idxs);
    %class2_idxs = class2_idxs(find(class2_idxs~=89)); % manually remove 89
    %class2_idxs = [class2_idxs 29]; % manually add 29
    
    
    % Remove class 2 from class 1
	class1_idxs = setdiff(class1_idxs, class2_idxs);
    
    % Class 3 is the remainder
    class3_idxs = setdiff(1:n_spikes,[class1_idxs class2_idxs]);
    
    class1_color = 'r';
    class2_color = 'g';
    class3_color = 'b';
    
    fprintf('Class1: %i\n',length(class1_idxs));
    fprintf('Class2: %i\n',length(class2_idxs));    
    fprintf('Class3: %i\n',length(class3_idxs)); 
    
    % plotting the spikes ground truth and detected
    plot(filt_ch_data);
    yline(noise_level_mean,'y','LineWidth',1);
    yline(noise_level_median,'g','LineWidth',1);
    
    hold on;
     for j=1:length(spike_max_idxs)
         scatter(spike_max_idxs(j),1e+6,"red","*");
         scatter(spike_min_idxs(j),1e+6,"magenta","*");
         
     end     
    groundTruth=load('ground_truth.mat');
    spike_index_gt=groundTruth.spike_first_sample;
    array= spike_index_gt(1);
    new=array(1,:)
    out = cat(1,new{:});
    for i=1:27
         scatter(out(i),1e+6,"black","o");
    
    
    end
end

    %% Pass channel through EDM bank for feature extraction
    alphas = [3,4,5,6,7,8,9,10];
    n_alphas = length(alphas);
    edm_output = ap_edm_bipolar_extract(filt_ch_data, alphas);
    edm_output_zscore = zscore(edm_output,[],2);
    n_features = size(edm_output,1); %No of features is an appropriate measure here as compared to ground truth!!
    
    edm_sample_offset = 20; %ceil(ap_width_s_typ/4);
    spike_edm_vectors = edm_output_zscore(:,spike_min_idxs+edm_sample_offset);
    
    PLOT_EDM_VECTORS=0;
    if(PLOT_EDM_VECTORS)
        fig = figure(114);
        clf(fig);       
        subplot(2,1,1);
        plt3 = plot(spike_mins','k');  
        axis tight;
        subplot(2,1,2);
        sort_dimension = 4;
        [remap,srt_idxs] = sort(spike_edm_vectors(sort_dimension,:));
        imagesc(spike_edm_vectors(:,srt_idxs));
    end
    
   
    

    
    %% Train Autoencoder    
    hiddenSize = 2;
    tic;
    autoenc = trainAutoencoder(spike_edm_vectors,hiddenSize,...
        'EncoderTransferFunction','logsig',...
        'DecoderTransferFunction','purelin',...
        'SparsityRegularization',0,...
        'MaxEpochs',2000,...
        'L2WeightRegularization',0,...
        'ScaleData',true,... % not sure about this option
        'SparsityProportion',0.1); %% this really depends on the number of training samples from each
    toc;
%     'ShowProgressWindow',false
% 'EncoderTransferFunction',satlin'logsig',...

    % Works
%     tic;
%     autoenc = trainAutoencoder(spike_edm_vectors_sorted,hiddenSize,...
%         'EncoderTransferFunction','satlin',...
%         'DecoderTransferFunction','purelin',...
%         'SparsityRegularization',0,...
%         'MaxEpochs',2000,...
%         'L2WeightRegularization',0,...
%         'ScaleData',true,... % not sure about this option
%         'SparsityProportion',0.1); %% this really depends on the number of training samples from each
%     toc;
    tic;
    spike_edm_vectors_enc = encode(autoenc,spike_edm_vectors);
    toc;
    
    tic;
    spike_edm_vectors_dec = decode(autoenc,spike_edm_vectors_enc);
    toc;    

  
    % Plot autoenc output
%     fig = figure(27);
%     clf(fig);     
% %     scatter(spike_edm_vectors_enc(1,:),spike_edm_vectors_enc(2,:));
% %     hold on
% %     scatter(spike_edm_vectors_enc(1,class1_idxs),spike_edm_vectors_enc(2,class1_idxs));
%     scatter3(   spike_edm_vectors_enc(1,:),...
%                 spike_edm_vectors_enc(2,:),...
%                 spike_edm_vectors_enc(3,:));    
%     hold on;
%     scatter3(   spike_edm_vectors_enc(1,class1_idxs),...
%                 spike_edm_vectors_enc(2,class1_idxs),...
%                 spike_edm_vectors_enc(3,class1_idxs),class1_color);
%     axis tight;
    
    %% Run K-Means
    k=n_features;
    X=spike_edm_vectors_enc';
%     rng(1);
    [idx,C] = kmeans(X,k);
%     idx = kmeans(spike_edm_vectors_enc,k);
    %idx now has 16 classes as per ground truth !!!!
    %plotting the spikes with their ids and ground truth 
     
    %Unit id to color association 
    colors=[1 0 0; 0 1 0	; 0 0 1 ; 0.3 0.2 0.1; 1 1 0; 1 0 1	; 0 1 1	; 1 1 1	; 0 0 0; 0 0.4470 0.7410;  0.8500 0.3250 0.0980 ; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560 ; 0.4660 0.6740 0.1880;0.3010 0.7450 0.9330;0.6350 0.0780 0.1840];
    

    
   
    fig=figure(115);
    % plotting the spikes ground truth and detected
    plot(filt_ch_data);
   
    hold on;
    for j=1:length(spike_max_idxs)
      
         scatter(spike_min_idxs(j),1e+6,'MarkerFaceColor',colors(idx(j,1),:),'Marker','o');
         
         
     end     
    groundTruth=load('ground_truth.mat');
    spike_index_class=groundTruth.spike_classes;
    array_gt_class=spike_index_class(1);
    new_class=array_gt_class(1,:);
    out = cat(1,new{:});
    out_class=cat(1,new_class{:});
    for i=1:27
        
        if(out_class(1,i)~=0) 
           s=scatter(out(i),0.5e+6,'Marker','d');

           s.MarkerFaceColor=colors(out_class(1,i),:);
        end
       
    
    
    end
    
  %attempting to plot the features of ground truth and spike edm vectors
  fig=figure(116);
  spike_waveforms=groundTruth.su_waveforms;
  array_gt_waveforms=spike_waveforms(1);
  new_waveforms=array_gt_waveforms(1,:);
  out_waveforms=cat(1,new_waveforms{:});
  for k=1:10
      plot(out_waveforms(k,:))
  end
  
  
  
    
    
    
    
    
    
    