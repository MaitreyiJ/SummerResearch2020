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
    
    %% Plot different classes of spikes, based solely on threshold    
    fig = figure(34);
    clf(fig);     
    subplot(2,1,1); 
    hold on
    plot(spike_mins(class1_idxs,:)',class1_color,'LineWidth',1);
    plot(spike_mins(class2_idxs,:)',class2_color,'LineWidth',1);
    plot(spike_mins(class3_idxs,:)',class3_color,'LineWidth',1);
    axis([-inf inf min_peak max_peak]);
    axis tight;
%     plot([edm_sample_point edm_sample_point],[min_peak max_peak],'k')   
    subplot(2,1,2); 
    sorted_class_edms = [spike_edm_vectors(:,class1_idxs), spike_edm_vectors(:,class2_idxs), spike_edm_vectors(:,class3_idxs)];
    imagesc(sorted_class_edms);    
    
    
    %% Try to find the optimal EDM sampling point
    sum_vectors_c1 = [];
    sum_vectors_c2 = [];
    sum_vectors_c3 = [];
    SWEEP_OFFSETS=1;
    if(SWEEP_OFFSETS)
        edm_sample_point_sweep_range = -ceil(search_window/2)+1:1:ceil(search_window/2);
        t_sec = (edm_sample_point_sweep_range/Fs)/1e-3;
        for edm_sample_offset = edm_sample_point_sweep_range
            fprintf('%i\n',edm_sample_offset);  
            fig = figure(112);
            clf(fig);
            edm_sample_point = ceil(search_window/2)+edm_sample_offset;
%             spike_edm_vectors_zscore = zscore(spike_edm_vectors,[],2);
            sum_vectors_c1(edm_sample_point) = sum(sum(abs(spike_edm_vectors(:,class1_idxs))));
            sum_vectors_c2(edm_sample_point) = sum(sum(abs(spike_edm_vectors(:,class2_idxs))));
            sum_vectors_c3(edm_sample_point) = sum(sum(abs(spike_edm_vectors(:,class3_idxs))));
            
            subplot(2,1,1); 
            hold on
            plot(spike_mins(class1_idxs,:)',class1_color,'LineWidth',1);
            plot(spike_mins(class2_idxs,:)',class2_color,'LineWidth',1);
            plot(spike_mins(class3_idxs,:)',class3_color,'LineWidth',1);
            plot(spike_mins(1,:),'k','LineWidth',2);
            axis([-inf inf min_peak max_peak]);
            axis tight;
            plot([edm_sample_point edm_sample_point],[min_peak max_peak],'k')   
            subplot(2,1,2); 
            spike_edm_vectors = edm_output_zscore(:,spike_min_idxs+edm_sample_offset);
            imagesc(spike_edm_vectors);
%             pause;
        end
                        
        fig = figure(116);
        clf(fig);
        
        ax(1) = subplot(2,1,1);
        hold on;
        for j=1:length(class1_idxs) % need stupid loop to set opacity of line plot)
            plt = plot(t_sec, spike_mins(class1_idxs(j),:)',class1_color,'LineWidth',0.5);
            plt.Color(4) = 0.1;       
        end
                
        for j=1:length(class2_idxs) % need stupid loop to set opacity of line plot)
            plt = plot(t_sec, spike_mins(class2_idxs(j),:)',class2_color,'LineWidth',0.5);
            plt.Color(4) = 0.1;       
        end
                 
        for j=1:length(class3_idxs) % need stupid loop to set opacity of line plot)
            plt = plot(t_sec, spike_mins(class3_idxs(j),:)',class3_color,'LineWidth',0.5);
            plt.Color(4) = 0.1;       
        end
                
        plt = plot(t_sec, mean(spike_mins(class1_idxs,:))',class1_color,'LineWidth',2);
        plt = plot(t_sec, mean(spike_mins(class2_idxs,:))',class2_color,'LineWidth',2);  
        plt = plot(t_sec, mean(spike_mins(class3_idxs,:))',class3_color,'LineWidth',2);           
        
        align_point = ceil(search_window/2);
        align_point = (align_point/Fs)/1e-3;
        plot([align_point align_point],[min_peak max_peak],'k') 
        axis tight
        
        ax(2) = subplot(2,1,2); 
        norm_response_c1 = sum_vectors_c1/max(sum_vectors_c1);
        norm_response_c2 = sum_vectors_c2/max(sum_vectors_c2);
        norm_response_c3 = sum_vectors_c3/max(sum_vectors_c3);
        sum_resp = (norm_response_c1 + norm_response_c2 + norm_response_c3);
        norm_resp = sum_resp/max(sum_resp);
        [max_norm, optimal_edm_sample_point] = max(norm_resp);
        fprintf('Optimal EDM Sampling point: %i\n',optimal_edm_sample_point); 
        
        % Plot optimal edm point sweep graph
        hold on;
        plt = plot(t_sec, norm_response_c1,class1_color,'LineWidth',1.5);
        plt.Color(4) = 0.4; 
        plt = plot(t_sec, norm_response_c2,class2_color,'LineWidth',1.5);
        plt.Color(4) = 0.4; 
        plt = plot(t_sec, norm_response_c3,class3_color,'LineWidth',1.5);
        plt.Color(4) = 0.4; 
        plt = plot((edm_sample_point_sweep_range/Fs)/1e-3 , norm_resp,'k','LineWidth',2);
        plt.Color(4) = 0.7;
        plot([t_sec(optimal_edm_sample_point) t_sec(optimal_edm_sample_point)],[0 1],'-k')  
        linkaxes(ax,'x');
        axis([-1 1 0 1.05]);
        title('Sampling Point vs. Normalized EDM Bank Variance');
        xlabel('Sampling Point (Relative to Spike Min)')
        ylabel('Normalized EDM Bank Variance')
    end
    
    %% Go through all features to see if any pop out as informative to sort by
    edm_sample_point = (search_window/2)+edm_sample_offset;
    
    PLOT_SORTED_BY_FEATURES = 0;
    if (PLOT_SORTED_BY_FEATURES)
        for sort_dimension = 1:n_features %1:n_features % n_alphas
            % Sort EDMs by 1 dimension
            fprintf('%i\n',sort_dimension);  
            [remap,srt_idxs] = sort(spike_edm_vectors(sort_dimension,:));
            spike_edm_vectors_sorted=spike_edm_vectors(:,srt_idxs);

            c = ismember(srt_idxs, class1_idxs);
            class1_idxs_remap = find(c);

            c = ismember(srt_idxs, class2_idxs);
            class2_idxs_remap = find(c);

            % Plot 
            fig = figure(31);
            clf(fig);     
            subplot(2,1,1);
            plt3 = plot(spike_mins',class2_color);
    %         plt4.Color(4) = 0.3;axis([-inf edm_sample_point min_peak max_peak]); 
            hold on;
%             axis([-inf edm_sample_point min_peak max_peak]); 
            for j=1:1:10
                plt1 = plot(spike_mins(srt_idxs(j),:),class3_color,'LineWidth',4);
                plt1.Color(4) = 1/j;
                plt2 = plot(spike_mins(srt_idxs(end-j+1),:),class1_color,'LineWidth',4);
                plt2.Color(4) = 1/j;
            end
            plot([edm_sample_point edm_sample_point],[min_peak max_peak],'k')    
    %         subplot(3,1,2);
    %         plot(spike_maxs');
    %         hold on;
    %         plot(spike_maxs(srt_idxs(1),:),class1_color,'LineWidth',4);
    %         plot(spike_maxs(srt_idxs(end),:),class3_color,'LineWidth',4); 
    %         axis([-inf inf min_peak max_peak]);
        %     plot([edm_sample_point edm_sample_point],[min_peak max_peak],'k')    
            subplot(2,1,2);
            imagesc(spike_edm_vectors_sorted);   
            hold on;

            for i=1:length(class1_idxs_remap)
                plot([class1_idxs_remap(i) class1_idxs_remap(i)],[0 n_features],class1_color)  
            end

            pause;
        end
    end
    
%     for j = 1:1:430
%         subplot(3,1,2);    
%         plot(spike_maxs(index(j),:),class3_color,'LineWidth',2);
%         fprintf('%i\n',j);
%         pause
%     end
    

    
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
    %% Plot
    fig = figure(29);
    clf(fig);     
    subplot(4,1,1);
    imagesc(sorted_class_edms);     
    title('Autoencoder Input');
    subplot(4,1,4);
    imagesc(decode(autoenc,encode(autoenc,sorted_class_edms)));
    title('Reconstructed Output');
%     spike_edm_vectors_sorted = flipud(spike_edm_vectors_sorted)    

    subplot(4,1,2);
    cla();
    hold on;
    scatter(spike_edm_vectors_enc(1,class1_idxs),spike_edm_vectors_enc(2,class1_idxs),class1_color);
    scatter(spike_edm_vectors_enc(1,class2_idxs),spike_edm_vectors_enc(2,class2_idxs),class2_color);
    scatter(spike_edm_vectors_enc(1,class3_idxs),spike_edm_vectors_enc(2,class3_idxs),class3_color);

    subplot(4,1,3); 
    cla();
    hold on;
    plot(spike_maxs(class3_idxs,:)',class3_color,'LineWidth',1);
    plot(spike_maxs(class1_idxs,:)',class1_color,'LineWidth',1);
    plot(spike_maxs(class2_idxs,:)',class2_color,'LineWidth',1);
    axis([-inf inf min_peak max_peak]);
    title('Raw Spikes');   
    
    CHECK_ALL_POINTS = 0;
    if(CHECK_ALL_POINTS)
        for s = 25:30;
            subplot(4,1,2);
            cla();
            hold on;
            scatter(spike_edm_vectors_enc(1,class1_idxs),spike_edm_vectors_enc(2,class1_idxs),class1_color);
            scatter(spike_edm_vectors_enc(1,class2_idxs),spike_edm_vectors_enc(2,class2_idxs),class2_color);
            scatter(spike_edm_vectors_enc(1,class3_idxs),spike_edm_vectors_enc(2,class3_idxs),class3_color);
            scatter(spike_edm_vectors_enc(1,s),spike_edm_vectors_enc(2,s),'k');
            title('Autoencoder Hidden Units');

            subplot(4,1,3); 
            cla();
            hold on
            plot(spike_maxs(class1_idxs,:)',class1_color,'LineWidth',1);
            plot(spike_maxs(class2_idxs,:)',class2_color,'LineWidth',1);
            plot(spike_maxs(class3_idxs,:)',class3_color,'LineWidth',1);
            plot(spike_maxs(s,:),'k','LineWidth',2);
            axis([-inf inf min_peak max_peak]);
            title('Raw Spikes');

            fprintf('%i\n',s);
            pause        
        end

    end
    
    
    %% Run K-Means
    k=3;
    X=spike_edm_vectors_enc';
%     rng(1);
    [idx,C] = kmeans(X,k);
%     idx = kmeans(spike_edm_vectors_enc,k);
    
    fig = figure(40);
    clf(fig);    
    plot(idx);
%     grid_min = min(X(:,1));
%     grid_max = max(X(:,1));
    grid_min = 0;
    grid_max = 1;    
    x1 = grid_min:0.001:grid_max;
    x2 = grid_min:0.001:grid_max;
    [x1G,x2G] = meshgrid(x1,x2);
    XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
    idx2Region = kmeans(XGrid,3,'MaxIter',1,'Start',C); % get values of this grid
    
%     gscatter(XGrid(:,1),XGrid(:,2),idx2Region);
    imagesc(XGrid(:,1),XGrid(:,2),reshape(idx2Region, size(x1G)), 'AlphaData', .5);
%     mymap = [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0];
    mymap = [1,0,0;0,1,0;0,0,1]; %rgb
    mymap = [1,0,0;0,0,1;0,1,0]; %rbg
    colormap(mymap)
    set(gca,'YDir','normal')
%     imagesc(XGrid(:,1),XGrid(:,2),idx2Region);
%     gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
%     [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
    hold on;
    
    
    scatter(spike_edm_vectors_enc(1,class1_idxs),spike_edm_vectors_enc(2,class1_idxs), 'o', 'MarkerFaceColor', class1_color);
    scatter(spike_edm_vectors_enc(1,class2_idxs),spike_edm_vectors_enc(2,class2_idxs), 'o', 'MarkerFaceColor', class2_color);
    scatter(spike_edm_vectors_enc(1,class3_idxs),spike_edm_vectors_enc(2,class3_idxs), 'o', 'MarkerFaceColor', class3_color);    

    plot(C(:,1),C(:,2),'kx', 'MarkerSize',15,'LineWidth',3)  % Centroids
%     
    %% Save feature vectors to be exported to python
%     save('../DATA/spike_edm_vectors.mat','spike_edm_vectors_sorted','-v7')
    

    %% Plot each spike, position in tim eseries, and corresponding EDM activity   
%     ap_trace = zeros(length(filt_ch_data),1);   
%     for i = 1:n_spikes
%         spike_idx = spike_max_idxs(i);
%         subplot(3,1,1);
%         ap_trace(spike_idx) = ap_thresh;
%     end
    PLOT_EDM_VS_SPIKE_MAXS_VS_EDM = 0;
    if (PLOT_EDM_VS_SPIKE_MAXS_VS_EDM)
        fig = figure(21);
        clf(fig); 
        subplot(3,1,1);

        plot(filt_ch_data,'k');
        hold on;

        edm_trace = zeros(n_samp,1);
        edm_trace(edm_sample_idxs) = ap_thresh;
        stairs(edm_trace, class1_color);

        subplot(3,1,2);
        plot(spike_maxs');
        hold on;
        axis tight;
        subplot(3,1,3);
        imagesc(spike_edms_zscore{end})     
    end
    
    
    %% Plot time series v.s. EDM output and sampling points
    fig = figure(22);
    clf(fig); 
    ax(1) = subplot(2,1,1);

    plot(filt_ch_data,'k');
    hold on;
    
    edm_trace = zeros(n_samp,1);
    %edm_trace(edm_sample_idxs) = ap_thresh;
    stairs(edm_trace, class1_color);
    
    ax(2) = subplot(2,1,2);
    imagesc(edm_output_zscore)     
    linkaxes(ax,'x');
    axis tight;
    
    
    %% Plot autoencoder hidden layer representations to show clusters?

        
end


%% Plot an individual channel
% Perform EDM signal energy extraction for a single channel and perform PCA
EDMSEAE = 0;
if (EDMSEAE)
    fig = figure(1);
    clf(fig);
    plot(filt_ch_data, 'k');
    
    c_num = 10;
    channel_data = mcs_data(c_num,:);
    filt_ch_data = aps_filter(channel_data, Fs);
end



