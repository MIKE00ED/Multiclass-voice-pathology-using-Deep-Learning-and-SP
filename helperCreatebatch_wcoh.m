function helperCreatebatch_wcoh(ads,writetofolder1)

for i = 1:length(ads.Files)
    i
    imgLoc1 = fullfile(writetofolder1,char(ads.Labels(i)));
    imFileName1 = strcat(char(ads.Labels(i)),'_',num2str(i));
    filename1=fullfile(imgLoc1,imFileName1);
        
    [sig Fs] = audioread(ads.Files{i});
    sig=resample(sig,8000,Fs);
    Fs=8000;

    fb_b = cwtfilterbank('SignalLength',length(sig),...
       'SamplingFrequency',Fs,...
        'VoicesPerOctave',4,'Wavelet','bump');
    [cfs_b,frq_b] = wt(fb_b,sig);

    fb_m = cwtfilterbank('SignalLength',length(sig),...
        'SamplingFrequency',Fs,...
        'VoicesPerOctave',4,'Wavelet','morse');
    [cfs_m,frq_m] = wt(fb_m,sig);

    signal1 = abs(cfs_b);
    [r1 c1] = size(signal1);

    signal2 = abs(cfs_m);
    [r2 c2] = size(signal2);

    r3 = max(r1,r2); c3 = max(c1,c2);

    signal1_new = imresize(signal1,[r3,c3]);
    signal2_new = imresize(signal2,[r3,c3]);

    signal1_new = (signal1_new(:))';
    signal2_new = (signal2_new(:))';
    
    [wcoh,~,F] = wcoherence(signal1_new,signal2_new);
    
    pcolor(wcoh) 
    colormap(jet(128))
    set(gca,'yscale','log', 'Visible', 'off');shading interp;axis tight; % set visible settings on or off to view
    colorbar('off');
    export_fig(filename1,'-jpg')   
    pause(2);
    close all;
   
    
end
end