clear

% T = parquetread('dra_speech_total.parquet');

snrcrop = -10; % set -Inf if no cropping is required

% parse data
% fraction = column2array(T(:, 'proj_frac'));
% stoi = column2array(T(:, 'stoi'));
% pesq = column2array(T(:, 'pesq'));
% snr = column2array(T(:, 'snr'));
% snr_gap = column2array(T(:, 'snr_gap'));
% snr_max = column2array(T(:, 'SNR_MAX'));


load("Denoise50.mat")
load("Denoise50_std.mat")
load("DRA50.mat")
load("DRA50_std.mat")
load("DRA500.mat")
load("DRA500_std.mat")

fraction    =[DRA50(:,1),Denoise50(:,1),DRA500(:,1)];
stoi        =[DRA50(:,2),Denoise50(:,2),DRA500(:,2)];
pesq        =[DRA50(:,3),Denoise50(:,3),DRA500(:,3)];
snr         =[DRA50(:,4),Denoise50(:,4),DRA500(:,4)];
snr_gap     =[DRA50(:,5),Denoise50(:,5),DRA500(:,5)];
snr_max     =[DRA50(:,6),Denoise50(:,6),DRA500(:,6)];


fraction_std    =[DRA50_STD(:,1),Denoise50_STD(:,1),DRA500_STD(:,1)];
stoi_std        =[DRA50_STD(:,2),Denoise50_STD(:,2),DRA500_STD(:,2)];
pesq_std        =[DRA50_STD(:,3),Denoise50_STD(:,3),DRA500_STD(:,3)];
snr_std         =[DRA50_STD(:,4),Denoise50_STD(:,4),DRA500_STD(:,4)];
snr_gap_std     =[DRA50_STD(:,5),Denoise50_STD(:,5),DRA500_STD(:,5)];
snr_max_std     =[DRA50_STD(:,6),Denoise50_STD(:,6),DRA500_STD(:,6)];


legend_titles = {'DRA 50', 'Denoise','DRA 500'};



% plot
figure
tiledlayout('flow')

nexttile
bar(fraction, stoi)
xlabel('proj fraction')
ylabel('STOI')
legend(legend_titles, 'location', 'southeast')

hold on

% er = errorbar(x',stoi,stoi_std);                             
% er.LineStyle = 'none';  
% hold off

nexttile
bar(fraction, pesq)
xlabel('proj fraction')
ylabel('PESQ')
legend(legend_titles, 'location', 'southeast')

nexttile
bar(fraction, snr)
xlabel('proj fraction')
ylabel('SNR (dB)')
lims = ylim;
ylim([max(lims(1), snrcrop), Inf])
legend(legend_titles, 'location', 'southeast')

nexttile
bar(fraction, snr_gap)
xlabel('proj fraction')
ylabel('SNR in gap (dB)')
lims = ylim;
ylim([max(lims(1), snrcrop), Inf])
legend(legend_titles, 'location', 'southeast')

nexttile
bar(fraction, snr_max)
xlabel('proj fraction')
ylabel('SNR max (dB)')
lims = ylim;
ylim([max(lims(1), snrcrop), Inf])
legend(legend_titles, 'location', 'southeast')

function arr = column2array(col)

    arr = table2array(col);
    arr = reshape(arr', 3, [])'; % reshape to 3 columns

end



