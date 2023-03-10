clear

T = parquetread('dra_gamma_test.parquet');

snrcrop = -10; % set -Inf if no cropping is required
legcell = {'\gamma = 0.001', '\gamma = 0.01', '\gamma = 0.1'};

% parse data
fraction = column2array(T(:, 'proj_frac'));
stoi = column2array(T(:, 'stoi'));
pesq = column2array(T(:, 'pesq'));
snr = column2array(T(:, 'snr'));
snr_gap = column2array(T(:, 'snr_gap'));
snr_max = column2array(T(:, 'SNR_MAX'));

% plot
figure
tiledlayout('flow')

% nexttile
% bar(fraction, stoi,1.0)
% xlabel('proj fraction')
% ylabel('STOI')
% legend(legcell, 'location', 'southeast')

nexttile
bar(fraction, pesq,1.0)
xlabel('proj fraction')
ylabel('PESQ')
legend(legcell, 'location', 'southeast')

% nexttile
% bar(fraction, snr)
% xlabel('proj fraction')
% ylabel('SNR (dB)')
% lims = ylim;
% ylim([max(lims(1), snrcrop), Inf])
% legend(legcell, 'location', 'southeast')
% 
% nexttile
% bar(fraction, snr_gap)
% xlabel('proj fraction')
% ylabel('SNR in gap (dB)')
% lims = ylim;
% ylim([max(lims(1), snrcrop), Inf])
% legend(legcell, 'location', 'southeast')
% 
% nexttile
% bar(fraction, snr_max)
% xlabel('proj fraction')
% ylabel('SNR max (dB)')
% lims = ylim;
% ylim([max(lims(1), snrcrop), Inf])
% legend(legcell, 'location', 'southeast')

function arr = column2array(col)

    arr = table2array(col);
    arr = reshape(arr, [], 3); % reshape to 3 columns

end