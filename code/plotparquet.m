clear

T = parquetread('td_input_noise_ref_clean/dra_speech_total_noisein_cleanref.parquet');

snrcrop = -10; % set -Inf if no cropping is required

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

nexttile
bar(fraction, stoi)
xlabel('proj fraction')
ylabel('STOI')
legend('conventional', 'denoiser', 'location', 'southeast')

nexttile
bar(fraction, pesq)
xlabel('proj fraction')
ylabel('PESQ')
legend('conventional', 'denoiser', 'location', 'southeast')

nexttile
bar(fraction, snr)
xlabel('proj fraction')
ylabel('SNR (dB)')
lims = ylim;
ylim([max(lims(1), snrcrop), Inf])
legend('conventional', 'denoiser', 'location', 'southeast')

nexttile
bar(fraction, snr_gap)
xlabel('proj fraction')
ylabel('SNR in gap (dB)')
lims = ylim;
ylim([max(lims(1), snrcrop), Inf])
legend('conventional', 'denoiser', 'location', 'southeast')

nexttile
bar(fraction, snr_max)
xlabel('proj fraction')
ylabel('SNR max (dB)')
lims = ylim;
ylim([max(lims(1), snrcrop), Inf])
legend('conventional', 'denoiser', 'location', 'southeast')

function arr = column2array(col)

    arr = table2array(col);
    arr = reshape(arr', 2, [])'; % reshape to 2 columns

end