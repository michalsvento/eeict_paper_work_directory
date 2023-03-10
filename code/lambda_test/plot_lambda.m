clear

T = parquetread('lambda.parquet');

SNR = table2array(T);
legcell = {'\lambda = 0.1 descending','\lambda = 0.1 static','\lambda = 1 descending','\lambda = 1 static'};

figure
plot(SNR,LineWidth=1)
xlabel('iterations')
ylabel('SNR [dB]')

legend(legcell, 'location', 'southeast')