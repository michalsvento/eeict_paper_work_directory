clear

T = parquetread('lambda.parquet');

SNR = table2array(T);
legcell = {'\lambda_0 = 0.1 descending','\lambda_0 = 0.1 static','\lambda_0 = 1 descending','\lambda_0 = 1 static'};

figure
plot(SNR,LineWidth=1)
xlabel('iterations')
ylabel('SNR [dB]')

legend(legcell, 'location', 'southeast')