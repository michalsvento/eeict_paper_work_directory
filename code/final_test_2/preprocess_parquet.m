clear

T = parquetread('dra_speech_total.parquet');

dra50 = T(1:3:end,:);
deep50 = T(2:3:end,:);
dra500 = T(3:3:end,:);




[fraction, stoi, pesq, snr, snr_gap, snr_max] = parse_table(dra50);

[fraction,std_fraction] = calc_avg(fraction);
[stoi    ,std_stoi    ] = calc_avg(stoi);
[pesq    ,std_pesq    ] = calc_avg(pesq);
[snr     ,std_snr     ] = calc_avg(snr);
[snr_gap ,std_snr_gap ] = calc_avg(snr_gap);
[snr_max ,std_snr_max ] = calc_avg(snr_max);


DRA50_STD = [std_fraction,std_stoi,std_pesq,std_snr,std_snr_gap,std_snr_max];
DRA50 = [fraction, stoi, pesq, snr, snr_gap, snr_max];
save('DRA50.mat',"DRA50",'-mat');
save('DRA50_std.mat',"DRA50_STD",'-mat');



[fraction, stoi, pesq, snr, snr_gap, snr_max] = parse_table(dra500);

[fraction,std_fraction] = calc_avg(fraction);
[stoi    ,std_stoi    ] = calc_avg(stoi);
[pesq    ,std_pesq    ] = calc_avg(pesq);
[snr     ,std_snr     ] = calc_avg(snr);
[snr_gap ,std_snr_gap ] = calc_avg(snr_gap);
[snr_max ,std_snr_max ] = calc_avg(snr_max);

DRA500_STD = [std_fraction,std_stoi,std_pesq,std_snr,std_snr_gap,std_snr_max];
DRA500 = [fraction, stoi, pesq, snr, snr_gap, snr_max];
save('DRA500.mat',"DRA500",'-mat');
save('DRA500_std.mat',"DRA500_STD",'-mat');


[fraction, stoi, pesq, snr, snr_gap, snr_max] = parse_table(deep50);

[fraction,std_fraction] = calc_avg(fraction);
[stoi    ,std_stoi    ] = calc_avg(stoi);
[pesq    ,std_pesq    ] = calc_avg(pesq);
[snr     ,std_snr     ] = calc_avg(snr);
[snr_gap ,std_snr_gap ] = calc_avg(snr_gap);
[snr_max ,std_snr_max ] = calc_avg(snr_max);


Denoise50_STD = [std_fraction,std_stoi,std_pesq,std_snr,std_snr_gap,std_snr_max];
Denoise50 = [fraction, stoi, pesq, snr, snr_gap, snr_max];
save('Denoise50.mat',"Denoise50",'-mat');
save('Denoise50_std.mat',"Denoise50_STD",'-mat');







function [avg_arr, std_arr] = calc_avg(data)
    avg_arr=zeros(11,1);
    std_arr=zeros(11,1);
    for i = 1:11
        avg_arr(i)=sum(data(i:11:end))./10;
    end

    for i = 1:11
        std_arr(i)=sqrt(sum((data(i:11:end)-avg_arr(i)).^2)./10);
    end
end




function [fraction, stoi, pesq, snr, snr_gap, snr_max] = parse_table(T)   

% parse data
fraction = column2array(T(:, 'proj_frac'));
stoi = column2array(T(:, 'stoi'));
pesq = column2array(T(:, 'pesq'));
snr = column2array(T(:, 'snr'));
snr_gap = column2array(T(:, 'snr_gap'));
snr_max = column2array(T(:, 'SNR_MAX'));

end


function arr = column2array(col)

    arr = table2array(col);
    arr = reshape(arr', 1, [])'; % reshape to 3 columns

end
