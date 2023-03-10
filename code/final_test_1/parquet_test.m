data = parquetread("dra_speech_total.parquet");



% for i=1:3:size(data,1)
%     data.x__index_level_0__(i) = i-1;
% end
% 
% parquetwrite("dra_speech_total.parquet",data)
