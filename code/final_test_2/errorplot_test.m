% Example data 
model_series = [10 40 50 60; 20 50 60 70; 30 60 80 90]; 
model_error = [1 4 8 6; 2 5 9 12; 3 6 10 13]; 
b = bar(model_series, 'grouped');
hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);
% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
% Plot the errorbars
errorbar(x',model_series,model_error,'linestyle','none');
hold off