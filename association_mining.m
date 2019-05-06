%% Restart
close all;
clear variables;
clc;

%% Load data
filepath = 'abalone.csv';
data = readtable(filepath);
AttributeNames = {'sex' 'length' 'diameter' 'height' 'whole_weight' 'shucked_weight' 'viscera_weight' 'shell_weight' 'rings'};
data.Properties.VariableNames = AttributeNames;

%% Binarize data
% One-of-K encode 'sex'
bin_data = table;
bin_data.male = cell2mat(data.sex) == 'M';
bin_data.female = cell2mat(data.sex) == 'F';
bin_data.infant = cell2mat(data.sex) == 'I';

% Split other features into N quantiles, where N >= 2
N = 4;
for i = 2:size(data,2)
    if N == 2
        Q_sep = quantile(table2array(data(:, i)), [0.5]);
    else
        Q_sep = quantile(table2array(data(:, i)), N-1);
    end
    
    Q = cell(1, N);
    varnames = cell(1,N);
    
    for j = 1:N
        if j == 1
            Q{j} = table2array(data(:,i)) < Q_sep(j);
        elseif j == N
            Q{j} = Q_sep(j-1) <= table2array(data(:,i));
        else
            Q{j} = Q_sep(j-1) <= table2array(data(:,i)) & table2array(data(:,i)) < Q_sep(j);
        end
        
        varnames{j} = join([data.Properties.VariableNames{i}, '_Q', num2str(j)]);
        bin_data = addvars(bin_data, Q{j}, 'NewVariableNames', varnames{j});
    end
end

%% Find rules using Apriori algorithm
apriori_data = table2array(bin_data);
apriori_attributes = bin_data.Properties.VariableNames;

[Rules, FreqItemsets] = findRules(apriori_data, 0.2, 0.95, 5000, 2, apriori_attributes, 'rules');

%% Plot size distribution of frequent itemsets
figure(1);
hold on;
for i = 1:size(FreqItemsets, 2)
    bar(i, size(FreqItemsets{1,i},1));
end
hold off;
title('Distribution of size of frequent itemsets');
xlabel('size of itemset');
ylabel('number of itemsets');
saveas(gcf, 'itemset_distribution', 'epsc');
