%% Project 3 - 02450 Introduction to Machine Learning and Data Mining
%#########################################%
%##   This script only works in 2018b   ##%
%##   It will probably not function     ##%
%##   completely in earlier versions.   ##%
%##                                     ##%
%##   The course toolbox for Matlab     ##%
%##   must be installed for the         ##%
%##   script to work.                   ##%
%#########################################%

% clc; clear; close all;
%% Data Preparation:
%Path to the dataset
My_mfilename = mfilename('fullpath');
cdir = fileparts(cd());
% Path to the file
% Works if BOTH the script and data are in the 'Scripts' folder!
% Instructions: Run setup. Run the script (press 'add to path').
file_path = fullfile(cd(), 'Scripts/02450_ML_DM_Project3/abalone.csv');

%file_path = fullfile(cd(), 'Scripts\02450_ML_DM_Project3\abalone.csv');
%"No! We are here"
%
% Load the data
abalone_table = readtable(file_path);
AttributeNames = {'Sex' 'Length' 'Diameter' 'Height' 'Whole weight' 'Shucked weight' 'Viscera weight' 'Shell weight' 'Rings'};

% 1 out of K coded:
Col1   =  OneOutOfKCoding(stringArray2num(string(table2array(abalone_table(:, 1)))));
% Numerical values:
Col2_9 = table2array(abalone_table(:, 2:9)); 
% Join arrays back together:
JoinedData = horzcat(Col1,Col2_9);
% Normalize the data
NormData1 = (JoinedData-mean(JoinedData))./std(JoinedData);
NormData2 = horzcat((Col1-mean(Col1))./sqrt(size(Col1,2)),(Col2_9-mean(Col2_9)./std(Col2_9)));
NormData3 = (Col2_9-mean(Col2_9))./std(Col2_9);
%%
X = NormData3;
N = size(X,1);
%% Known class labels:
SubLabel = JoinedData(:,1:3);
y = SubLabel(:,1).*1 + SubLabel(:,2).*2 + SubLabel(:,3).*3; 
%% Clustering using Gaussian Mixture Model (GMM)



%% Hierarchical Clustering:
% How do we make it actually cluster the data?
% Should we use something other than eucledian distance?
% Should we use the raw data, X, in linkage or the distance between the
% data in pdist as the input to linkage?

Maxclust = 6
XHC = pdist(X,'euclidean');
Z = linkage(XHC,'average');
i = cluster(Z,'Maxclust',Maxclust)
%% Plot result
% Plot dendrogram
mfig('Dendrogram'); clf;
dendrogram(Z);

%% Plot data
mfig('Hierarchical'); clf; 
clusterplot(X, y, i);

%%
%% K-means clustering - Taken from ex10_1_3.m
% Maximum number of clusters
K = 10;
% Allocate variables
Rand = nan(K,1);
Jaccard = nan(K,1);
NMI = nan(K,1);
% minArrayIndex = zeros(N,K);
% minArrayVal = zeros(N,K);
for k = 1:K
    % Run k-means
    [i, Xc, SUMD, D] = kmeans(X, k,'distance','correlation');%, 'Display', 'iter', 'OnlinePhase', 'off'
%     [minArrayVal(:,k), minArrayIndex(:,k)] = min(D,[],2); % Get the index with the smallest index
    % Compute cluster validities
    [Rand(k), Jaccard(k), NMI(k)] = ...
        clusterval(y, i);
    figname = string("My K-means "+k)
    mfig(figname); clf; 
    clusterplot(X, y, i, Xc);%, Xc
end
%%


    
    
%% Plot results

mfig('Cluster validity'); clf; hold all;
plot(1:K, Rand);
plot(1:K, Jaccard);
plot(1:K, NMI);

legend({'Rand', 'Jaccard','NMI'});


%% Outlier Detection

[N,M] = size(X);


%%
%Vector to save indexes!
index_vector = zeros(3,20);


%% Gausian Kernel density estimator
% cross-validate kernel width by leave-one-out-cross-validation
% automatically implemented in the script gausKernelDensity
widths=max(var(X))*(2.^[-10:2]); % evaluate for a range of kernel widths
for w=1:length(widths)
   [density,log_density]=gausKernelDensity(X,widths(w));
   logP(w)=sum(log_density);
end
[val,ind]=max(logP);
width=widths(ind);
display(['Optimal kernel width is ' num2str(width)])
% evaluate density for estimated width
density=gausKernelDensity(X,width);

% Sort the densities
[y,i] = sort(density);
index_vector(1,:) = i(1:20);


% Plot outlier scores
mfig('Gaussian Kernel Density: outlier score'); clf;
bar(y(1:20));
%%
%mfig('Gaussian Kernel Density: outlier score (index_vector)'); clf;
%bar(y(1:20));
%set(gca,'XTickLabel',string(index_vector(1,:)))



%%
% % Plot possible outliers
% mfig('Gaussian Kernel Density: Possible outliers'); clf;
% for k = 1:20
%     subplot(4,5,k);
%     imagesc(reshape(X(i(k),:), 16, 16)); 
%     title(k);
%     colormap(1-gray); 
%     axis image off;
% end

%% K-nearest neighbor density estimator 

% Number of neighbors
K = 5;

% Find the k nearest neighbors
[idx, D] = knnsearch(X, X, 'K', K+1);

% Compute the density
density = 1./(sum(D(:,2:end),2)/K);

% Sort the densities
[y,i] = sort(density);
index_vector(2,:) = i(1:20);


% Plot outlier scores
mfig('KNN density: outlier score'); clf;
bar(y(1:20));

% % Plot possible outliers
% mfig('KNN density: Possible outliers'); clf;
% for k = 1:20
%     subplot(4,5,k);
%     imagesc(reshape(X(i(k),:), 16, 16)); 
%     title(k);
%     colormap(1-gray); 
%     axis image off;
% end

%% K-nearest neigbor average relative density
% Compute the average relative density
avg_rel_density=density./(sum(density(idx(:,2:end)),2)/K);

% Sort the densities
[y,i] = sort(avg_rel_density);
index_vector(3,:) = i(1:20);

% Plot outlier scores
mfig('KNN average relative density: outlier score'); clf;
bar(y(1:20));

% % Plot possible outliers
% mfig('KNN average relative density: Possible outliers'); clf;
% for k = 1:20
%     subplot(4,5,k);
%     imagesc(reshape(X(i(k),:), 16, 16)); 
%     title(k);
%     colormap(1-gray); 
%     axis image off;
% end

 %% Distance to 5'th nearest neighbor outlier score
% 
% % Neighbor to use
% K = 5;
% 
% % Find the k nearest neighbors
% [i, D] = knnsearch(X, X, 'K', K+1);
% 
% % Outlier score
% f = D(:,K+1);
% 
% % Sort the outlier scores
% [y,i] = sort(f, 'descend');
% 
% % Plot kernel density estimate outlier scores
% mfig('Distance: Outlier score'); clf;
% bar(y(1:20));
% 
% % % Plot possible outliers
% % mfig('Distance: Possible outliers'); clf;
% % for k = 1:20
% %     subplot(4,5,k);
% %     imagesc(reshape(X(i(k),:), 16, 16)); 
% %     title(k);
% %     colormap(1-gray); 
% %     axis image off;
% % end
% 



%%
%% Functions:
function out = OneOutOfKCoding(A) %A is Assumed 1D with the same attribute in a column
    C = unique(A);
    out = zeros(length(A(:)),length(C));
    for i = 1 : length(A(:))
        out(i,find(C == A(i),1)) = 1;
    end
end

function B = stringArray2num(A)
B = zeros(length(A),1);
    for n = 1 : length(A)
        B(n) = c2n(A(n));
    end
end

function out = c2n(colorStr)
switch colorStr
    case "F"
        out = 1;
    case "M"
        out = 2;
    case "I"
        out = 3;
    case "gold"
        out = 4;
    case "white"
        out = 5;
    case "black"
        out = 6;
    case "brown"
        out = 7;
    case "orange"
        out = 8;
    otherwise
        out = -1;
end
end
