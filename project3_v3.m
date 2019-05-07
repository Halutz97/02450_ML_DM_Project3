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

clc; clear; close all;
%% Data Preparation:
%Path to the dataset
My_mfilename = mfilename('fullpath');
cdir = fileparts(cd());
% Path to the file
% Works if BOTH the script and data are in the 'Scripts' folder!
% Instructions: Run setup. Run the script (press 'add to path').
file_path = fullfile(cd(), 'Scripts/HomeComputer/02450_ML_DM_Project3/abalone.csv');

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
% Number of clusters
K = 3;
% Fit model
G = gmdistribution.fit(X, K,'regularize',10e-9);
% Compute clustering
i = cluster(G, X);
% %% Extract cluster centers
X_c = G.mu;
Sigma_c=G.Sigma;
% %% Plot results
mfig('GMM: Clustering K = 3'); clf; 
clusterplot(X, y, i, X_c, Sigma_c);


%% Cross-Validation for estimating the number of components in the GMM

n_clusters = 15; % Increase to estimate number of components needed.
iAverage = nan(size(y,1),n_clusters)

% Range of K's to try
KRange = 1:n_clusters;
T = length(KRange);

% Allocate variables
BIC = nan(T,1);
AIC = nan(T,1);
CVE = zeros(T,1);

% Create crossvalidation partition for evaluation
CV = cvpartition(N, 'Kfold', 10);

% For each model order
for t = 11  
    % Get the current K
    K = KRange(t);
    
    % Display information
    fprintf('Fitting model for K=%d\n', K);
    
    % Fit model
    G = gmdistribution.fit(X, K, 'Replicates', 10);
    
    % Get BIC and AIC
    BIC(t) = G.BIC;
    AIC(t) = G.AIC;
    
    % For each crossvalidation fold
    for k = 1:CV.NumTestSets
        % Extract the training and test set
        X_train = X(CV.training(k), :);
        X_test = X(CV.test(k), :);
        
        % Fit model to training set
        G = gmdistribution.fit(X_train, K, 'Replicates', 10);
        
        % Evaluation crossvalidation error
        [~, NLOGL] = posterior(G, X_test);
        CVE(t) = CVE(t)+NLOGL;
    end
    % Compute clustering
    i = cluster(G, X);
    % %% Extract cluster centers
    iAverage(:,t) = i;
    X_c = G.mu;
    Sigma_c=G.Sigma;
    % %% Plot results
    figname = "GMM: Clustering K = " + t;
    mfig('1234'); clf; 
    clusterplot(X, y, i, X_c, Sigma_c);
end


% Plot results

mfig('GMM: Number of clusters'); clf; hold all
plot(KRange, BIC);
plot(KRange, AIC);
plot(KRange, 2*CVE);
legend('BIC', 'AIC', 'Crossvalidation');
xlabel('K');
%%
iAverage
figure
for j = 1:T
    subplot(4,4,j)
    confusionchart(y,iAverage(:,j))
end
%%
Axislabels = {'Females','Males','Infants'}
% countries = {'Botswana','Lesotho','Iceland'};
figure
confusionchart(y,iAverage(:,11))
% legend('a','b','c')
%%

Maxclust = 11
    
% Allocate variables
% BIC = nan(T,1);
% AIC = nan(T,1);
CVE = zeros(T,1);

KRange = 1:Maxclust;
T = length(KRange);
for t = 1:T
    % Display information
    fprintf('Fitting model for K=%d\n', K);

    for k = 1:CV.NumTestSets
        % Extract the training and test set
        X_train = X(CV.training(k), :);
        X_test = X(CV.test(k), :);
        
        % Fit model to training set
%         G = gmdistribution.fit(X_train, K, 'Replicates', 10);
        Z = linkage(X_train,'complete')
        i = cluster(Z,'Maxclust',Maxclust)
        % Evaluation crossvalidation error
        [~, NLOGL] = posterior(G, X_test);
        CVE(t) = CVE(t)+NLOGL;
    end
end
%% Hierarchical Clustering:
% How do we make it actually cluster the data?
% Should we use something other than eucledian distance?
% Should we use the raw data, X, in linkage or the distance between the
% data in pdist as the input to linkage?

Maxclust = 11
% XHC = pdist(X,'euclidean');
% Z = linkage(XHC,'average');
% i = cluster(Z,'Maxclust',Maxclust)
% %% Plot result
% % Plot dendrogram
% mfig('Dendrogram'); clf;
% dendrogram(Z);
% 
% %% Plot data
% mfig('Hierarchical'); clf; 
% clusterplot(X, y, i);
% %%
Z2 = linkage(X,'complete')
i2 = cluster(Z2,'Maxclust',Maxclust)
mfig('Dendrogram 2'); clf;
dendrogram(Z2,0);
mfig('Dendrogram 3'); clf;
dendrogram(Z2);
% legend('Category 1','Category 2','Category 3','','Female','Male','Infant')
mfig('Hierarchical 2'); clf; 
clusterplot(X, y, i2);
legend('Category 1','Category 2','Category 3','Category 4','Female','Male','Infant')
figure
C = confusionmat(y,i2)
confusionchart(y,i2)
%%
figure
for i = 1:3
    subplot(3,1,i)
    confusionchart(y,mod(i2+i,3)+1)
end

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
xticklabels(index_vector(1,:))
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
xticklabels(index_vector(2,:))
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
xticklabels(index_vector(3,:))
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
