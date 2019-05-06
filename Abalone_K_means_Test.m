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

% Works if BOTH the script and data are in the GitKraken folder!
% Instructions: Run setup. Run the script (press 'add to path').

%file_path = fullfile(cd(), 'Scripts/abalone.csv');
file_path = fullfile(cd(), 'Scripts/02450_ML_DM_Project3/abalone.csv');

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
%% KMeans test


% Subset of data
SubData = NormData1(1:200,1:11);
% Number of clusters
K = 4;


y = 1;  % Don't know what y does.. I just assigned y some random value

% Run k-means
[i, Xc] = kmeans(SubData, K);
% Here, squared euclidean distance measure is used (default)

% Plot data
mfig('K-means 1'); clf; 
clusterplot(SubData, y, i, Xc);
%%
% Okay. We can see from the plot, that it seems to be more resonable to
% have 3 clusters..
% And the squared euclidean distance measure is not optimal in this case!


% Let's try something else:

K = 3;

% Update: y is labels.. So if we say that we have 3 labels.. Let's try to define y:
SubLabel = JoinedData(1:200,1:3);
y = zeros(200,1);
y = SubLabel(:,1).*1 + SubLabel(:,2).*2 + SubLabel(:,3).*3; 
%%
[i, Xc] = kmeans(SubData, K,'distance','correlation');
% Distance is set to 'correlation'.

% Plot data
mfig('K-means 2'); clf; 
clusterplot(SubData, y, i, Xc);
% Notice: The dots indicate the different labels, and the circles indicate
% the clusters - it's easy to get confused!

% Looks better!
% We see that the different clusters are indeed Male, Female and Infant..

%% Let's try some clustervalidation (We know the result should be 3):

K = 6; %Max clusters.

% Allocate variables
Rand = nan(K,1);
Jaccard = nan(K,1);
NMI = nan(K,1);

for k = 3 %1:K    
    % Run k-means
    [i, Xc] = kmeans(SubData, k);
    
    % Compute cluster validities
    [Rand(k), Jaccard(k), NMI(k)] = clusterval(y, i);
end

% Plot results

mfig('Cluster validity'); clf; hold all;
plot(1:K, Rand);
plot(1:K, Jaccard);
plot(1:K, NMI);

legend({'Rand', 'Jaccard','NMI'});

% Okay, this looks kind of weird..




















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
    otherwise
        out = -1;
end
end
