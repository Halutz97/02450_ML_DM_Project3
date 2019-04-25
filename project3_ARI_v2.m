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
file_path = fullfile(cd(), 'Scripts/Project3_ML-DM/abalone.csv');

%file_path = fullfile(cd(), 'Scripts\Project3_ML-DM\abalone.csv');
%"No! We are here"
%
% Load the data
abalone_table = readtable(file_path);
AttributeNames = {'Sex' 'Length' 'Diameter' 'Height' 'Whole weight' 'Shucked weight' 'Viscera weight' 'Shell weight' 'Rings'};

% 1 out of K coded:
Col1   =  OneOutOfKCoding(stringArray2num(string(table2array(abalone_table(:, 1)))))
% Numerical values:
Col2_9 = table2array(abalone_table(:, 2:9)); 
% Join arrays back together:
JoinedData = horzcat(Col1,Col2_9);
% Normalize the data
NormData1 = (JoinedData-mean(JoinedData))./std(JoinedData);
NormData2 = horzcat((Col1-mean(Col1))./sqrt(size(Col1,2)),(Col2_9-mean(Col2_9)./std(Col2_9)));
%% Clustering using Gaussian Mixture Model (GMM)






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
