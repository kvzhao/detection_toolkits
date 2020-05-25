
%
% This function shows samples from the dataset as a video sequence
%
% Requires Matlab 2017a or newer
%

function demo()

    %Reference Caltech Pedestrian evaluation code by P Dollar    
    addpath './caltecheval'
    
    % Referece Piotr's Image & Video Matlab Toolbox 
    % Download from: https://github.com/pdollar/toolbox
    
    addpath(genpath('/pdollar/toolbox/videos'))  %%% MODIFY HERE TO CORRESPOND TO YOUR LOCATION
    
    
    % We assume the data is located in ./data-nightowls folder, using the
    % following folder structure
    % the resulting folder structure should look like:
    %
    % ./data-nightowls/annotations/set01/V001.vbb
    % ./data-nightowls/annotations/set01/V002.vbb
    % 
    % ...
    % 
    % ./data-nightowls/videos/set01/V001.seq
    % ./data-nightowls/videos/set01/V001-seek.seq
    % ./data-nightowls/videos/set01/V002.seq
    % ./data-nightowls/videos/set01/V002-seek.seq
    % 
    % ...
    
    
    dbInfo('nightowlstrain')    
    
    vbbPlayer(1, 1)
end