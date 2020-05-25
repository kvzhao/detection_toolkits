%
% This function converts detections in Matlab MAT format 
%
% into a single JSON file compatible with COCO
%
% INPUT: A single MAT file with 1xN cell array called 'dt' where N is the number images
% in the dataset; each cell contains a Mx5 matrix of detections in the
% format [x, y, w, h, score]
% 
% OUTPUT: A JSON file in COCO-compatible format with all the detections
%
% Sample: matDetectionsToJson('nightowls_validation.json', 'C:\Work\detections.mat', 'C:\Work\detections.json')
function matDetectionsToJson(annotationJsonFile, detectionsMatFile, outputJsonFile)    
    

   
    fprintf('reading annotations from %s \n', annotationJsonFile) ;
    tmp = fileread(annotationJsonFile) ; 
    annoData = jsondecode(tmp);
    
    
    % Load image ids from the annotations file
    image_ids = [annoData.images.id];
    D = load(detectionsMatFile);
    
    assert(length(D.dt) == length(image_ids), 'The cell array must contain the same number of elements as the number of images in the dataset')
    
    ndt = 0;
    for i=1:length(D.dt)
        bbs=D.dt{i};
        for ibb=1:size(bbs,1)
            ndt=ndt+1;
            bb=bbs(ibb,:);
            dt_coco(ndt).image_id= int32(image_ids(i));
            dt_coco(ndt).category_id=1;
            dt_coco(ndt).bbox=bb(1:4);
            dt_coco(ndt).score=bb(5);
        end
    end
        
    
    dt_string = jsonencode(dt_coco);
    fp = fopen(outputJsonFile,'w');
    fprintf(fp,'%s',dt_string);
    fclose(fp);
    
    fprintf('Written JSON file %s\n', outputJsonFile)
    
    
end


% Adapted from Caltech Evaluation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = loadVbb( s, v )
[pth,sIds,vIds]=dbInfo;
fName=@(s,v) sprintf('%s/annotations/set%02i/V%03i',pth,s,v);
fPath = fName(sIds(s),vIds{s}(v));
fprintf('Getting image ids from %s.vbb\n', fPath)
A=vbb('vbbLoad', fPath);
end

