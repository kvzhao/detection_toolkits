%
% This function converts detections in TXT format as used by Caltech
% Evalution code (http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip) 
%
% into a single JSON file compatible with COCO
%
% INPUT: directory of files in the format Vxxx.txt where xxx is the
% sequence number
% 
% OUTPUT: A JSON file in COCO-compatible format with all the detections
%
% Sample: txtDetectionsToJson('nightowlsval', 'C:\Work\myDetectionsDirectory', 'C:\Work\detections.json')
function txtDetectionsToJson(datasetName, detectionsDir, outputJsonFile)    
    
    [~,setIds,vidIds,skip] = dbInfo(datasetName);
    assert(length(setIds) == 1, 'Single set expected')
    s=1;  
    
    dt_coco = [];
    ndt = 0;
    for v=1:length(vidIds{s}), v1=vidIds{s}(v);
      A=loadVbb(s,v); frames=skip-1:skip:A.nFrame-1;
      vName=sprintf('%s/V%03d',detectionsDir,v1);   
      fprintf('Loading detections from %s.txt\n', vName)      
      bbs=load([vName '.txt'],'-ascii');
      for f=frames
          dts=bbs(bbs(:,1)==f+1,:);
          
          for i=1:size(dts,1)                
            ndt=ndt+1;
            dt=dts(i,:);
            dt_coco(ndt).image_id=int32(A.imageIds(f+1));
            dt_coco(ndt).category_id=1;
            dt_coco(ndt).bbox=dt(2:5);
            dt_coco(ndt).score=dt(6);
        
          end
          
          
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

