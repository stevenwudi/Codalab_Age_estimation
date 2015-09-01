clc;close all;clear;
% This script will detect faces in a set of images
% The minimum face detection size is 36 pixels,
% the maximum size is the full image.
addpath(genpath('/idiap/user/dwu/spyder/Codalab_Age_estimation/voc-dpm-master'))

results_folder_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/DPM_face_detection_result/';

model_path ='/idiap/user/dwu/spyder/Codalab_Age_estimation/voc-dpm-master/dpm_baseline.mat';

face_model = load(model_path);

% lower detection threshold generates more detections
% detection_threshold = -0.5; 
detection_threshold = 0; 

% 0.3 or 0.2 are adequate for face detection.
nms_threshold = 0.3;

% we only save the bounding box
images_folder_path = '/media/dwu/REPERE/MORPH/morph/Album2';
image_folders = dir(fullfile(images_folder_path));




% ages = zeros(55134,1);
% boundingbox = ones(55134, 4);
load('boundingbox_morph.mat');
count = 0;


%%%%%%%%%%%%%
for f = 4:numel(image_folders)-1
    image_folder_name = image_folders(f).name;
    image_folder_path = fullfile(images_folder_path, image_folder_name);
    image_folders_invidual = dir(fullfile(image_folder_path, '*.JPG'));
    for i= 1:numel(image_folders_invidual)
        scaleVector = 1;  
        count = count + 1;
        disp(count);
        if count>51692
            disp(image_folders_invidual(i).name);
            image_name = image_folders_invidual(i).name;
            image_path = fullfile(images_folder_path, image_folder_name, image_name);
            image = imread(image_path);
            [ds, bs] = process_face(image, face_model.model,  ...
                            detection_threshold, nms_threshold);                       
            % if there is no detection result, we upscale the image of one octave
            while  sum(size(ds)) == 0 && scaleVector < 4
                scaleVector = scaleVector+1;
                disp('resize the image by one octave')
                image = imresize(image, 1.41);
                [ds, bs] = process_face(image, face_model.model,  ...
                                detection_threshold, nms_threshold);
            end
            % we choose the best detection result
    %         if sum(size(ds)) > 7
    %             ds = ds(1,:);
    %             disp('multiple faces detected');
    %         end
            %

            if sum(size(ds)) == 7 % one face detected
                boundingbox(count,:) = ds(1,1:4);
            elseif sum(size(ds)) == 0  % no face detected
                boundingbox(count,:) = [0, 0, 0, 0];
            end
            ages(count) = str2num(image_folders_invidual(i).name(11:12));
            %result_path = fullfile(results_folder_path, [image_name, '.result.png']);
            %showsboxes_face(image, ds, result_path);
            %disp(['Created ', result_path]);
        end
    end

end

save('boundingbox_morph.mat', 'boundingbox', 'ages');
disp('All images processed');


