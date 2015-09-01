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


if true
    fileID = fopen('/idiap/user/dwu/spyder/Codalab_Age_estimation/train_matlab_squence.csv','w');
    images_folder_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train';
    image_names = dir(fullfile(images_folder_path, '*.jpg'));
    % we only save the bounding box
    boundingbox = ones(numel(image_names), 4);
    for i= 1:numel(image_names)
         image_name = image_names(i).name;
         fprintf(fileID, '%s\n', image_name);
        disp(i)
        scaleVector = 1;  
        image_name = image_names(i).name;
        image_path = fullfile(images_folder_path, image_name);
        image = imread(image_path);
        [ds, bs] = process_face(image, face_model.model,  ...
                        detection_threshold, nms_threshold);                       
        % if there is no detection result, we upscale the image of one octave
%         while  sum(size(ds)) == 0 && scaleVector < 4
%             scaleVector = scaleVector+1;
%             disp('resize the image by one octave')
%             image = imresize(image, 1.41);
%             [ds, bs] = process_face(image, face_model.model,  ...
%                             detection_threshold, nms_threshold);
%         end
        % we choose the best detection result
        if sum(size(ds)) > 7
            ds = ds(1,:);
            disp('multiple faces detected');
        end
        if sum(size(ds)) ==  7 % one face detected
            boundingbox(i,:) = ds(1,1:4);
        elseif sum(size(ds)) == 0  % no face detected
            boundingbox(i,:) = [0, 0, 0, 0];
            disp('NO face detected');
        end
%         result_path = fullfile(results_folder_path, [image_name, '.result.png']);
%         showsboxes_face(image, ds, result_path);
%         disp(['Created ', result_path]);
    end

    save('boundingbox_train.mat', 'boundingbox');
    fclose(fileID);
    disp('All images processed');
end

if true
    images_folder_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Validation';
    image_names = dir(fullfile(images_folder_path, '*.jpg'));
    fileID = fopen('/idiap/user/dwu/spyder/Codalab_Age_estimation/valid_matlab_squence.csv','w');

    % we only save the bounding box
    boundingbox = ones(numel(image_names), 4);

    for i= 1:numel(image_names)
        image_name = image_names(i).name;
        fprintf(fileID, '%s\n', image_name);
        disp(i)
        scaleVector = 1;  
        image_name = image_names(i).name;
        image_path = fullfile(images_folder_path, image_name);
        image = imread(image_path);
        [ds, bs] = process_face(image, face_model.model,  ...
                        detection_threshold, nms_threshold);                       
        % if there is no detection result, we upscale the image of one octave
%         while  sum(size(ds)) == 0 && scaleVector < 4
%             scaleVector = scaleVector+1;
%             disp('resize the image by one octave')
%             image = imresize(image, 1.41);
%             [ds, bs] = process_face(image, face_model.model,  ...
%                             detection_threshold, nms_threshold);
%         end
        % we choose the best detection result
        if sum(size(ds)) > 7
            ds = ds(1,:);
            disp('multiple faces detected');
        end
        
        if sum(size(ds)) == 7 % one face detected
            boundingbox(i,:) = ds(1,1:4);
        elseif sum(size(ds)) == 0  % no face detected
            boundingbox(i,:) = [0, 0, 0, 0];
            disp('NO face detected');
        end

%         result_path = fullfile(results_folder_path, [image_name, '.result.png']);
%         showsboxes_face(image, ds, result_path);
%         disp(['Created ', result_path]);
    end

    save('boundingbox_valid.mat', 'boundingbox');
    fclose(fileID);
    disp('All images processed');
end