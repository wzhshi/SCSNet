% clear all; close all;

addpath('E:\wzhshi_code\ScalableCSNet\data\utilities');
run('D:\AcademicSoftware\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;

ratio = '01'; % coarse granular corresponds to the sampling ratons of 0.01 0.05 0.1 0.2 0.3 0.4 and 0.5 in the pretrained model.

netfolder = '.\model';
res01 = zeros(200,2);
res02 = zeros(200,2);
res03 = zeros(200,2);
res04 = zeros(200,2);
res05 = zeros(200,2);
res06 = zeros(200,2);
res07 = zeros(200,2);

netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));
net = load(fullfile(netfolder,netpaths(1).name));
net = dagnn.DagNN.loadobj(net.net);
 
showResult  = 0;
useGPU      = 1;
pauseTime   = 0;
if useGPU
%         net1.move('gpu');
    net.move('gpu');
end
 
folderTest = 'E:\wzhshi_code\ScalableCSNet\TempTest';%E:\datasets\Set5';
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filepaths   =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folderTest,ext{i})));
end


PSNRs01 = zeros(1,length(filepaths));
SSIMs01 = zeros(1,length(filepaths));
PSNRs02 = zeros(1,length(filepaths));
SSIMs02 = zeros(1,length(filepaths));
PSNRs03 = zeros(1,length(filepaths));
SSIMs03 = zeros(1,length(filepaths));
PSNRs04 = zeros(1,length(filepaths));
SSIMs04 = zeros(1,length(filepaths));
PSNRs05 = zeros(1,length(filepaths));
SSIMs05 = zeros(1,length(filepaths));
PSNRs06 = zeros(1,length(filepaths));
SSIMs06 = zeros(1,length(filepaths));
PSNRs07 = zeros(1,length(filepaths));
SSIMs07 = zeros(1,length(filepaths));

for i = 1:length(filepaths)
    image = imread(fullfile(folderTest,filepaths(i).name));
    if size(image,3)==3
    image = rgb2ycbcr(image);
    image = im2single(image(:, :, 1));
    
    else
        image =im2single(image);
    end
    image = modcrop(image,32); 
    if useGPU

        input = gpuArray(image);

    else
        input = image;

    end
    label = image;
    net.conserveMemory = false;

    net.eval({'input',input});

    if useGPU
%         output01 = gather(net.vars(net.getVarIndex('s01dr_pred')).value);% + gather(net.vars(net.getVarIndex('s02dr_pred')).value);
%         output02 = gather(net.vars(net.getVarIndex('s02dr_pred')).value);

        output03 = gather(net.vars(net.getVarIndex('s03dr_pred')).value);
%         output04 = gather(net.vars(net.getVarIndex('s04dr_pred')).value);
%  output05 = gather(net.vars(net.getVarIndex('s05dr_pred')).value);
%  output06 = gather(net.vars(net.getVarIndex('s06dr_pred')).value);
%         output07 = gather(net.vars(net.getVarIndex('s07dr_pred')).value);
    else
%         output01 = net.vars(net.getVarIndex('s01dr_pred')).value;
%         output02 = net.vars(net.getVarIndex('s02dr_pred')).value;
        output03 = net.vars(net.getVarIndex('s03dr_pred')).value;
%         output04 = net.vars(net.getVarIndex('s04dr_pred')).value;
%         output05 = net.vars(net.getVarIndex('s05dr_pred')).value;
%         output06 = net.vars(net.getVarIndex('s06dr_pred')).value;
%         output07 = net.vars(net.getVarIndex('s07dr_pred')).value;
    end
    
%     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output01),im2uint8(label),0,0);
%     PSNRs01(i) = PSNRCur;
%     SSIMs01(i) = SSIMCur; 
%     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output02),im2uint8(label),0,0);
%     PSNRs02(i) = PSNRCur;
%     SSIMs02(i) = SSIMCur; 
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output03),im2uint8(label),0,0);
    PSNRs03(i) = PSNRCur;
    SSIMs03(i) = SSIMCur; 
%     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output04),im2uint8(label),0,0);
%     PSNRs04(i) = PSNRCur;
%     SSIMs04(i) = SSIMCur; 
%     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output05),im2uint8(label),0,0);
%     PSNRs05(i) = PSNRCur;
%     SSIMs05(i) = SSIMCur; 
%     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output06),im2uint8(label),0,0);
%     PSNRs06(i) = PSNRCur;
%     SSIMs06(i) = SSIMCur; 
%     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output07),im2uint8(label),0,0);
%     PSNRs07(i) = PSNRCur;
%     SSIMs07(i) = SSIMCur; 
   if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filepaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
        pause(pauseTime)
    end 
     

end
% res01(iii,:) = [mean(PSNRs01),mean(SSIMs01)];
% disp([mean(PSNRs01),mean(SSIMs01)])
% res02(iii,:) = [mean(PSNRs02),mean(SSIMs02)];
% disp([mean(PSNRs02),mean(SSIMs02)])
res03(iii,:) = [mean(PSNRs03),mean(SSIMs03)];
disp([mean(PSNRs03),mean(SSIMs03)])
% res04(iii,:) = [mean(PSNRs04),mean(SSIMs04)];
% disp([mean(PSNRs04),mean(SSIMs04)])
% res05(iii,:) = [mean(PSNRs05),mean(SSIMs05)];
% disp([mean(PSNRs05),mean(SSIMs05)])
% res06(iii,:) = [mean(PSNRs06),mean(SSIMs06)];
% disp([mean(PSNRs06),mean(SSIMs06)])
% res07(iii,:) = [mean(PSNRs07),mean(SSIMs07)];
% disp([mean(PSNRs07),mean(SSIMs07)]);
