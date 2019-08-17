function net = SCSNet_init( varargin )
%CNN_DDCN_INIT Summary of this function goes here
%   Detailed explanation goes here

net = dagnn.DagNN();
d=128;
s=32;
rng('default');
rng(0) ;

reluLeak = 0;
% bnormal =false;
net.meta.solver = 'Adam';
net.meta.inputSize = [96 96] ;
% net.meta.trainOpts.weightDecay = 0.0001 ;
% net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 32;
net.meta.trainOpts.learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.adjGradClipping = false;
% net.meta.gradthresh = 0.005;

net.meta.derOutputs = {'s01pdist',1,'pdistInits01',1,'s02pdist',1,'pdistInits02',1,'s03pdist',1,'pdistInits03',...
1,'s04pdist',1,'pdistInits04',1,'s05pdist',1,'pdistInits05',1,'s06pdist',1,'pdistInits06',1,'s07pdist',1,'pdistInits07',1};
%for raw

net.meta.imdbPath ='E:\wzhshi_code\ScalableCSNet\data\model_64_96_Adam-1400\imdb.mat';
%-------------------------------------------------------------------
%basic layer

% sampling
block = dagnn.Conv('size',  [32 32 1 10], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's01sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction
block = dagnn.Conv('size',  [1 1 10 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's01initRecon';
net.addLayer(lName, block, 's01sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's01combine';
net.addLayer(lName,block,'s01initRecon',lName);

% %---------------------------------------------------------------------------
%enhancement layer1
% sampling
block = dagnn.Conv('size',  [32 32 1 41], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's02sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction

% net.addLayer('s02samplingE',dagnn.Concat('dim',3),{'s01sampling','s02sampling'},'s02samplingE');

block = dagnn.Conv('size',  [1 1 41 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's02initRecon';
net.addLayer(lName, block, 's02sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's02combine';
net.addLayer(lName,block,'s02initRecon',lName);


% %---------------------------------------------------------------------------
%enhancement layer2
% sampling
block = dagnn.Conv('size',  [32 32 1 51], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's03sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction

% net.addLayer('s02samplingE',dagnn.Concat('dim',3),{'s01sampling','s02sampling'},'s02samplingE');

block = dagnn.Conv('size',  [1 1 51 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's03initRecon';
net.addLayer(lName, block, 's03sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's03combine';
net.addLayer(lName,block,'s03initRecon',lName);

%---------------------------------------------------------------------------
%enhancement layer3
% sampling
block = dagnn.Conv('size',  [32 32 1 103], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's04sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction

% net.addLayer('s02samplingE',dagnn.Concat('dim',3),{'s01sampling','s02sampling'},'s02samplingE');

block = dagnn.Conv('size', [1 1 103 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's04initRecon';
net.addLayer(lName, block, 's04sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's04combine';
net.addLayer(lName,block,'s04initRecon',lName);


%enhancement layer4
% sampling
block = dagnn.Conv('size',  [32 32 1 102], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's05sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction

% net.addLayer('s02samplingE',dagnn.Concat('dim',3),{'s01sampling','s02sampling'},'s02samplingE');

block = dagnn.Conv('size', [1 1 102 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's05initRecon';
net.addLayer(lName, block, 's05sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's05combine';
net.addLayer(lName,block,'s05initRecon',lName);

%enhancement layer5
% sampling
block = dagnn.Conv('size',  [32 32 1 103], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's06sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction

% net.addLayer('s02samplingE',dagnn.Concat('dim',3),{'s01sampling','s02sampling'},'s02samplingE');

block = dagnn.Conv('size', [1 1 103 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's06initRecon';
net.addLayer(lName, block, 's06sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's06combine';
net.addLayer(lName,block,'s06initRecon',lName);

%enhancement layer6
% sampling
block = dagnn.Conv('size',  [32 32 1 102], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 's07sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction

% net.addLayer('s02samplingE',dagnn.Concat('dim',3),{'s01sampling','s02sampling'},'s02samplingE');

block = dagnn.Conv('size', [1 1 102 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 's07initRecon';
net.addLayer(lName, block, 's07sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 's07combine';
net.addLayer(lName,block,'s07initRecon',lName);

% 1th deep reconstruction
enl = 1;
% net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine','s03combine','s04combine','s05combine','s06combine','s07combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combine'],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combine'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combine']},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);

% 2th deep reconstruction
enl = 2;
net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combineAdds0' num2str(enl-1)], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);

% 3th deep reconstruction
enl = 3;
net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine','s03combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combineAdds0' num2str(enl-1)], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);

% 4th deep reconstruction
enl = 4;
net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine','s03combine','s04combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combineAdds0' num2str(enl-1)], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);

% 5th deep reconstruction
enl = 5;
net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine','s03combine','s04combine','s05combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combineAdds0' num2str(enl-1)], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);


% 6th deep reconstruction
enl = 6;
net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine','s03combine','s04combine','s05combine','s06combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combineAdds0' num2str(enl-1)], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);


% 7th deep reconstruction
enl = 7;
net.addLayer(['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],dagnn.Sum(),{'s01combine','s02combine','s03combine','s04combine','s05combine','s06combine','s07combine'},['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]);
net.addLayer(['pdistInits0' num2str(enl)],dagnn.EuclidLoss(),{['s0' num2str(enl) 'combineAdds0' num2str(enl-1)],'label'},['pdistInits0' num2str(enl)]);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, ['s0' num2str(enl) 'combineAdds0' num2str(enl-1)], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01dr_pred'},[lName1 '_cat']);

i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combineAdds0' num2str(enl-1)]},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);

net.initParams();