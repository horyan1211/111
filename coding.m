clc;
clear all;
outputFolder=fullfile('recycle101');
rootFolder=fullfile(outputFolder,'recycle');
categories={'can','plastic','drinkcartonbox'};

imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl=countEachLabel(imds)
minSetCount=min(tbl{:,2});
imds=splitEachLabel(imds,minSetCount,'randomize');

countEachLabel(imds);

can=find(imds.Labels=='can',1);
plastic=find(imds.Labels=='plastic',1);
drinkcartonbox=find(imds.Labels=='drinkcartonbox',1);

figure
subplot(2,2,1);
imshow(readimage(imds,can));
subplot(2,2,2);
imshow(readimage(imds,plastic));
subplot(2,2,3);
imshow(readimage(imds,drinkcartonbox));

net=resnet50();
figure
plot(net)
title('Architecture of ResNet-50')
set(gca,'YLim',[150 170]);

net.Layers(1);
net.Layers(end)

numel(net.Layers(end).ClassNames)
[trainingSet,testSet]=splitEachLabel(imds,0.3,'randomize');

imageSize=net.Layers(1).InputSize;

augmentedTrainingSet=augmentedImageDatastore(imageSize,...
    trainingSet,'ColorPreprocessing','gray2rgb');

augmentedTestSet=augmentedImageDatastore(imageSize,...
    testSet,'ColorPreprocessing','gray2rgb');

w1=net.Layers(2).Weights;
w1=mat2gray(w1);

 figure
montage(w1)
title('First Concolutional Layer Weight')

featureLayer='fc1000';
trainingFeatures=activations(net,...
    augmentedTrainingSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

trainingLabels=trainingSet.Labels;
classifier=fitcecoc(trainingFeatures,trainingLabels,...
    'Learner','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures=activations(net,...
    augmentedTestSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

predictLabels=predict(classifier,testFeatures,'ObservationsIn','columns');

testLabels=testSet.Labels;
confMat=confusionmat(testLabels,predictLabels);
confMat=bsxfun(@rdivide,confMat,sum(confMat,2));

mean(diag(confMat));

newImage=imread(fullfile('test103.png'));
%% Read in image
I=imread('test103.png');
imshow(I);
 
%% Solution: Thresholding the image
%Im=double(img)/255;
Im=I;
 
rmat=Im(:,:,1);
gmat=Im(:,:,2);
bmat=Im(:,:,3);
 
subplot (2,2,1), imshow(rmat);
title('Red Plane');
subplot(2,2,2), imshow(gmat);
title('Green Plane');
subplot(2,2,3); imshow(bmat);
title('Blue Plane');
subplot(2,2,4); imshow(I);
title('Original Plane');
 
%%
levelr=0.6;
levelg=0.3;
levelb=0.4;
i1=im2bw(rmat,levelr);
i2=im2bw(gmat,levelg);
i3=im2bw(bmat,levelb);
Isum=(i1&i2&i3);
 
%Plot the data
subplot (2,2,1), imshow(i1);
title('Red Plane');
subplot(2,2,2), imshow(i2);
title('Green Plane');
subplot(2,2,3); imshow(i3);
title('Blue Plane');
subplot(2,2,4); imshow(Isum);
title('Sum of all the plane');
 
%% Complement Image and Fill in holes
Icomp=imcomplement(Isum);
Ifilled=imfill(Icomp,'holes');
figure,imshow(Ifilled);
 
%%
se=strel('disk',1);
Iopenned=imopen(Ifilled,se);
%figure,imshowpair(Iopenned,I);
imshow(Iopenned);
 

ds=augmentedImageDatastore(imageSize,...
    newImage,'ColorPreprocessing','gray2rgb');
imageFeatures= activations(net,...
    ds,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

label= predict(classifier,imageFeatures,'ObservationsIn','columns');

sprintf('The loaded image belongs to %s class', label)
