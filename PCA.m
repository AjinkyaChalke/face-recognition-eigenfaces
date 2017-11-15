function PCA()
	allImages = 'FaceRecognition_Data/ALL/*.TIF';
	allImageFolder = 'FaceRecognition_Data/ALL/';
	trainingImages = 'FaceRecognition_Data/FA/*.TIF';
	trainingImageFolder = 'FaceRecognition_Data/FA/';
	testImages = 'FaceRecognition_Data/FB/*.TIF';
	testImageFolder = 'FaceRecognition_Data/FB/';
	numberOfEigenVectors = 10;

	%Read images from files, vectorize and load it in a matrix
	trainingFiles = dir(allImages);
	numberOfTrainingFiles = size(trainingFiles,1);
	trainingImageMatrix = zeros(1024, numberOfTrainingFiles );

	for index = 1 : numberOfTrainingFiles
	    file = trainingFiles(index).name;
	    imageMatrix = imread(strcat(allImageFolder,file));
	    trainingImageMatrix(:,index) = imageMatrix(:);
	end

	%Performing PCA
	m = mean(trainingImageMatrix,2);

	%show the mean image
	figure('Name','Mean Image','NumberTitle','off');
	colormap('gray');
	imshow((reshape(m,32,32)),[]);

	S = (trainingImageMatrix - m)*(trainingImageMatrix-m)';
	[V , D] = eig(S);
	eValues = diag( D );
	[~ , indices] = sort( eValues, 'descend' );
	sorteddEigenVectors = V(:,indices);
	W = sorteddEigenVectors(:, [1:numberOfEigenVectors]);

	%show the top eigen face image
	figure('Name','Eigen Image','NumberTitle','off');
	colormap('gray');
	imagesc((reshape(W(:,1),32,32)));


	%Read images from training folder, vectorize and load it in a matrix
	modelTrainingFiles = dir(trainingImages);

	for index = 1 : size(modelTrainingFiles,1)
	    file = modelTrainingFiles(index).name;
	    fileName = strcat(trainingImageFolder,file);
	    imageMatrix = imread(fileName);
	    knownFaceImages(:,index) = double(imageMatrix(:));
	end

	for index = 1:size(knownFaceImages,2)
		knownFaceDataBase(:,index) = W'*(knownFaceImages(:,index) - m);
	end

	%Test the images from testing folder
	testFiles = dir(testImages);
	numberOfTestFiles = size(testFiles,1);

	for index = 1:numberOfTestFiles
		testImageName = testFiles(index).name;
		testFileName = strcat(testImageFolder,testImageName );
		testFile = imread(testFileName);
		testImage = double(testFile(:));
		y = W'*(testImage - m);

		%Calculate Euclidean distance
		for index_euclidean = 1:size(knownFaceDataBase,2)
			euclideanDist(index_euclidean) = sqrt(sum((knownFaceDataBase(:,index_euclidean) - y) .^ 2));
		end

		%Get the result, store it and visualize the images.
		resultCell(index,1) = cellstr(testImageName);
		%figure('Name','Plot of individual results','NumberTitle','off');
		%colormap('gray');
		%subplot(1,4,1), imshow((reshape( testImage,32,32)),[]);
		%title(extractBetween(testImageName,'small_','.TIF'));
		for counter = 2:4
			[~ , index_min] = min(euclideanDist);
			resultCell(index,counter) = cellstr(modelTrainingFiles(index_min).name);
			euclideanDist(index_min) = Inf;
			%subplot(1,4,counter), imshow((reshape( knownFaceImages(:,index_min),32,32)),[]);
			%title(extractBetween(modelTrainingFiles(index_min).name,'small_','.TIF'));
		end

	end

	%Displaying top 3 results in table format
	T = cell2table(resultCell,'VariableNames',{'TestImageFileName' 'ResultantTrainingImageFileName1' 'ResultantTrainingImageFileName2' 'ResultantTrainingImageFileName3'});
	display(T)

	%Plot cumulative distribution of eigen values
	sumEValues = sum(eValues);
	cumulativeDistrb = cumsum( eValues );
	cumulativeDistrb = cumulativeDistrb / sumEValues;
	figure('Name','Plot of Cummulative Distribution','NumberTitle','off');
	plot(cumulativeDistrb);