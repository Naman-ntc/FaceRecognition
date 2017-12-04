load('faces_database.mat');

faces = reshape(faces,280800,100);
mean_face = mean(faces,2);
%fig = image(uint8(reshape(mean_face,360,260,3)));
%saveas(fig,'mean_face.jpg');
for i=1:100
    faces(:,i)   = faces(:,i) - mean_face;
end
matrix = faces.' * faces;
[vectors values] = eig(matrix);
values = diag(values);
%fig = plot(1:100,values);
%saveas(fig,'eigenvalues.jpg');
%%Now we will take the top 12 eigenvectors of matrix

eigenvectors = faces * vectors(:,88:100);

%fig = figure;    
%for i =1:12
%    subplot(3,4,i) ,imshow(uint8(reshape(eigenvectors(:,i),360,260,3)))
%end
%saveas(fig,'eigenfaces.jpg');

face_space = zeros(100,12);
face_space = faces.' * eigenvectors;

load('test.mat');

test = reshape(test,280800,10);


for i = 1:10
    test(:,i) = test(:,i) - mean_face;
end

%test' ==> 10 x 280800
%faces' ==> 100 x 280800

test_space = zeros(10,12);
test_space = test.' * eigenvectors;

%city_block_dist = zeros(10,100);
%euclidean_dist = zeros(10,100);
%mahalanobis_dist = zeros(10,100);

city_block_dist = pdist2(test_space,face_space,'cityblock');
euclidean_dist = pdist2(test_space,face_space);

city_block_faces = zeros(10,100);
euclidean_faces = zeros(10,100);

for i=1:10
    [rand , city_block_faces(i,:)] = sort(city_block_dist(i,:));
    [rand , euclidean_faces(i,:)] = sort(euclidean_dist(i,:));
end

city_block_faces = city_block_faces(:,1:3);
euclidean_faces = euclidean_faces(:,1:3);


for i=1:100
    faces(:,i)   = faces(:,i) + mean_face;
end

for i = 1:10
    test(:,i) = test(:,i) + mean_face;
end


%fig = figure;
%for i=1:5
%subplot(5,4,4*(i-1)+1) , imshow(uint8(reshape(test(:,i),360,260,3)));
%hold on;
%subplot(5,4,4*(i-1)+2) , imshow(uint8(reshape(faces(:,euclidean_faces(i,1)),360,260,3)));
%hold on;
%subplot(5,4,4*(i-1)+3) , imshow(uint8(reshape(faces(:,euclidean_faces(i,2)),360,260,3)));
%hold on;
%subplot(5,4,4*(i-1)+4) , imshow(uint8(reshape(faces(:,euclidean_faces(i,3)),360,260,3)));
%hold on;
%end
%saveacs(fig,'eucledian.jpg');

%fig = figure;
%for i=1:5
%subplot(5,4,4*(i-1)+1) , imshow(uint8(reshape(test(:,i),360,260,3)));
%hold on;
%subplot(5,4,4*(i-1)+2) , imshow(uint8(reshape(faces(:,city_block_faces(i,1)),360,260,3)));
%hold on;
%subplot(5,4,4*(i-1)+3) , imshow(uint8(reshape(faces(:,city_block_faces(i,2)),360,260,3)));
%hold on;
%subplot(5,4,4*(i-1)+4) , imshow(uint8(reshape(faces(:,city_block_faces(i,3)),360,260,3)));
%hold on;
%end
%saveas(fig,'city_block.jpg');