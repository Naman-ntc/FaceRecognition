classdef Fisherfaces < handle
    %FISHERFACES : A class that takes input training data as class
    %attribute (or property as called here). Contains function that do
    %classification of input image
    properties
    X_train
    mu_orig
    train_mu_i
    Y_train
    X_test
    Y_test
    m
    m_i
    m_test
    n
    k
    n_image
    PCA_basis
    PCAed_train
    PCAed_mu
    PCAed_mu_i
    LDA_basis
    LDAed_train
    LDAed_mu
    LDAed_mu_i
    LDAed_test
    Y_predict
    end
    
    methods
            function obj = Fisherfaces(X_train,Y_train,num_classes,image_dimensions)
                %X_train are image dimension array mxn and Y_train => mx1
                %Y_train assumes its values between 1 and k (inclusive obviously)
                obj.X_train = X_train;
                obj.Y_train = Y_train;
                obj.mu_orig = mean(X_train,1);
                temp_size = size(X_train);
                obj.m = temp_size(1);
                obj.n = temp_size(2);
                obj.k = num_classes;
                for i=1:obj.m
                    obj.X_train(i,:) = obj.X_train(i,:) - obj.mu_orig;
                end
                obj.n_image = image_dimensions;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            function train(obj)
                
                %First we'll do a PCA on the input images and select the
                %first m - k eigenvectors of the covariance matrix
                B = obj.X_train * obj.X_train'; % B has dimensions mxm
                [V,~] = eig(B);
                first_reduction = obj.m - obj.k;
                obj.PCA_basis = V(:,end-first_reduction+1:end);
                obj.PCA_basis =  obj.X_train' * obj.PCA_basis; % PCA_basis has dimensions n x m-k
                obj.PCAed_train = obj.X_train * obj.PCA_basis; % PCAed_train has dimensions m x m-k
                %Now we will perform LDA. We will find 2 matrices S_b & S_w
                %select the first k-1 generalized eigenvectors of S_b and S_w
                %S_b and S_w will have dimensions m-k x m-k
                S_b = zeros(obj.m-obj.k,obj.m-obj.k);
                S_w = zeros(obj.m-obj.k,obj.m-obj.k);
                obj.PCAed_mu = mean(obj.PCAed_train,1);
                for i = 1:obj.k
                    obj.m_i(i) = sum(obj.Y_train==i);% m_i stores number of samples of class i
                    obj.PCAed_mu_i(i,:) = mean(obj.PCAed_train(obj.Y_train==i,:),1);
                    temp_S_b = obj.PCAed_mu_i(i,:) - obj.PCAed_mu;
                    S_b = S_b + obj.m_i(i) * (temp_S_b'*temp_S_b);
                    S_w = S_w + obj.m_i(i) * cov(obj.PCAed_train(obj.Y_train==i,:));
                end
                [V,~] = eig(S_b,S_w);
                second_reduction = obj.k-1;
                obj.LDA_basis = V(:,end-second_reduction+1:end);
                obj.LDA_basis = obj.PCA_basis * obj.LDA_basis;% LDA_basis has dimension  n x k-1
                obj.LDAed_train = obj.X_train * obj.LDA_basis;
                obj.LDAed_mu = mean(obj.LDAed_train,1);
                for i = 1:obj.k
                   obj.LDAed_mu_i(i,:) = mean(obj.LDAed_train(obj.Y_train==i,:),1);
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            function train_LDA(obj)
                S_b = zeros(obj.n,obj.n);
                S_w = zeros(obj.n,obj.n);
                for i = 1:obj.k
                    obj.m_i(i) = sum(obj.Y_train==i);% m_i stores number of samples of class i
                    obj.train_mu_i(i,:) = mean(obj.X_train(obj.Y_train==i,:),1);
                    temp_S_b = obj.train_mu_i(i,:);
                    S_b = S_b + obj.m_i(i) * (temp_S_b'*temp_S_b);
                    S_w = S_w + obj.m_i(i) * cov(obj.X_train(obj.Y_train==i,:));
                end
                [V,~] = eig(S_b,S_w);
                second_reduction = obj.k-1;
                obj.LDA_basis = V(:,end-second_reduction+1:end); % LDA_basis has dimension  n x k-1
                obj.LDAed_train = obj.X_train * obj.LDA_basis;
                obj.LDAed_mu = mean(obj.LDAed_train,1);
                for i = 1:obj.k
                   obj.LDAed_mu_i(i,:) = mean(obj.LDAed_train(obj.Y_train==i,:),1);
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
            function give_test_data(obj,X_test,Y_test)
                obj.X_test = X_test;
                obj.Y_test = Y_test;
                obj.m_test = size(obj.X_test);
                obj.m_test = obj.m_test(1);
                for i=1:obj.m_test
                    obj.X_test(i,:) = obj.X_test(i,:) - obj.mu_orig;
                end
            end
            function accuracy = test_and_give_accuracy(obj)
                obj.LDAed_test = obj.X_test * obj.LDA_basis;
                
                %obj.Y_predict = knnsearch(obj.LDAed_mu_i,obj.LDAed_test);
                
                %Using knnsearch can be avoided by using pdist2 function
                %and sort function and some sampling
                
                %Since Statistics_Toolbox is not always there, a bruteforce
                %knnsearch using loops!
                obj.Y_predict = zeros(obj.m_test,1);
                for i=1:obj.m_test
                    min_norm = norm(obj.LDAed_test(i,:)-obj.LDAed_mu_i(1,:));
                    obj.Y_predict(i) = 1;
                    for j=1:obj.k
                        curr_norm = norm(obj.LDAed_test(i,:)-obj.LDAed_mu_i(j,:));
                        if (curr_norm < min_norm)
                            min_norm = curr_norm;
                            obj.Y_predict(i) = j;
                        end
                    end
                end
                
                accuracy = sum(obj.Y_predict==obj.Y_test);
            end
            function plot_3_nearest(obj)
                %ensure X_test is not to big or you'll be screwed
                obj.LDAed_test = obj.X_test * obj.LDA_basis;
                obj.Y_predict = knnsearch(obj.LDAed_mu_i,obj.LDAed_test);
                fig = figure;
                for i = 1:obj.m_test
                    temp = obj.LDAed_train(obj.Y_train==obj.Y_predict(i),:);
                    nearest_neigbours = knnsearch(temp,obj.LDAed_test(i,:),'K',3);
                    nearest_neigbours = temp(nearest_neigbours,:);
                    nearest_neigbours = nearest_neigbours*obj.LDA_basis';
                    subplot(obj.m_test,4,4*(i-1)+1), imshow(uint8(reshape(obj.X_test(i,:),obj.n_image)));
                    subplot(obj.m_test,4,4*(i-1)+2), imshow(uint8(reshape(nearest_neigbours(1,:),obj.n_image)));
                    subplot(obj.m_test,4,4*(i-1)+3), imshow(uint8(reshape(nearest_neigbours(2,:),obj.n_image)));
                    subplot(obj.m_test,4,4*(i-1)+4), imshow(uint8(reshape(nearest_neigbours(3,:),obj.n_image)));
                end
            end
    end
end 

