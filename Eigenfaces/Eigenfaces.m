classdef Eigenfaces < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X_train
        Y_train
        X_test
        Y_test
        n
        n_image
        m
        m_test
        mu_orig
        k
        eig_vals
        PCA_basis
        PCAed_train
        basis_size
        PCAed_test
        Y_predict
    end
    
    methods
        function obj = Eigenfaces(X_train,Y_train,num_classes,image_dimensions)
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
            function train(obj,bool)
                %This function trains on the data. Bool  checks if m > n
                % We perform PCA and give option of forming PCA basis and                 
                % projecting data on PCA basis 
                if (bool)
                    B = obj.X_train' * obj.X_train;
                    [obj.PCA_basis,obj.eig_vals] = eig(B); 
                else
                    B = obj.X_train * obj.X_train';
                    [obj.PCA_basis,obj.eig_vals] = eig(B); 
                    obj.PCA_basis =  obj.X_train' * obj.PCA_basis;
                end    
            end
            function eigenvalues = show_and_plot_eig_vals(obj)
                obj.eig_vals = diag(obj.eig_vals);
                eigenvalues = obj.eig_vals;
                plot(obj.eig_vals);
            end
            function Give_Basis_Dim(obj,reduction)
                obj.basis_size = reduction;
                obj.PCA_basis = obj.PCA_basis(:,end-reduction+1:end);
                obj.PCAed_train = obj.X_train * obj.PCA_basis;
            end
            function give_test_data(obj,X_test,Y_test)
                obj.X_test = X_test;
                obj.Y_test = Y_test;
                obj.m_test = size(obj.X_test);
                obj.m_test = obj.m_test(1);
                for i=1:obj.m_test
                    obj.X_test(i,:) = obj.X_test(i,:) - obj.mu_orig;
                end
            end
            function plot_eigenfaces(obj,count)
                rows = ceil(count/3);
                fig = figure();
                for i = 1:count
                    subplot(rows,3,i), imshow(uint8(reshape(obj.PCA_basis(i,:),32,32)));
                end
            end
            function accuracy = test_and_give_accuracy(obj)
                obj.PCAed_test = obj.X_test * obj.PCA_basis;
                
                %obj.Y_predict = knnsearch(obj.LDAed_mu_i,obj.LDAed_test);
                
                %Using knnsearch can be avoided by using pdist2 function
                %and sort function and some sampling
                
                %Since Statistics_Toolbox is not always there, a bruteforce
                %knnsearch using loops!
                obj.Y_predict = zeros(obj.m_test,1);
                for i=1:obj.m_test
                    min_norm = norm(obj.PCAed_test(i,:)-obj.PCAed_train(1,:));
                    obj.Y_predict(i) = 1;
                    for j=1:obj.m
                        curr_norm = norm(obj.PCAed_test(i,:)-obj.PCAed_train(j,:));
                        if (curr_norm < min_norm)
                            min_norm = curr_norm;
                            obj.Y_predict(i) = obj.Y_train(j);
                        end
                    end
                end
                
                accuracy = sum(obj.Y_predict==obj.Y_test);
            end
    end
    
end

