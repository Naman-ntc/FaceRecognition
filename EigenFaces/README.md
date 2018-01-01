# Eigenfaces for face recognition
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components

### Usage
```matlab
Model = Eigenface(X_train,Y_train,num_classes,image_dimensions);
Model.train({{true if m>n}});
Model.show_and_plot_eig_vals()
%Plot the eigenvalues
Model.Give_Basis_Dim(integer=number of eigenfaces to use)
Model.give_test_data(X_test,Y_test)
Model.test_and_give_accuracy()

%You can also plot eignefaces using
Model.plot_eigenfaces()
```

### Results
Below is the set of 12 eigenfaces generated

![Eigenfaces](eigenfaces.jpg)

References :
* [Turk's Paper on Eigenfaces](http://www.face-rec.org/algorithms/pca/jcn.pdf)
* [Nice Tutorial](https://onionesquereality.wordpress.com/2009/02/11/face-recognition-using-eigenfaces-and-distance-classifiers-a-tutorial/)
* [Another Nice Tutorial](http://www.maths.dur.ac.uk/users/kasper.peeters/pdf/face_recognition/reports/Barker.pdf)
* [Nice Work](https://courses.cs.washington.edu/courses/cse576/08sp/projects/project3/artifact/ankit/artifact/index.html)
