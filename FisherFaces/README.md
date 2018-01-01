# Fisherfaces for face recognition
Linear Discriminant Analysis is a nice algorithms when we want to perform dimensionality reduction on data set with labels. We try to maximize the within class scatter of samples, whereas in PA we maximize total scatter of samples

### Usage
```matlab
%load training data

% Accuracies = [];

rand = randperm(2414);
test = rand(1:300);
train = rand(301:2414);
X_train = fea(train,:); X_test = fea(test,:); Y_train = gnd(train); Y_test = gnd(test);
Model = Fisherfaces(X_train,Y_train,38,[32 32]);
Model.train_LDA();
Model.give_test_data(X_test,Y_test);
Accuracies(end+1) = Model.test_and_give_accuracy();


```

### Results
The model Achivels above 96% accuracy on the Extended Yale database!