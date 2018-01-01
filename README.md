# FaceRecognition
A face recognition system is a computer application capable of identifying or verifying a person from a digital image or a video frame from a video source.
A simple approach can be to do a nearest neighbours from all the faces in the training data.
But a facial image can have huge number of features as huge number of pixels and lots of faces in database.
So we project the faces in a lower dimensional space, thus we need dimensionality reduction algorithms.

PCA and LDA are two good dimensionality reduction algorithms and provide good results.

### In this Repo

* [Fisherfaces](FisherFaces)
* [Eigenfaces](EigenFaces)