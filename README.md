# PCA_Autoencoder_FisherFace

## Eigen Faces

### PCA for the faces

A traditional way to analyze the genral infromation of the data would be using PCA to project the dataset into lower dimensions extracting the general infromation. For the faces data, we could using this way to see how what are the features to construct the face. The faces condensed in lower dimensions would be the eigen faces. Below are the 10 egien faces of the celebrity faces data used for this project.

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/10eigenfaces.png" width="400px"</img> 
</div>

We could use these 10 eigen faces (PCA) to reconstruct the original face images. Here are 10 examples. Let's compare them with the original faces.

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/10ReconstructFace.png" width="400px"</img> 
</div>

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/Original10faces.png" width="400px"</img> 
</div>

The reconstructed faces losing some detailed information, which cause the blurness.

The reconstruction error per pixel over the number of eigen-faces K=1,5,10,15,...,50:

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/1.png" width="400px"</img> 
</div>

We can see that as the number of principle components increase, the error rate decreases. And for the first 5 principle components, it decreases the most.

### PCA for the faces geometry

The reconstruction error per pixel over the number of eigen-warping K=1,5,10,15,...,50:

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/2.png" width="400px"</img> 
</div>

We can see that as the number of principle components increase, the error rate decreases.

Combine the 10 eigen geometries into one plot, we can clearaly see how the face geometrices move. This is the first 10 eigen warping.

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/eigenWarpings.png" width="400px"</img> 
</div>

### Random synthesized fac images by Eigen faces and geometrices

We just do the random synthesize for the V channel (grey images). Here are 50 examples.

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/Random50.png" width="400px"</img> 
</div>

## AutoEncoder for faces

### Direct reconstruction of faces

The whole work would contain two parts, the first part is to align the images in to the mean geometry and then train the 800 face images and 800 face geometries separately.

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/Autoencoder_Reconstruct_Faces.png" width="400px"</img> 
</div>

### Random Generation by modifying the latent values

Use the latent variables returned when training (Appearance: 800x50, Geometry 800x10), we could calculate the mean and the variance for each column vectors first. And then according to these variances, pick the top 4 variances vectors for Appearance and top 2 variances vectors for Geometry. Hold the rest dimensions fixed and generate one dimension by a random normal value with the empirical mean of this latent vector and the empirical variance of this latent vector times an amplifier constant (decide by our self to make the difference a little more clear, for appearance is 4, for geometry is 100) every time.

The chosen image for the Interpolation:

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/Autoencoder_Origin.png" width="200px"</img> 
</div>

The random generated faces:

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/Autoencoder_random1.png" width="400px"</img>
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/Autoencoder_random2.png" width="400px"</img>
</div>

## Fisher faces for gender discrimination

To do the Linear Discriminant Analysis on the HSV face images V channel, first we need to reshape images (aligned in center geometry) V channel as a row vector. We do the same for the geometry. However, because the within covariance matrix would be too large, we chose to do a PCA first and chose the first 50 principles for appearance and first 10 principles for geometry. Then we perform two LDAs on these two datasets. After solving the generalized eigen problem, we could find the eigen vectors and eigen values for the W matrix. Because we only have 2 classes (male and female), there will be only one eigen vector with non-zero eigenvalue, which is the Linear Discriminant boundary we want. Then to do the prediction, we just use the reduced-dimension test matrix dot product this eigen vector to project the data to that boundary hyperplane. In my case, if the projection is positive, then it is classified as male and if it is negative, it is classified as female.

The Fisher Face that distinguish male from female by training set:

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/FisherFace.png" width="200px"</img> 
</div>

The Classification Error Rate is 12.5%, the Accuracy is 87.5%.

The projection in the 2D feature space:

<div align="center">
        <img src="https://github.com/nji3/PCA_Autoencoder_FisherFace/blob/master/readme_images/2Dspaceprojection.png" width="400px"</img> 
</div>
