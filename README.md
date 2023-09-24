# Counting_People_in_Crowds
## Objective
This is a project for crowd counting using density maps. In this project, we utilize the Shanghai dataset, employ various data augmentation techniques, and create several deep learning models for crowd counting. I will also introduce a custom loss function that combines mean squared error (MSE) and structural similarity index (SSIM) for improved accuracy.

## To view the Notebook
Download the notebook or clone the repo using the following command:

``` git clone https://github.com/alyraafat/Counting_People_in_Crowds```

## Dataset
I used the Shanghai dataset for crowd counting from kaggle. The dataset contains images with labeled points representing the heads of people in each image. Gaussian distributions are used to represent these points.

## Data Augmentation

I have implemented various data augmentation techniques, including:

1. Normal dataset without augmentation.
2. Dataset with Gaussian filtering using SciPy.
3. Dataset with standard augmentation techniques like flipping, brightness adjustment, and rotation.
4. New data augmentation technique involving patch extraction, shuffling, and mixing to create new images and labels.

## Data Generators
I have implemented custom data generators that inherit from tf.keras.utils.Sequence to efficiently load and preprocess data during training.

## Models

I have designed and implemented multiple crowd counting models:

1. Two autoencoders with different encoder and decoder architectures. These autoencoders are used separately.
1. A CSRNet-like model based on VGG16 for transfer learning.
3. A multi-column model inspired by the Inception model.
4. Another autoencoder using InceptionV3 as the encoder and a custom decoder composed of Conv2DTranspose layers.

##  Loss Function 

1. Gaussian Kernel: I created a Gaussian kernel, which is a mathematical function that resembles a bell curve. This kernel is used to smooth the density maps.
2. Smoothing: We apply this Gaussian kernel to both the ground truth density map (y_true) and the predicted density map (y_pred). This smoothing process helps to make the density maps more continuous and less noisy.
3. Mean and Variance: We calculate the mean (average) and variance (spread) of the smoothed density maps for both the ground truth and predicted maps. This allows us to understand the distribution of points in the images.
4. Covariance: We also compute the covariance, which measures how two sets of data (in this case, the ground truth and predicted density maps) change together. It helps us understand how well the predicted points match the actual points.
5. Structural Similarity (SSIM): SSIM is a metric that compares two images (density maps, in our case) to assess their similarity. We use SSIM to measure how well the predicted density map matches the ground truth map. Higher SSIM indicates a better match.
6. Consistency Loss: We calculate a consistency loss based on SSIM. This loss quantifies how well the predicted density map matches the ground truth map in terms of structure and similarity. We want this loss to be as low as possible, indicating a close match.
7. Mean Squared Error (MSE): MSE is a standard loss metric used in deep learning to measure the average squared difference between the predicted and actual values. In our case, it assesses how well the predicted density map matches the ground truth map.
8. Combined Loss: Finally, we combine the MSE loss and the consistency loss. The MSE loss focuses on the accuracy of the predicted points, while the consistency loss ensures that the predicted density map is structurally similar to the ground truth map. By combining these two losses, we create an overall loss function that guides the training of the model.

The goal of this custom loss function is to train the crowd counting model to produce accurate density maps while also ensuring that the structure and similarity of these maps closely match the ground truth maps. This helps in achieving better accuracy in crowd counting tasks.

## Limitations

During the course of this project, we encountered several limitations:

1. Resource Exhaustion: Dealing with GPU resources can be challenging, especially when working with large datasets and complex models. To address this, we had to downsample the images significantly to reduce their size. This downsampling can lead to a loss of fine-grained details and may affect the accuracy of the models. However, the dataset itself was not that large but the augmentation and preprocessing techniques as well as the models were a bit computationally expensive.
2. Computational Intensity: The use of custom data generators, while beneficial for handling large datasets efficiently, can be computationally intensive. Training models with data generators may require longer training times and substantial GPU resources.

It's important to keep these limitations in mind when working with this project, especially when considering the hardware and computational resources needed for training and evaluation. Depending on your specific setup and dataset size, you may need to adjust parameters and strategies to overcome these limitations effectively.Therefore, I do plan to enhance this project in the near future to get better results.
