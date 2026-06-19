# Dog Breed Classification

This project implements a machine learning pipeline for fine-grained dog breed classification using classical computer vision methods and supervised learning.

The goal is to classify dog images into different breeds using preprocessing, feature extraction, dimensionality reduction, and an SVM classifier.

## Overview

Dog breed classification is a supervised learning task where each image is associated with a label corresponding to the dog breed.

This task is challenging because some dog breeds have similar visual characteristics, images may have different backgrounds and lighting conditions, and the original images do not all have the same size.

To solve this problem, this project builds a complete machine learning pipeline including:

* Image preprocessing
* Dog localization using XML annotations
* Aspect-ratio preserving resizing
* Data matrix construction
* PCA feature extraction
* HOG feature extraction
* SVM classification
* Hyperparameter optimization with GridSearchCV
* Model evaluation using accuracy, confusion matrix, precision, recall, and F1-score

## Dataset

This project uses the `SmallDB` dataset, a reduced version of the Stanford Dogs Dataset used for educational purposes.

The dataset contains dog images from 6 classes:

* Chihuahua
* Basset
* Kerry blue terrier
* Groenendael
* Malinois
* Chow

Each image is associated with an XML annotation file containing the bounding box of the dog. These annotations are used to crop the image and focus the model on the dog instead of the background.

The class distribution is almost balanced. The Shannon entropy value obtained is:

```text
H = 1.7863
```

This value is very close to the theoretical maximum:

```text
log(6) ≈ 1.79
```

This means that the classes are almost uniformly distributed.

### Dataset Availability

The dataset is not included in this repository.

To run the project, place the `SmallDB` folder in the root directory of the project before executing the scripts.

Expected structure:

```text
dog-breed-classification/
│
├── SmallDB/
├── figures/
├── Rapport.pdf
├── Script01_PreprocessingExploration.py
├── Script02_SVC-CV.py
├── README.md
└── .gitignore
```

The `SmallDB` folder is ignored by Git using `.gitignore` because the dataset may not be allowed to be redistributed publicly.

## Preprocessing

The images are first cropped using the bounding boxes from the XML annotation files.

Then, each image is resized to:

```text
64 x 64 pixels
```

Since the original images have different sizes and aspect ratios, an aspect-ratio preserving resizing method is used. Padding is added to obtain a square image without deforming the dog.

This avoids geometric distortion and helps preserve the real shape of the animal.

## Data Matrix Construction

After preprocessing, each image is converted into a numerical vector.

Since each image has a size of `64 x 64`, each grayscale image is represented as a vector of:

```text
4096 values
```

The final data matrix contains:

* One row per image
* One column per pixel

This representation makes it possible to use classical machine learning algorithms from scikit-learn.

## Feature Extraction

Two feature extraction methods are used: PCA and HOG.

### PCA

Principal Component Analysis is used to reduce the dimensionality of the image data while keeping most of the important information.

PCA helps to:

* Reduce the number of variables
* Remove noise
* Make the classification task easier
* Visualize the structure of the dataset

The PCA analysis shows that around 90% of the variance can be captured using a limited number of components.

### HOG

Histogram of Oriented Gradients is used to extract local edge and shape information from the images.

HOG is based on image gradients and helps capture:

* Edges
* Contours
* Silhouettes
* Local structure of the dog

The combination of PCA and HOG allows the model to use both global image information and local edge-based features.

## Machine Learning Pipeline

The project uses a scikit-learn pipeline to organize the full training process.

The feature extraction step combines PCA and HOG features using `FeatureUnion`:

```python
all_features = FeatureUnion([
    ("pca_info", PCAInfoPreprocessing()),
    ("edge_info", EdgeInfoPreprocessing())
])
```

The SVM pipeline is built as follows:

```python
pipeline_svc = Pipeline([
    ("minmax_scaler", MinMaxScaler()),
    ("all_features", all_features),
    ("standard_scaler", StandardScaler()),
    ("svc", SVC(kernel="linear"))
])
```

The dataset is split into training and test sets using stratification in order to preserve the class distribution.

The similarity between the train and test class distributions is:

```text
Similarity = 0.999951
```

This value is very close to 1, which means that the train/test split is well balanced.

## Model Optimization

The hyperparameters were optimized using `GridSearchCV`.

GridSearchCV tests several external model configurations, such as:

* SVM kernel
* C value
* Gamma value

The best configuration found was:

| Kernel |  C | Gamma | CV Score |
| ------ | -: | ----: | -------: |
| RBF    | 20 | 0.005 |   57.52% |

The final model uses:

```text
Kernel = RBF
C = 20
gamma = 0.005
```

## Results

The final model obtained the following test accuracy:

```text
Accuracy = 54.98%
```

The classification report is:

| Class              | Precision | Recall | F1-score | Support |
| ------------------ | --------: | -----: | -------: | ------: |
| Chihuahua          |      0.55 |   0.47 |     0.51 |      38 |
| Basset             |      0.57 |   0.66 |     0.61 |      44 |
| Kerry blue terrier |      0.51 |   0.49 |     0.50 |      45 |
| Groenendael        |      0.51 |   0.63 |     0.56 |      38 |
| Malinois           |      0.45 |   0.46 |     0.45 |      37 |
| Chow               |      0.72 |   0.57 |     0.64 |      49 |
| Accuracy           |         - |      - |     0.55 |     251 |
| Macro avg          |      0.55 |   0.55 |     0.55 |     251 |
| Weighted avg       |      0.56 |   0.55 |     0.55 |     251 |

The results show that the model learns useful visual patterns, but some breeds remain difficult to separate because of visual similarities.

The best precision is obtained for the class `Chow`, with a precision of `0.72`.

The weakest class is `Malinois`, with an F1-score of `0.45`.

## Confusion Matrix

A confusion matrix was used to analyze the errors of the model.

The diagonal values represent correct predictions, while the values outside the diagonal represent confusion between different breeds.

The confusion matrix shows that some classes are better recognized than others, while visually similar breeds are more often confused.

## OvO vs OvR Comparison

Two multiclass SVM strategies were compared:

* One-vs-One
* One-vs-Rest

| Strategy    | Accuracy |   Time | Number of Models |
| ----------- | -------: | -----: | ---------------: |
| One-vs-One  |   55.38% | 2.22 s |               15 |
| One-vs-Rest |   55.78% | 2.19 s |                6 |

The results are very close, but One-vs-Rest is slightly better in accuracy, faster, and uses fewer models.

For 6 classes:

```text
OvO = 15 models
OvR = 6 models
```

OvR is also more scalable because its complexity is linear, while OvO has quadratic complexity.

## Project Structure

```text
dog-breed-classification/
│
├── figures/
│   └── Generated figures used in the report
│
├── Rapport.pdf
│   └── Final project report
│
├── Script01_PreprocessingExploration.py
│   └── Image loading, cropping, preprocessing, PCA and HOG feature extraction
│
├── Script02_SVC-CV.py
│   └── SVM training, cross-validation, GridSearchCV and evaluation
│
├── README.md
│
└── .gitignore
```

The `SmallDB` folder must be added locally to run the project, but it is not included in the repository.

## Installation

Install the required Python packages:

```bash
pip install numpy matplotlib scikit-learn scikit-image opencv-python
```

Depending on your environment, you may also need:

```bash
pip install pandas
```

## How to Run

1. Place the `SmallDB` folder in the root directory of the project.

2. Run the preprocessing and exploration script:

```bash
python Script01_PreprocessingExploration.py
```

3. Run the SVM training and evaluation script:

```bash
python Script02_SVC-CV.py
```

## Generated Figures

The `figures/` folder contains visual results used in the report, such as:

* Dataset class distribution
* Cropping example using XML annotations
* Resizing and padding comparison
* PCA reconstruction
* PCA explained variance
* PCA projection
* Sobel gradient visualization
* Train/test distribution comparison
* Generalization diagnostic plots
* Confusion matrix
* OvO vs OvR comparison

## Future Improvements

Possible improvements include:

* Using more training images
* Applying data augmentation
* Testing other classifiers
* Using convolutional neural networks
* Applying transfer learning with pretrained models
* Improving feature extraction and hyperparameter tuning

## Conclusion

This project demonstrates a complete classical machine learning pipeline for image classification.

PCA and HOG provide useful visual features, and SVM allows supervised classification of dog breeds.

The final accuracy of 54.98% shows that the model can learn discriminative patterns, but fine-grained dog breed classification remains difficult with classical methods.

More advanced approaches such as CNNs or transfer learning could improve the results.
