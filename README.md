# MusicCoverAnalysis
The project focuses on analyzing the relationship between the dominant color on music album covers and their corresponding music genres using machine learning techniques. Models were developed using image processing techniques for feature extraction, followed by machine learning algorithms for classification. The results were then compared with survey responses.

A dataset of 30,000 album covers was retrieved from the Last.fm API and normalized to ensure standardization in format and quality for further processing.

# Key Features
### Feature Extraction
Feature extraction uses K-means clustering for dominant colors, average RGB values, and HSV color space analysis, with a weighting map prioritizing image center pixels. Extracted features, genre labels, and filenames are stored in compressed .npz files for use in machine learning.
### Data Processing
The processing includes normalizing RGB/HSV values and scaling percentage features to ensure consistency. Dimensionality reduction techniques, such as PCA and t-SNE, are applied to simplify the feature space and visualize patterns, highlighting the relationships between color features and music genres.
### Models Development
The model development involves implementing SVM, Random Forest, XGBoost, and DNN, with hyperparameter optimization using GridSearch, Optuna, and Hyperband to improve classification performance. The best models are saved and shared in the repository.
### Survey Comparison
The survey application is built using the MERN stack. It supports functionalities like dynamic question navigation, color picker integration, and data storage in MongoDB, ensuring efficient user interaction and data collection.


# Author
JMacioch

