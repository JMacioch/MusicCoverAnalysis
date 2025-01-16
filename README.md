# MusicCoverAnalysis
The project focuses on analyzing the relationship between the dominant color on music album covers and their corresponding music genres using machine learning techniques. Models were developed using image processing techniques for feature extraction, followed by machine learning algorithms for classification. The results were then compared with survey responses.

# Key Features
### Data Preprocessing
The preprocessing includes normalizing RGB/HSV values and manually scaling percentage features to ensure consistency. Dimensionality reduction techniques, such as PCA and t-SNE, are applied to simplify the feature space and visualize patterns, highlighting the relationships between color features and music genres.
### Feature Extraction
Feature extraction used K-means clustering for dominant colors, average RGB values, and HSV color space analysis, with a weighting map prioritizing image center pixels. Extracted features, genre labels, and filenames were stored in compressed .npz files for later use in machine learning.
### Models Development
The model development involves implementing SVM, Random Forest, XGBoost, and DNN, with hyperparameter optimization using GridSearch, Optuna, and Hyperband to improve classification performance. The best models are saved and shared in the repository.
### Survey Comparison
The survey application is built using the MERN stack. It supports functionalities like dynamic question navigation, color picker integration, and data storage in MongoDB, ensuring efficient user interaction and data collection.


# Author
JMacioch

