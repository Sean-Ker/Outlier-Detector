# Multidimensional Outlier Detector
A collection of of Jupyter Notebook files that implement an outlier detector in an Rn space.

The detector is implemented in two distinct ways. 

The first mehtod is more robust and sound than the second implementation. It uses PCA to dimensionally reduce the data set to a lower dimension. Then, Mahalanobis Distance is used to filter out the outlier in the dataset with a certain threshold. [Learn more.](https://www.youtube.com/watch?v=spNpfmWZBmg&t=0s)

The second method uses the a Neural Network Autoencoder to filter out data points which greatly contribute to increase of the loss with a certain thershold.

Note: The demos follow the Mahalanobis Distance Implementation.

## Screenshots
### Sample Date Covariance Matrix
![Sample Date Covariance Matrix](/assets/covariance_matrix.png?raw=true)

### 2D Outlier Detector Demo
![2D Outlier Detector Demo](/assets/boston_demo.jpg?raw=true)

### 2D Outlier Detector Demo: Principal Component Analysis
![2D Outlier Detector Demo PCA](/assets/boston_PAC_demo.jpg?raw=true)

## Other Projects
[Casino Assignment for the course (Krabby Patty)](https://github.com/Sean-Ker/data_patty_hunter)
