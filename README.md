<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <h2>CS-433 Machine Learning Project 1</h2>
  <h3>Predicting the likelihood of developing MICHD given indivdiuals' clinical and lifestyle situation</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#directory-description">Directory Description</a></li>
        <li><a href="#methodology-and-results">Methodology and Results</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
   </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The project is accomplished by team JEC with members:
<ul>
  <li>Peh Jin Yang: @jinyangp</li>
  <li>Celest Angela Tjong : @ninjahw</li>
  <li>Lee Ee Cheer : @leeeecheer</li> 
</ul>

Using data from the Behavioral Risk Factor Surveillance System (BRFSS), a system of health-related telephone surveys was conducted to collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. In particular, respondents were classified as having coronary heart disease (MICHD) if they reported having been told by a provider they had MICHD. Respondents were also classified as having MICHD if they reported having been told they had a heart attack (i.e.,myocardial infarction) or angina.

This project hopes to estimate the likelihood of developing MICHD given a certain clinical and lifestyle situation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Directory Description

This section gives a brief description of the content inside each folder/ file.

| Folder/ File name | Description |
| ----------- | ----------- |
| eda | Contains notebooks used to perform preliminary Exploratory Data Analysis |
| data_loader.py | Contains implementated functions used to preprocess the data, formulate the design matrix and format the labels of the response variable |
| implementations.py | Contains the 6 implemented functions required (mean_squared_error_gd, mean squared error sgd, etc.) |
| models.py | Contains implementation code for Model 1-5 |
| provided_helpers.py | Contains provided helper functions to assist the loading of data and to generate csv submission |
| run.py | Python script to generate the csv submission |
| train.ipynb | Contains implentation code used to train the model |
| utils.py | Contains helper functions to assist the implemntation of loss functions in implementations.py |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Methodology and Results


|             |         Dataset       |     Training method          |
| :---------: | --------------------- | ---------------------------- |
| Model 1     |  Original data set with 91% negative samples | Mean Squared Error with GD |
| Model 2     |  Original data set with 91% negative samples | Logistic Regression with GD |
| Model 3     |  Down sampled data set with 50% negative samples | Logistic Regression with GD |
| Model 4     |  Feature reduced, down sampled data set with 50% negative samples | Logistic Regression with GD |
| Model 5     |  Regularised, feature reduced, down sampled data set with 50% negative samples | Regularised Logistic Regression with GD |

*GD: Gradient Descent*

|              |      Accuracy   | Accuracy (Test) |  Precision   |   Recall     |      F1    |      F1 (Test)      |
| :------------: | :-----------: |  :------------: | :------------: | :------------: | :---: | :-----------------: |
|    Model 1   |   0.912 ± 0.002 |     -      | 0.819 ± 0.150 | 0.003 ± 0.001 | 0.006 ± 0.002 |         -           |
|    Model 2   |   0.912 ± 0.002 |     -      | 0.769 ± 0.060 | 0.007 ± 0.003 | 0.014 ± 0.005 |         -           |
|    Model 3   |   0.750 ± 0.005 |   0.808    | 0.748 ± 0.010 | 0.754 ± 0.020 | 0.751 ± 0.008 |         0.376       |
|    Model 4   |   0.750 ± 0.008 |   0.751    | 0.748 ± 0.012 | 0.755 ± 0.011 | 0.752 ± 0.008 |         0.353       |
|    Model 5   |   0.717 ± 0.014 |   0.838    | 0.801 ± 0.022 | 0.579 ± 0.050 | 0.671 ± 0.026 |         0.387       |

*Note 1: Columns not labelled with (Test) refers to results derived from the validation data set.* </br>
*Note 2: Assuming normality, the values shown above is a 95% confidence interval.*

Model 5 was chosen as the optimal model, with a step-size $\gamma$ of 0.1 and a regularisation term $\lambda$ of 8.25.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

You need to clone this repository.

Create a folder ```datasets``` in the base directory.

In the base directory, upload the csv dataset files (x_train.csv, x_test.csv, y_train.csv). These .csv files are the original datasets provided before the updated datasets were provided.
Then, in the ```datasets``` folder, create a nested folder ```updated_ds``` and upload the updated numpy dataset files (test_ids.npy, train_ids.npy, etc.). These .npy files are the updated datasets provided.

A longer description of the dataset can be found [here](https://www.cdc.gov/brfss/annual_data/annual_2015.html).

Finally, you can run the 'run.py' to get submissions, saved as 'submission.csv'.

### Prerequisites

This project was completed mainly using Python, Numpy and Pandas and a few other common packages in Python.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
