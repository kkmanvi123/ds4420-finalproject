# Predicting Alzheimer's Related Cognitive Decline
Annika Salpukas, Manvi Kottakota, Ruby Mason

## Overview
Alzheimer’s disease (AD) is a form of dementia that progresses slowly, affecting memory, thinking, and behavior. Over time, the disease damages brain cells, leading to worsening cognitive decline. Early stages of AD often include short-term memory loss, impaired thinking, and difficulty with problem solving, while later stages involve severe cognitive impairment, physical decline, and the need for full-time care. Although AD currently has no cure, the ability to detect early cognitive decline can help doctors monitor disease progression, and family members have a better treatment plan for the patient.

This project explores predicting cognitive decline metrics using machine learning approches on clinical data. The three different models are as follows:
1. **Collaborative Filtering (CF)** using clinical visit functional test scores to model cognitive decline based on similar patients.
2. **Multi-Layer Perceptron (MLP)** from MRI and PET scan derived metadata to predict the presence of AD or a cognitive score.
3. **Bayesian Modeling** using clinical visit functional test scores to estimate probabalistic disease progression.

## Datasets
The National Alzheimer’s Coordinating Center (NACC) aggregates data from more than 40 Alzheimer's Disease Research Centers (ADRCs) across the country. The NACC houses a number of datasets relating to AD including: longitudinal patient phenotype and cognitive data, fluid biomarker data, imaging data, and more. 
Two subsets of the NACC database are used in this project.

### Uniform Data Set
The Uniform Data Set (UDS) contains patient data collected from participants over multiple visits. Some variables in the dataset that are relevant predicting cognitive decline include:
* Demographic information (including age)
* Clinical diagnosis
* Cognitive assessment scores (Mini-Mental State Examination (MMSE), Clinical Dementia Rating (CDR), etc.)

This dataset is used for the models that rely on patient cognitive trajectories over time, including Collaborative Filtering and Bayesian Modeling, to model future or unknown predicitons.

### Standardized Centralized Alzheimer's & Related Dementias Neuroimaging
The Standardized Centralized Alzheimer's & Related Dementias Neuroimaging (SCAN) dataset consists of MRI/PET derived metadata rather than the raw images. These features describe the measurements of the brain that are gathered from the images. Some variables of the dataset that are relevant to predicting cognitive decline include:
* Brain region volumes (hippocampus, temporal lobes, and parietal lobes)
* White matter volume

This dataset is used for the Multi-Layered Perceptron to predict cognitive decline metrics.

## Methods
This project evaluates three different machine learning approaches for predicting cognitive decline using clinical and neuroimaging-derived data. Each model uses a different representation of patient information to predict overall cognitive decline. The methods are designed to capture patterns between patients and make an accurrate prediction for a given patient. 

### Model 1: Collaborative Filtering Model
The Collaborative Filtering (CF) model uses longitudinal cognitive test scores to identify similarities between patients and predict future cognitive outcomes. The model is inspired by a typical CF model that predicts missing points for a given user based on `k` similar users. 

The UDS contains multiple visits from the same patient with each visit contained in a row. The "age" variable allows for the data to be transformed into a matrix with the users as rows and the columns as ages (or bins). Each cell contains a cognitive score such as the Mini-Mental State Examination (MMSE) (ranked up to 30 with lower scores indicating more severe decline). An example of the structure is shown below.

| Patient | Age 60 | Age 62 | Age 64 | Age 66 |
| ------- | ------ | ------ | ------ | ------ |
| U1      | 29     | 28     | 27     | 25     |
| U2      | 30     | ?      | 29     | ?      |
| U3      | 28     | 26     | 24     | 22     |

The goal of the model is to predict missing future (and past) values for a patient by identifying other patients with similar historical score trajectories. Since the original dataset contains ~55,000 users, data preprocessing and filtering techniques are used to shrink the pool size. Additionally, a Simon Funk based approach can be implemented to learn the latent factors and avoid creating the entire user x user matrix.

### Model 2: Multi-Layer Perceptron
The Multi-Layer Perceptron (MLP) model uses structural data derived from MRI/PET image scans. The data captured represents physical measurements of the brain which have variables that are correlated. Principal Component Analysis (PCA) is used to reduce the dimensionality of the data by determining the latent factors and transforming the data into uncorrelated variables. The number of principal components is specified as `k` and the results from PCA are passed into the MLP. The goal of the MLP is to learn nonlinear relationships between structural brain features and cognitive outcomes to predict either Alzheimer's (binary) or a cognitive score such as MMSE.

MRI derived data --> PCA --> Results as Input to MLP --> Train MLP --> Predict score

The full pipleine can be run as a single command with `python runner.py` specifing a number of command line arguments. Alternatively, some commands have been compiled into `run_models.sh` and the shell script can be executed from the command line. 

### Model 3: Bayesian Model
The Bayesian approach to modeling cognitive decline represents predictions as probability distributions and explicitly incorporates uncertainty. Using the longitudinal clinical visit data, the bayesian model attempts to estimate the probability of cognitive decline given prior observations and patient characteristics. The model then updates beliefs once new patient data is observed. 

Exact modeling strategy TBD as Bayesian Modeling unit is taught!  



