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
The Collaborative Filtering (CF) model uses longitudinal cognitive test scores to identify similarities between patients and predict missing cognitive variables. The model is inspired by a typical user-user CF model that predicts missing points for a given user based on `k` similar users. 

The UDS contains multiple visits from the same patient with each visit contained in a row. These observations contained over 1,900 different clinical features, so we filtered the number of predictor variables in order to avoid sparsity issues in the data set. Since these variables vary in their value ranges (ex. 0-3 for speech and 0-30 for MMSE), values were first standardized by the feature deviation before traditional user-based standardization was applied. This standardization was then taken into account when calculating predictions so that predicted scores would match the original data ranges.

The goal of the model is to predict missing future (and past) values for a patient by identifying other patients with similar historical score trajectories. Since the original dataset contains ~55,000 users, data preprocessing and filtering techniques are used to shrink the pool size. Leave-one-out cross validation was then utilized to determine the optimal set of hyperparameters for predicting missing features.

### Model 2: Multi-Layer Perceptron
The Multi-Layer Perceptron (MLP) model uses structural data derived from MRI/PET image scans. The data captured represents physical measurements of the brain which have variables that are correlated. Principal Component Analysis (PCA) is used to reduce the dimensionality of the data by determining the latent factors and transforming the data into uncorrelated variables. The number of principal components is specified as `k` and the results from PCA are passed into the MLP. The goal of the MLP is to learn nonlinear relationships between structural brain features and cognitive outcomes to predict either Alzheimer's (classification) or a cognitive score such as MMSE.

MRI derived data --> PCA --> Results as Input to MLP --> Train MLP --> Predict score

The full pipleine can be run as a single command with `python runner.py` specifing a number of command line arguments. Alternatively, some commands have been compiled into `run_models.sh` and the shell script can be executed from the command line. Note: Hws for DS4440 were used as a code skeleton inspiration.

### Model 3: Bayesian Model
The Bayesian approach to modeling cognitive decline represents predictions as probability distributions and explicitly incorporates uncertainty. Using the longitudinal clinical visit data, the bayesian model attempts to estimate the probability of cognitive decline given prior observations and patient characteristics. The model then updates beliefs once new patient data is observed. 

Two Bayesian models were implemented using longitudinal MMSE scores from the NACC clinical visit data.
1. Bayesian Linear Regression: Models each patient's MMSE as a linear function of age using a Normal-Inverse-Gamma conjugate prior and a custom Gibbs sampler (2,000 iterations per patient). The prior mean is set via empirical Bayes across all training patients. Evaluated using RMSE and 89% credible interval coverage on held-out visits.
2. Bayesian Logistic Regression: Predicts whether a patient's MMSE will eventually fall below 24 (the clinical impairment cutoff) using only their baseline age and MMSE score. Metropolis-Hastings sampling is used via MCMCpack (10,000 iterations, 2,000 burn-in).

Both models are implemented in R — knit the `.Rmd` files in the `bayesian/` directory to run them. Pre-computed sampling files are already saved in that directory, so the MCMC steps will be skipped automatically if you don't want to re-run them.
