# Fake-News-Detection

### Project Description
Social media has provided an excellent interactive technology platform that allows the creation, sharing, and exchange of interests, ideas, or information via virtual networks very quickly. A new platform enables endless opportunities for marketing to reach new or existing customers. However, it has also opened the devil’s door for falsified information which has no proven source of information, facts, or quotes. It is really hard to detect whether the given news or information is correct or not. Here, as a part of this project, we need to detect the authenticity of given news using DL.              
The dataset contains around 7k fake news, including a title, body, and label (FAKE or REAL). The task is to train the data to predict if the given news is fake or real.

### Project Outcomes
•	Pre-process the data  
•	Evaluate the various algorithms which can affect best outcome  
•	Train a model to predict the likelihood of REAL news

### Prerequisites & Version
•	Sklearn -- 0.24.2  
•	TensorFlow -- 2.0.0  
•	Keras -- 2.3.1  
•	NLTK -- 3.6.5  
•	Gensim -- 3.8.3  
•	H5py -- 2.10.0  

### Data Clean/Text Pre-processing:
  1. Load dataset from csv file and delete the missing data points

  2. Remove url in the text

  3. Remove newline signals from text

  4. Remove numbers and punctuations from text

  5. Convert all letters into lowercase

  6. Tokenization text

  7. Remove stopwords from text

  8. Normalization: including stemming and Lemmatization

  9. Remove words whose length are equal or less than 2

### Choice of approach to implement
•	ML algorithm - Naive Bayes, K-Nearest Neighbours, Support Vector Machine, Logistic Regression, Decision Tree  
•	DL algorithm - MLP, CNN, LSTM

### Project Structure
<p align='left'>
  <img src='imgs/Proposed Approach Outline.png.gif' width='400'/>
</p>
