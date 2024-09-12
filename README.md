# Sentiment-Analysis

## Objective

In this project, I aim to use GPU accelerated Classification packages by rapids.ai in order to perform Sentiment Analysis on a Financial News dataset (https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news).

## Methodology

The dataset has two columns, one for the news and the other with the sentiment of the news. In order to classify the news, I have generated text-embeddings using the 'BERT-Base-Cased' model. The reason for selecting this model was to avoid cases of confusion regarding the context of company names like Apple, which can be easily confused with the fruit by the uncased model.

In order to perform the train-test split, I had to use the pandas DataFrame as train_test_split from cuML did not support splitting columns with string values (I had not label encoded the target column (sentiment) yet).

I performed Principal Component Analysis on the text embeddings while maintaining over 95% variance. Post this, I used four models, LightGBM, XGBoost, Random Forest Classifier and SVC and performed a trial using their default parameters. Since XGBoost and LightGBM do not support cuML DataFrames, I had to convert them to NumPy arrays. SVC turned out to be the model with the best test accuracy.

I used Optuna for hyperparameter optimization. This uses a method called the tree parzen estimator. Optuna keeps track of historical trials in order to guess parameters within the given ranges and tries to maximize the ratio of probability of getting a good result to the probability of getting a bad result with a combination of hyperparameters. I optimized the F1 Score, which is given by the formula: 2/((Precision)^-1 + (Recall)^-1). So, the higher the precision and recall, the higher the f1 score. In addition, the average of the f1 score was set to 'macro', which means that each f1 score of each class is weighed equally irrespective of the distribution of data, leading to a fair evaluation of the model on imbalanced data. I also penalized overfitting greater than 5% by returning a negative value as the f1 score.

I found that the data was imbalanced in favour of the neutral sentiment, so I used StratifiedKFold in order to maintain the weight of the classes rather than splitting the training and validation set randomly. This also required me to use a NumPy array, and then convert everything back to a cuML DataFrame for each fold. 

## Final result

I observed a training accuracy of 82.8% and a test accuracy of 78.4%, which is lacklustre to say the least. On talking to a superbisor at work, I found out that they have also been getting bad results on GPU accelerated classification models such as these, the reason, I'll have to explore!
