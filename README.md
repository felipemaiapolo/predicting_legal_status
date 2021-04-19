# Repository containing the code used in the paper "Predicting Legal Proceedings Status: Approaches Based on Sequential Text Data" published in ICAIL 2021.

### *Felipe Maia Polo (felipemaiapolo)*

In case you have any question, please get in touch sending us an e-mail in *felipemaiapolo@gmail.com*.

A brief description of the files and folders. Some folders got their content deleted due to their sizes. Please check https://bit.ly/3cAl7pD for the full content.

## ".ipynb" files in the Main Directory

An important note is that all notebooks have a priority indicator number in their name, e.g., "(2)". The numbers indicate in which order the notebooks should be run. For example, a notebook "(2)" must be run before a notebook "(3)". Notebooks with the same numbering have no priority order. Let us briefly describe the functionality of notebooks according to their numbering:

0. In this notebook, we train an unsupervised model to find out which sets of 2, 3, and 4 words should be considered unique tokens;
1. In these notebooks, we train four representations for texts: word2vec, doc2vec, BERT, and TFIDF;
2. In these notebooks, we use the representations already learned to extract the features from the labeled data set and already prepare it for use;
3. In this notebook, we use grid search to choose the best values ​​for the classifier hyperparameters;
4. In this notebook, we train and evaluate the final and optimized models. Also, we generate graphs and tables for the article;
5. In this notebook, we obtain interpretability results;
 
--------------
## ".py" files in the main directory

These files were used as auxiliaries while running the notebooks to make everything cleaner and more organized.

0. packages.py: a file that opens all packages and functions used in the article;
1. tokenizer.py: a file containing the tokenizer used in conjunction with word2vec, doc2vec, and tfidf;
2. clean_functions.py: a file containing the functions used to clean the texts;
3. random_state.py: a file that we run in order to guarantee a fixed seed and reproducible results;
4. fit_models.py: file for training models;

--------------
## Folders in the main directory

0. date: we contain the databases used in the paper;
1. models: contains all models trained in making the paper;
2. plots: contains the graphics generated in the making of the paper;
3. hyper: contains tables with best values for hyperparameters;
