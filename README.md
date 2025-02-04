# Toolkit for DS application
Self-written toolkit for data cleaning, visualization, preprocessing, EDA, &amp; ML application


## Table of content

**data.py**

* Basic info for data (`class Data.info()`): includes dtype, missingness, unique values
* Simple preprocessing (`Data.clean()`): includes clean header, string processing, & standardize NAs

### Future work

**eda.py**

Class Data:

* Diagnosis: identify primary key, visualize NA patterns, statistics summary, suggesstion for dtypes
* transform dtypes in easy way (follow diagnosis | customized)
* integrate LLM/ML for diagnosis & more applications
* save: metadata (dtypes, summary) & actual data
* turn this into python pkg
Class Vis:
* visualization
* wrapper for pgywalker

Class Preprocess:
* need to separate preprocess for all data, train data, test data

Class Model:
* wrapper for lazymodels