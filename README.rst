HMLI: Heuristic Machine-learning Imputation
==============================================

`HMIL`_is univeriate time series imputation method based on Heuristic search and machine learning.

.. image:: https://github.com/byeungchun/HeuristicImputation/blob/master/doc/images/hmli_flowchart.jpg
   :width: 70%
   :scale: 70%
   :alt: HMLI flow chart


Features
------------

* Genetic algorithm to find optimum dependent variables
* Support Vector Machine for regression and prediction

Using HMLI
------------

Edit data file location and evolutationary parameter in hmli.ini:

Default parameter values:

* number of prediction observations: 12
* number of genes per chromosome: 6
* number of chromosomes per population: 10
* number of populations: 100
* cutoff rate of bad chromosome: 0.5
* cutoff correlation coefficient: 0.97
* random seed list: [1, 2, 3]
* missing rate list: [0.1, 0.4, 0.7]
* hdf5 data file: data/mei.h5
* result file: output/finalRes1.pickle

Execute the main file:

..code-block:: bash

   $python main.py
   
It will execute parallel process and consume CPU resource a lot. 

Sample data is OECD MEI monthly series. The total number series in the file is 5,411.
