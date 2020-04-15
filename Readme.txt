--------------------------------------------------------------------------
Implementation of Artificial Neural Networks(ANNs) & K Nearest Neighbors
--------------------------------------------------------------------------

In this project, I have implemented Artificial Neural Networks(ANNs) & K Nearest Neighbors on two data sets. My first data set is 'SGEMM GPU Kernel Performance Prediction' and my second data set is 'Rain in Australia'.

-----------------
Dataset Source:
-----------------

We have used the SGEMM GPU kernel performance Data Set available for download at -

https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance 

Second Data set can be obtained from Kaggle, link to the dataset is given below –

https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

---------------
Prerequisites: 
---------------

Below Packages are prerequisites to run Artificial Neural Networks(ANNs) & K Nearest Neighbors -

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing, model_selection, metrics
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn import preprocessing  
from sklearn.tree import export_graphviz
from sklearn import datasets, utils, tree
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from IPython import display
from graphviz import Source
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization
from keras import optimizers
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

