#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hakanskn/Blindness-Detection/blob/main/Blindness_Detection_by_Feature_Extraction_from_CNN_%2B_SVM_Classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # KAGGLE DATA

# In[ ]:


get_ipython().system('pip install -q kaggle')


# In[ ]:


# from google.colab import files
# files.upload()


# In[ ]:


get_ipython().system('mkdir ~/.kaggle')


# In[ ]:


#!cp kaggle.json ~/.kaggle/

get_ipython().system('cp /content/drive/MyDrive/COMMON/APTOS_2019/kaggle.json ~/.kaggle/')


# In[ ]:


get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle datasets list')


# In[ ]:


get_ipython().system('kaggle competitions download -c aptos2019-blindness-detection')


# In[ ]:


get_ipython().system('mkdir aptos_2019')


# In[ ]:


get_ipython().system('unzip aptos2019-blindness-detection.zip -d aptos_2019')


# # LIBRARIES & CONFIGS

# ## GPU Info

# In[ ]:


gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


# ## Drive Mount Code

# In[ ]:


# !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
# !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
# !apt-get update -qq 2>&1 > /dev/null
# !apt-get -y install -qq google-drive-ocamlfuse fuse
# from google.colab import auth
# auth.authenticate_user()
# from oauth2client.client import GoogleCredentials
# creds = GoogleCredentials.get_application_default()
# import getpass
# !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
# vcode = getpass.getpass()
# !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
# %cd /content
# !mkdir drive
# %cd drive
# !mkdir MyDrive
# %cd ..
# %cd ..
# !google-drive-ocamlfuse /content/drive/MyDrive


# ## Libraries

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import pandas as pd\nimport numpy as np\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nfrom warnings import filterwarnings\nfrom sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve\nfrom tensorflow.keras.models import Sequential\nfrom keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D, Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM\nfrom keras import models\nimport tensorflow as tf\nimport os\nimport os.path\nfrom pathlib import Path\nimport cv2\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\nfrom keras.utils.np_utils import to_categorical\nfrom sklearn.model_selection import train_test_split\nfrom keras import regularizers\nfrom tensorflow.keras.optimizers import *\nimport glob\nfrom pathlib import Path\nfrom PIL import Image\nimport PIL\nimport copy\nfrom tensorflow import keras\nfrom keras.utils.layer_utils import count_params\n\nfrom tensorflow.python.keras.utils.vis_utils import model_to_dot\nfrom keras.utils.vis_utils import plot_model\nfrom IPython.display import SVG\nimport pydot\nimport graphviz\n\nfrom keras.models import model_from_json\n!pip install ipython-autotime\n%load_ext autotime\n\n\n!pip install tensorflow_addons\nimport tensorflow_addons as tfa\n\nfrom tensorflow.keras.callbacks import *\nfrom tensorflow.keras.layers import *\n\n%matplotlib inline\nimport datetime\nimport time\nimport gc\nimport shutil\n\nfrom sklearn.model_selection import KFold, StratifiedKFold\n\n!pip install -U tensorboard-plugin-profile\n\n# IGNORING UNNECESSARRY WARNINGS\n\nfilterwarnings("ignore",category=DeprecationWarning)\nfilterwarnings("ignore", category=FutureWarning)\nfilterwarnings("ignore", category=UserWarning)\n')


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

print('TF Version:', tf.__version__)
print('TFA Version:', tfa.__version__)


# # VARIABLES & HYPERPARAMETERS

# In[ ]:


BATCH_SIZE = 64
EPOCHS = 20
LR = 0.003
IMG_WIDTH=128
IMG_HEIGHT=128
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
NUM_CLASSES = 5
_SPLIT_RATE = 0.9

_AUGMENTED = "TRUE"
AUGMENT_RATE_PER_CLASS = 0.1

data_type = ''
model_name = 'ResNet50_tuned'


# DeepCovX_NM3
# ResNet50_random
# ResNet50_tuned
# Vgg19_random
# Vgg19_tuned
# DenseNet121_random
# DenseNet121_tuned
# InceptionV3_random
# InceptionV3_tuned
# Ctroke
# BarisStroke
# AlexNet

# original_data
# median_data
# nlm_data
# tv_data
# wt_data

_DESC = 'setup_test'


CV_PARTS = 5

RANDOM_STATE = 42

#/content/drive/MyDrive/HAKANSKN - DR/Dersler/Tıpta Bilişim/Proje/Prostat
# /content/aptos_2019

root_path = '/content/aptos_2019/'#+ data_type + '/'
comparison_save_path = '/content/drive/MyDrive/COMMON/APTOS_2019/Results/' + model_name + '/' + model_name + '.xlsx'
root_folder_to_save = '/content/drive/MyDrive/COMMON/APTOS_2019/Results/' #root_path + 'Results/'
#BENIGN_DIR = root_path + 'BENIGN'
#MALIGN_DIR = root_path + 'MALIGN'

model_name_to_save = '_ep_' + str(EPOCHS) + '_bs_' + str(BATCH_SIZE)
datetime_suffix = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
print(datetime_suffix)


# # DATA SETTINGS

# In[ ]:


df = pd.read_csv("/content/aptos_2019/train.csv")
df['id_code'] = root_path + "train_images/" + df['id_code'].astype(str) + ".png"
df.rename(columns = {'id_code':'FILE', 'diagnosis':'CATEGORY'}, inplace = True)
df = df.astype(str)
df.head()


# In[ ]:


from google.colab.patches import cv2_imshow
image = cv2.imread(df.iloc[1,0], cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = np.array(image)
plt.imshow(pixels)
plt.show()


# In[ ]:


Main_Train_Data = df.sample(frac=1).reset_index(drop=True)
Main_Train_Data.head()


# In[ ]:


Main_Train_Data["CATEGORY"].value_counts()


# In[ ]:


# train_data,test_data = train_test_split(Main_Train_Data,train_size=_SPLIT_RATE, random_state = RANDOM_STATE, stratify = JPG_Category_Series)
train_data,test_data = train_test_split(Main_Train_Data,train_size=_SPLIT_RATE, random_state = RANDOM_STATE, stratify = Main_Train_Data.CATEGORY)

print("train_data.shape: ", train_data.shape)
print("-----"*20)

print("test_data.shape: ", test_data.shape)
print("-----"*20)

# print("train_data first 3 rows:")
# print(train_data.iloc[0:3, 0].values)
# print("-----"*20)

# print("test_data first 3 rows:")
# print(test_data.iloc[0:3, 0].values)
# print("-----"*20)


# In[ ]:


test_data.to_csv("test_data.csv", index=False)


# # FEATURE EXTRACTION

# In[ ]:


train_data.columns


# In[ ]:


densenet_model = keras.applications.densenet
conv_model = densenet_model.DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH,IMG_HEIGHT,3))
#conv_model.trainable = True

for layers in conv_model.layers:
    layers.trainable=False

print(conv_model.output)


# In[ ]:


conv_model.summary()


# In[ ]:


from tensorflow.keras.utils import array_to_img, img_to_array, load_img


# In[ ]:


feature_list = []
for path in train_data['FILE'].to_numpy():
    x = load_img(path,target_size=(IMG_WIDTH,IMG_WIDTH))
    img_array = img_to_array(x)
    img_array = np.expand_dims(img_array, axis=0)
    features = conv_model.predict(img_array)
    feature_list.append(features)

feat_lst = np.reshape(feature_list,(-1,4*4*1024))


# In[ ]:


print(feat_lst.shape)


# In[ ]:


gender = {'0': 0,'1': 1,'2': 2,'3': 3,'4': 4}
y = [gender[item] for item in train_data.CATEGORY]
len(y)


# In[ ]:


from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(feat_lst, y, test_size=0.2, random_state=RANDOM_STATE)

glm = LogisticRegression(C=0.1)
glm.fit(X_train,y_train)


# In[ ]:


print("Accuracy on validation set using Logistic Regression: ",glm.score(X_test,y_test))


# ## Default SVM Classifier

# In[ ]:


# import SVC classifier
from sklearn.svm import SVC

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score

# instantiate classifier with default hyperparameters
svc=SVC()

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0)

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0)

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# make predictions on train set
y_pred_train = svc.predict(X_train)

# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[ ]:


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(svc)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


# ## Linear Kernel SVM Classifier

# In[ ]:


# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0)


# fit classifier to training set
linear_svc.fit(X_train,y_train)


# make predictions on test set
y_pred_test=linear_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# In[ ]:


# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0)


# fit classifier to training set
linear_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0)


# fit classifier to training set
linear_svc1000.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc1000.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ## Ploynomial Kernel SVM Classifier

# In[ ]:


# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0)


# fit classifier to training set
poly_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=poly_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0)


# fit classifier to training set
poly_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# instantiate classifier with polynomial kernel and C=100.0
poly_svc1000=SVC(kernel='poly', C=1000.0)


# fit classifier to training set
poly_svc1000.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc1000.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ## Sigmoid Kernel SVM Classifier

# In[ ]:


# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0)


# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0)


# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ## RF

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# instantiate classifier
rf = RandomForestClassifier()

# fit classifier to training set
rf.fit(X_train,y_train)


# make predictions on test set
y_pred=rf.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with Random Forest : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

