import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import roc_curve,auc
#Add stuff from cgan model
from CGAN_model import generator_network_w_label, discriminator_network, define_models_CGAN, training_steps_GAN
from tensorflow.keras.optimizers.legacy import Adam

#%% Data preprocessing
raw_data = pd.read_csv('C:/Users/Raed Karkoub/Desktop/DS340/Data/creditcard_p3.csv',sep=',')
# Count number of fraud cases before any splitting
print("Number of fraud cases in raw data:", raw_data[raw_data['Class'] == 1].shape[0])
plt.figure(figsize=(6,4),dpi=150)

pd.value_counts(raw_data['Class']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
plt.close()



X_train, test = train_test_split(raw_data, train_size=0.8, random_state=0, stratify=raw_data['Class'])
X_train, X_test = train_test_split(X_train, train_size=0.8, random_state=0, stratify=X_train['Class'])
# Count number of fraud cases after splitting
print("Number of fraud cases in training set:", X_train[X_train['Class'] == 1].shape[0])
print("Number of fraud cases in test set:", test[test['Class'] == 1].shape[0])
print("Number of fraud cases in validation set:", X_test[X_test['Class'] == 1].shape[0])
X_train.loc[:,"Time"] = X_train["Time"].astype(np.float64).apply(lambda x : x / 3600 % 24)
X_train.loc[:,'Amount'] = np.log(X_train['Amount']+1)
X_test.loc[:,"Time"] = X_test["Time"].astype(np.float64).apply(lambda x : x / 3600 % 24)
X_test.loc[:,'Amount'] = np.log(X_test['Amount']+1)

test.loc[:,"Time"] = test["Time"].apply(lambda x : x / 3600 % 24).astype(np.float64) 
test.loc[:,'Amount'] = np.log(test['Amount']+1)

y_train = X_train['Class'].values
X_train = X_train.drop(['Class'], axis=1).values
test_y_train = test['Class'].values
test_x_train = test.drop(['Class'], axis=1).values

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2, k_neighbors=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
# Count number of fraud cases after applying SMOTE
print("Number of fraud cases in training set after SMOTE:", sum(y_train_res))

# \R save the resampled dataset for CGAN
data_cgan=pd.DataFrame(X_train_res)
data_cgan["Label"]= y_train_res


plt.figure(figsize=(6,6),dpi=150)

pd.value_counts(y_train_res).plot.bar()
plt.title('SMOTE Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
plt.close()
print("Label shape:", y_train_res.shape)
print("Label sample:", y_train_res[:5])  # Display first 5 labels
print("Label type:", type(y_train_res))
print("Unique labels:", np.unique(y_train_res))  # This will help show how many classes you have and how they are structured


#Use CGAN 
# Define CGAN parameters
rand_dim = 100  # Dimension of random noise for generator input
data_dim = X_train_res.shape[1]  # Number of features in the dataset
label_dim = 1  # Assuming binary labels (fraud vs non-fraud)
base_n_count = 128

# Create CGAN models using the imported define_models_CGAN function
generator_model, discriminator_model, combined_model = define_models_CGAN(
    rand_dim, data_dim, label_dim, base_n_count
)
print(generator_model.input)

# Compile the models as needed
adam = Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
combined_model.compile(optimizer=adam, loss='binary_crossentropy')

# Training CGAN models
nb_steps = 100  # Number of training steps
batch_size = 128  # Size of the training batch
k_d = 1  # Number of times to train discriminator per generator step
k_g = 1  # Number of times to train generator per step

# Use the imported training function to train the CGAN models
training_steps_GAN([
    'CGAN',
    False,  # Starting step index
    0,  # Initial step
    data_cgan,  # Training data
    [f'Feature_{i}' for i in range(data_dim)],  # Column names for data features
    data_dim,
    ['Label'],  # Label column
    label_dim,
    generator_model,
    discriminator_model,
    combined_model,
    rand_dim,
    nb_steps,
    batch_size,
    k_d,
    k_g,
    0,  # Pre-train discriminator steps
    100,  # Logging interval
    0.0002,  # Learning rate
    base_n_count,
    '',  # Data directory
    None,  # Generator model path (for loading)
    None,  # Discriminator model path (for loading)
    False,  # Show training visuals (disable for speed)
    [],  # Combined loss list
    [],  # Discriminator generated loss list
    [],  # Discriminator real loss list
    []  # XGBoost loss list (not used here)
])

# Generate synthetic data using the CGAN generator
random_noise_input = np.random.normal(0, 1, (10000, rand_dim))  # Generate random noise
synthetic_labels = np.random.randint(0, 2, (10000, 1))  # Generate random labels
synthetic_data = generator_model.predict([random_noise_input, synthetic_labels])  # Generate synthetic samples

synthetic_data=synthetic_data[:,:-1]
# Combine synthetic data with the SMOTE-resampled data
X_combined = np.vstack([X_train_res, synthetic_data])  # Combine real and synthetic features
y_combined = np.hstack([y_train_res, synthetic_labels.flatten()])  # Combine real and synthetic labels

# Check the new dataset distribution
print("Combined dataset shape:", X_combined.shape)
print("Class distribution after combining with CGAN outputs:")
print(pd.value_counts(y_combined))

# Update the training dataset for the autoencoder
X_autoencoder_train = X_combined



import tensorflow_addons as tfa
#%% Autoencoder
activation = 'relu'
encoding_dim = 256
nb_epoch = 100
batch_size = 256

input_dim=30
input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim,kernel_regularizer='l2')(input_layer)
encoder=tfa.activations.mish(encoder)
encoder = Dense(int(encoding_dim / 2),kernel_regularizer='l2')(encoder)
encoder=tfa.activations.mish(encoder)
encoder = Dense(int(encoding_dim / 4),kernel_regularizer='l2')(encoder)



decoder=tfa.activations.mish(encoder)


decoder = Dense(encoding_dim / 4,kernel_regularizer='l2')(decoder)
decoder=tfa.activations.mish(decoder)
decoder = Dense(encoding_dim / 2,kernel_regularizer='l2')(decoder)
decoder=tfa.activations.mish(decoder)
decoder = Dense(encoding_dim ,kernel_regularizer='l2')(decoder)
decoder=tfa.activations.mish(decoder)
decoder = Dense(input_dim,kernel_regularizer='l2')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)




print(autoencoder.summary())


autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="./best_model.h5",
                              verbose=0,
                              save_best_only=True)
rp=tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    mode="min",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.001,
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

history = autoencoder.fit(X_autoencoder_train, X_autoencoder_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    verbose=1,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[rp,early_stopping]).history
                    # 调节图像大小,清晰度
plt.figure(figsize=(6,6),dpi=150)

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
    #plt.ylim(ymin=0.70,ymax=1)
plt.show()

encoder_all = Model(input_layer,encoder)
#encoder_all.save("./encoder.h5")
#encoder_all = tf.keras.models.load_model('encoder.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
enc_all = encoder_all.predict(X_train)


y_test = X_test['Class'].values
X_test = X_test.drop(['Class'], axis=1).values


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
#%% Lgbt

lgb_model = lgb.LGBMClassifier(num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                              
                              
# Apply the encoder to the SMOTE resampled training set instead of the original training set
enc_all_res = encoder_all.predict(X_train_res)

# Now use the resampled encoded data to train the LightGBM model
lgb_model.fit(enc_all_res, y_train_res)

# Apply the encoder to the test set for prediction
test_x = encoder_all.predict(X_test)

# Predict probabilities for the test set
predicted_proba = lgb_model.predict_proba(test_x)
train_predictions = (predicted_proba[:, 1] >= 0.25).astype('int')
# Metrics calculation
precision = precision_score(y_test, train_predictions)
recall = recall_score(y_test, train_predictions)
f1 = f1_score(y_test, train_predictions)
accuracy = accuracy_score(y_test, train_predictions)
mcc = matthews_corrcoef(y_test, train_predictions)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, train_predictions)

# Display metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("MCC:", mcc)

# Plot confusion matrix
plt.figure(figsize=(6, 6), dpi=150)
sns.heatmap(conf_matrix, xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"], annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, predicted_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6), dpi=150)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
