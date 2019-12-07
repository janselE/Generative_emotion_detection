#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import json


# This is an example from https://www.freecodecamp.org/news/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667/

# In[2]:


df_idf = pd.read_csv('amazon/reviews.csv')
df_dataset = pd.read_json('clothing_dataset/renttherunway_final_data.json', lines = True)


# In[3]:


def pre_process(text):
    # to lowercase
    text=text.lower()
    
    # remove tags
    text = re.sub("&lt;/?.*?&gt;", "&lt;&gt; ", text)
    
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    
    return text

def get_stop_words(stop_file_path):
    with open(stop_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


# In[4]:


df_idf['text'] = df_idf['title'] + " " + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x: pre_process(str(x)))

df_dataset['text'] = df_dataset['review_summary'] + " " + df_dataset['review_text']
df_dataset['text'] = df_dataset['text'].apply(lambda x: pre_process(str(x)))

sub_dataset = df_dataset[['text', 'rating']]


# In[5]:


all_data = df_idf['text'].append(sub_dataset['text'])
all_rating = df_idf['rating'].append(sub_dataset['rating'])

print(len(all_data))
print(len(all_rating))


# In[6]:


stopwords = get_stop_words('stopwords.txt')
docs = all_data.tolist()


# In[7]:


cv = CountVectorizer(max_df = .85, stop_words=stopwords)
wordCountVec = cv.fit_transform(docs)


# In[8]:


list(cv.vocabulary_.keys())[:10]


# In[9]:


y = all_rating
# fixing the labels, if > 3.5 is going to be 1 which is positive, else 0
y[:len(df_idf)] = y[:len(df_idf)].apply(lambda x: 1 if x > 3.5 else 0)#y.apply(lambda x: 1 if x > 3.5 else 0) 
y[len(df_idf):] = y[len(df_idf):].apply(lambda x: 1 if x > 5 else 0)#y.apply(lambda x: 1 if x > 3.5 else 0) 
y = y.to_numpy()
x = wordCountVec.toarray()

X_train = x[len(df_idf):]
y_train = y[len(df_idf):]

X_test = x[:len(df_idf)]
y_test = y[:len(df_idf)]
# X_train, _test, y_train, y_test = train_test_split(x, y, test_size=0.5)


# In[1]:


# print(x.shape, y.shape)
# print(y[:len(df_idf)])
# rat = all_rating.to_numpy()
# for i in range(0, len(df_idf)):
#     if y[i] == 1:
#         print(df_idf['text'][i], y_test[i], y[i])


# In[ ]:


from sklearn.neural_network import MLPClassifier
print("training with scikit")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)


# In[ ]:


# Save the model in a binary file
import pickle
filename = 'model2.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[ ]:


# Loads the model from the binary file
import pickle
filename = 'model.sav'
clf = pickle.load(open(filename, 'rb'))


# In[ ]:


test = cv.transform(["Hate", "Good", "Awful", "Best"]).toarray()
clf.predict(test)


# # Adversarial Neural Network

# In this section we start building the GANs, this model takes the word embedding and generate new embeddings that are similar to the given ones. 

# In[ ]:


def build_generator(img_shape):

    noise_shape = (100,)

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)

def build_discriminator(shape):

    img_shape = shape

    model = Sequential()

#     model.add(Flatten(input_shape=img_shape)) # is one dimension
    model.add(Dense(512, input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


# In[ ]:


from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D

img_rows = 1
img_cols = X_train[0].shape
img_shape = (img_cols)

optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

# Build and compile the generator
generator = build_generator(img_shape)
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# The generator takes noise as input and generated imgs
z = Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


# In[ ]:


def results(self, pred, actual):
    results = confusion_matrix(actual, pred)
    print('Confusion Matrix :')
    print(results)
    print ('Accuracy Score :',accuracy_score(actual, pred))
    print ('Report : ')
    print(classification_report(actual, pred))
    print()


# In[ ]:


def train(epochs, data, batch_size=128):


        # Rescale -1 to 1
        X_train = data #(X_train.astype(np.float32) - 127.5) / 127.5
#         X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[1], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = generator.predict(noise)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


# In[ ]:


train(epochs=100, data=X_train)


# In[ ]:


print(X_train.shape)


# In[ ]:


gen = 1
noise = np.random.normal(0, 1, (gen, 100))
new_mails = np.absolute(np.round(generator.predict(noise)))

idx = np.random.randint(0, X_train.shape[1], gen)
imgs = X_train[idx]

prediction = clf.predict(new_mails)

print("Predicted \t{}".format(new_mails))
print("Real \t\t{}".format(imgs))
print("prediction \t{}".format(prediction))


# In[ ]:


test = cv.transform(["Hate", "Good", "Awful", "Best"]).toarray()
cv.inverse_transform(test) # See the generated words


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




