#!/usr/bin/env python
# coding: utf-8

# # COEN 140 Final Project - Music Genre Classifer

# In[241]:


import os
import json
import numpy as np
import scipy
import pandas as pd
import librosa as lb
import warnings

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


# ## Import functions

# In[3]:


def import_features(n=None):
    return pd.read_csv('fma_metadata/features.csv', header=[0,1,2], index_col=0, nrows=n)

def import_tracks(n=None, col=':'):
    return pd.read_csv('fma_metadata/tracks.csv', header=[0,1], index_col=0, nrows=n)

def import_genres(n=None):
    return pd.read_csv('fma_metadata/genres.csv', header=0, index_col=0, nrows=n)


# ## Feature extraction function

# In[3]:


def get_song_features(name):
    # Get the file path to an included audio example
    file = os.path.join(os.getcwd(), "test_songs", name)
    
    # Load into waveform 'y', sampling rate 'sr'
    y, sr = lb.load(file)
    print('> \'{}\' successfully loaded'.format(name))

    ## Extract all features
    df = {}
    
    df["chroma_stft"] = lb.feature.chroma_stft(y=y, sr=sr)
    df["chroma_cqt"] = lb.feature.chroma_cqt(y=y, sr=sr)
    df["chroma_cens"] = lb.feature.chroma_cens(y=y, sr=sr)
    
    df["tonnetz"] = lb.feature.tonnetz(y=y, sr=sr)
    df["mfcc"] = lb.feature.mfcc(y=y, sr=sr)
    
    df["spectral_centroid"] = lb.feature.spectral_centroid(y=y, sr=sr)
    df["spectral_bandwidth"] = lb.feature.spectral_bandwidth(y=y, sr=sr)
    df["spectral_contrast"] = lb.feature.spectral_contrast(y=y, sr=sr)
    df["spectral_rolloff"] = lb.feature.spectral_rolloff(y=y, sr=sr)
    
    df["rmse"] = lb.feature.rms(y=y)
    df["zcr"] = lb.feature.zero_crossing_rate(y=y)
    
    print('> Successfully extracted into dict')
    return df


# In[4]:


## format new song features

# fetch new song features as dict
feat = get_song_features("one_summers_day.mp3")

# fetch empty array with correct format, then append empty row
df = import_features(0).append(pd.Series(dtype=float), ignore_index=True)

# apply stats to new song features
stats = ['kurtosis','max','mean','median','min','skew','std']
funcs = [scipy.stats.kurtosis, np.amax, np.mean, np.median, np.amin, scipy.stats.skew, np.std]

for ft in df.columns.unique(0):
    for st, fn in zip(stats, funcs):
        df.loc[:,(ft,st)] = fn(feat[ft], axis=1)
        
print('> Successfully applied statistics')


# ## Classifier Helper Functions

# In[4]:


def format_test_valid(X, y, drop_unique=False):
    # parse genres into array
    y = y.apply(lambda g: json.loads(g))
    
    # select/remove empty genres
    eg = y.index[y.map(lambda g: len(g)==0)]
    X = X.drop(eg)
    y = y.drop(eg)
    
    # split into train/validate groups
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # remove genres with only 1 entry
    if drop_unique:
        ug = yt.drop_duplicates(False).index
        Xt = Xt.drop(ug)
        yt = yt.drop(ug)
    
    return Xt, Xv, yt, yv

def format_track_data(data, cols=None):
    # select specified columns
    data = data.loc[:,('track',cols)]
    data.columns = cols
    
    # parse genres into array
    data = data.applymap(lambda t: json.loads(t))
    
    return data

def score(a, b):
    assert len(a) == len(b), 'Arrays are not the same size'
    
    c = 0
    for v1, v2 in zip(a,b):
        if isinstance(v2, (int, np.integer)):
            if (v1==v2):
                c=c+1
        else:
            if v1 in v2:
                c=c+1
    return c/len(a)

def print_scores(ts, vs, name='New'):
    print('> {} Scores:'.format(name))
    print('Training score: {:.3f}\nValidation score: {:.3f}\n'.format(ts, vs))


# ## Classifier Implementations
# ### LDA Function

# In[264]:


def m1_LDA(X, y, score_method='default', verbose=False):
    #split data
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)
    yt1 = yt.apply(lambda g: g[0])
    
    # fitting
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xt, yt1)
    
    # predictions
    pt = lda.predict(Xt)
    terr = score(pt,yt)
    
    pv = lda.predict(Xv)
    verr = score(pv,yv)
    
    if verbose:
        print_scores(terr, verr, 'LDA')
    
    return verr


# ### QDA Function

# In[227]:


def m2_QDA(X, y, score_method='default', verbose=False):
    #split data
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)
    yt1 = yt.apply(lambda g: g[0])
    
    # drop unique tracks
    ug = yt1.drop_duplicates(False).index
    Xt = Xt.drop(ug)
    yt = yt.drop(ug)
    yt1 = yt1.drop(ug)
    
    # fitting
    qda = QuadraticDiscriminantAnalysis(tol=10**-10)
    qda.fit(Xt, yt1)
    
    # predictions
    pt = qda.predict(Xt)
    terr = score(pt,yt)
    
    pv = qda.predict(Xv)
    verr = score(pv,yv)
    
    if verbose:
        print_scores(terr, verr, 'QDA')
    
    return verr


# ### KMC Function

# In[198]:


def m3_KMC(X, y, init_method='k-means++', verbose=False):
    #split data
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)
    yt1 = yt.apply(lambda g: g[0])
    
    # kmeans
    centroids = yt1.drop_duplicates()
    if init_method == 'centroids':
        init_method = Xt.loc[centroids.index]
    kmc = KMeans(n_clusters=len(centroids), init=init_method, n_init=10)
    kmc.fit(Xt)
    
    # predictions
    pt = pd.Series(kmc.predict(Xt), index=Xt.index)
    for ci in pt.unique():
        same_gen = pt[pt==ci].index.values
        #print('{} entries in cluster {}'.format(len(same_gen), ci))
        all_gen = np.concatenate(yt[same_gen].values)
        mode_gen, cont_gen = scipy.stats.mode(all_gen)
        pt.loc[same_gen] = mode_gen[0]
    terr = score(pt,yt)
    
    pv = pd.Series(kmc.predict(Xv), index=Xv.index)
    for ci in pv.unique():
        same_gen = pv[pv==ci].index.values
        mode_gen, cont_gen = scipy.stats.mode(np.concatenate(yv[same_gen].values))
        pv.loc[same_gen] = mode_gen[0]
    verr = score(pv,yv)
    
    if verbose:
        print_scores(terr, verr, 'KMeans')
    
    return verr


# ## Testing
# ### Initial Testing

# In[256]:


# Import data
n = None # None = 106,574
X = import_features(n)
y = import_tracks(n)
g = import_genres(n)

# format genres properly
cols = ['genres','genres_all']
y = y.loc[:,('track',cols)]
y.columns = cols
y = y.applymap(lambda t: json.loads(t))

# Remove entries with empty genres
eg = y.index[y['genres'].map(lambda t: len(t)==0)]
X = X.drop(eg)
y = y.drop(eg)

# Add another column holding parent genres
y['top_level'] = y['genres'].apply(lambda t: [g.loc[t[0]]['top_level']])
print('> Data imported successfully')


# In[ ]:


lda = m1_LDA(X, y['genres'], verbose=True)
lda = m1_LDA(X, y['genres_all'], verbose=True)
lda = m1_LDA(X, y['top_level'], verbose=True)


# In[ ]:


warnings.filterwarnings('default')
qda = m2_QDA(X, y['genres'], verbose=True)
qda = m2_QDA(X, y['genres_all'], verbose=True)
qda = m2_QDA(X, y['top_level'], verbose=True)


# In[ ]:


for km in ['k-means++', 'random', 'centroids']:
    print('> Testing method: ', km)
    m3_KMC(X, y['genres'], km, verbose=True)
    m3_KMC(X, y['genres_all'], km, verbose=True)
    m3_KMC(X, y['top_level'], km, verbose=True)


# ### Per Feature/Statistic Testing

# In[ ]:


yc = 'top_level'
cols = ['lda','qda','kmc-km++','kmc-rand','kmc-cent']
warnings.filterwarnings('ignore')

## test each feature
res_feat = pd.DataFrame(columns=cols)
for ft in X.columns.unique(level=0):
    print('> Testing feature: ', ft)
    res_feat.loc[ft, 'lda'] = m1_LDA(X.loc[:,pd.IndexSlice[ft, :, :]], y[yc])
    res_feat.loc[ft, 'qda'] = m2_QDA(X.loc[:,pd.IndexSlice[ft, :, :]], y[yc])
    res_feat.loc[ft, 'kmc-km++'] = m3_KMC(X.loc[:,pd.IndexSlice[ft, :, :]], y[yc], 'k-means++')
    res_feat.loc[ft, 'kmc-rand'] = m3_KMC(X.loc[:,pd.IndexSlice[ft, :, :]], y[yc], 'random')
    res_feat.loc[ft, 'kmc-cent'] = m3_KMC(X.loc[:,pd.IndexSlice[ft, :, :]], y[yc], 'centroids')

## test each statistic
res_stat = pd.DataFrame(columns=cols)
for st in X.columns.unique(level=1):
    print('> Testing statistic: ', st)
    res_stat.loc[st, 'lda'] = m1_LDA(X.loc[:,pd.IndexSlice[:, st, :]], y[yc])
    res_stat.loc[st, 'qda'] = m2_QDA(X.loc[:,pd.IndexSlice[:, st, :]], y[yc])
    res_stat.loc[st, 'kmc-km++'] = m3_KMC(X.loc[:,pd.IndexSlice[:, st, :]], y[yc], 'k-means++')
    res_stat.loc[st, 'kmc-rand'] = m3_KMC(X.loc[:,pd.IndexSlice[:, st, :]], y[yc], 'random')
    res_stat.loc[st, 'kmc-cent'] = m3_KMC(X.loc[:,pd.IndexSlice[:, st, :]], y[yc], 'centroids')

## print results
print('Results by feature:\n', res_feat)
print('Results by statistic:\n', res_stat)

## export results
res_feat.to_csv('res_feat_{}.csv'.format(yc))
res_stat.to_csv('res_stat_{}.csv'.format(yc))


# ### Final Testing

# In[296]:


m1_LDA(X.loc[:,pd.IndexSlice[:, :, :]], y['top_level'], verbose=True)
m1_LDA(X.loc[:,pd.IndexSlice[['chroma_cens','mfcc','spectral_contrast'], :, :]], y['top_level'], verbose=True)

qft = ['mfcc', 'spectral_bandwidth', 'spectral_centroid', 'spectral_contrast', 'spectral_rolloff', 'zcr']
m2_QDA(X.loc[:,pd.IndexSlice[:, ['std'], :]], y['top_level'], verbose=True)
m2_QDA(X.loc[:,pd.IndexSlice[qft, ['std'], :]], y['top_level'], verbose=True)
m2_QDA(X.loc[:,pd.IndexSlice[qft, ['mean','median','std'], :]], y['top_level'], verbose=True)

kft = ['mfcc', 'spectral_bandwidth', 'spectral_centroid', 'spectral_contrast', 'spectral_rolloff']
m3_KMC(X.loc[:,pd.IndexSlice[['mfcc'], ['skew'], :]], y['genres_all'], verbose=True)
m3_KMC(X.loc[:,pd.IndexSlice[['mfcc'], ['mean', 'median', 'skew'], :]], y['genres_all'], verbose=True)
m3_KMC(X.loc[:,pd.IndexSlice[['mfcc'], :, :]], y['genres_all'], verbose=True)

m3_KMC(X.loc[:,pd.IndexSlice[kft, ['skew'], :]], y['genres_all'], 'random', verbose=True)
m3_KMC(X.loc[:,pd.IndexSlice[kft, ['mean', 'median', 'skew'], :]], y['genres_all'], 'random', verbose=True)
m3_KMC(X.loc[:,pd.IndexSlice[kft, :, :]], y['genres_all'], 'random', verbose=True)

print('> Testing complete')

