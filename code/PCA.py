import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import sys
import datetime
import imageio
import skimage as skim
import glob
import random
from sklearn.preprocessing import normalize
import scipy.io

random.seed(231)

####### Part 1

### 2.1 PCA: a linear method

## 1
# Split the Training and Testing Dataset
images = [mpimg.imread(file) for file in glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/Project_1/images/*.jpg')]
train = random.sample(range(1000),800)
img_train = [skim.color.rgb2hsv(images[i]/255.0) for i in train]
img_test = [skim.color.rgb2hsv(images[i]/255.0) for i in range(1000) if i not in train]

# Do the PCA on the V Channel Images

def pca(sample):
    n,_ = sample.shape
    _, s, vh = np.linalg.svd(sample,full_matrices=False)
    principles = vh.T
    eigen_value = s**2/(n-1)
    return(principles,eigen_value)

def image_normalize(sample,IMG_HEIGHT,IMG_WIDTH,channel):
    if channel==1:
            Image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            immin=(Image[:,:]).min()
            immax=(Image[:,:]).max()
            Image=(Image-immin)/(immax-immin+1e-8)
    else:
            immin=(sample[:,:,:]).min()
            immax=(sample[:,:,:]).max()
            sample=(sample-immin)/(immax-immin+1e-8)
            Image = sample
    return(Image)

def reconst_face(test_v_single,principles,n_pc,mean_train):
    test = test_v_single - mean_train
    test_v_weight = np.dot(test,principles)
    recon = mean_train + np.dot(test_v_weight[:n_pc],principles.T[:n_pc,:])
    errs = sum((recon-test_v_single)**2)
    return(recon,errs)

# Extract V Channels into a matrix
train_v = np.zeros((800,128*128))
test_v = np.zeros((200,128*128))

for i in range(800):
    v = img_train[i]
    train_v[i,:] = np.reshape(v[:,:,2],(1,-1))

for i in range(200):
    v = img_test[i]
    test_v[i,:] = np.reshape(v[:,:,2],(1,-1))

mean_train = train_v.mean(axis=0)
train_v = train_v - mean_train

# Find the Eigen-Faces
principles,_ = pca(train_v)
principles = normalize(principles,axis=1,norm='l1')

fig=plt.figure(figsize=(4, 10))
columns = 5
rows = 2
for i in range(1, columns*rows+1):
    img = image_normalize(principles[:,i],128,128,1)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img,cmap ='gray')
plt.show()

# Reconstruct faces
recon_test = np.zeros((200,128*128))
err = np.zeros(200)
for i in range(200):
    recon_test[i,:], err[i] = reconst_face(test_v[i,:],principles,50,mean_train)

fig=plt.figure(figsize=(8, 10))
columns = 5
rows = 2
test_im = img_test
for i in range(1, columns*rows+1):
    img_v = np.reshape(recon_test[i-1,],(128,128))
    test_im[i-1][:,:,2] = img_v
    img = skim.color.hsv2rgb(image_normalize(test_im[i-1],128,128,2))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

## 2

landmarks = [scipy.io.loadmat(file) for file in glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/Project_1/landmarks/*.mat')]
lm_list = []
for i in range(1000):
    lm_list.append(np.array(list(landmarks[i].values()),dtype=object)[3])
lm_list = np.array(lm_list)
lm_train = [lm_list[i] for i in train]
lm_test = [lm_list[i] for i in range(1000) if i not in train]

train_lm = np.zeros((800,68*2))
test_lm = np.zeros((200,68*2))

for i in range(800):
    v = lm_train[i]
    train_lm[i,:] = np.reshape(v,(1,-1))

for i in range(200):
    v = lm_test[i]
    test_lm[i,:] = np.reshape(v,(1,-1))

mean_lm = train_lm.mean(axis=0)
train_lm = train_lm - mean_lm
principles_lm,e_value_s = pca(train_lm)
mean_lm_pl = np.reshape(mean_lm,(68,2))
color = ['0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.9']
ax = plt.subplot(111)
ax.scatter(mean_lm_pl[:,0],mean_lm_pl[:,1],s=10)
for i in range(10):
    lm = np.reshape(principles_lm[:,i]+mean_lm,(68,2))
    ax.scatter(lm[:,0],lm[:,1],c=color[i])
bx = plt.gca()
bx.invert_yaxis()
plt.show()

err_lm = np.zeros(200)
for i in range(200):
    _, err_lm[i] = reconst_face(test_lm[i,:],principles_lm,50,mean_lm)

## 3

from mywarper import warp
from mywarper import plot

# Assigne the training and testing images to the mean landmark
train_lm_img = []
test_lm_img = []
for i in range(800):
    train_lm_img.append(warp(img_train[i],lm_train[i],mean_lm_pl))
for i in range(200):
    test_lm_img.append(warp(img_test[i],lm_test[i],mean_lm_pl))

train_v_lm = np.zeros((800,128*128))
test_v_lm = np.zeros((200,128*128))
for i in range(800):
    v = train_lm_img[i]
    train_v_lm[i,:] = np.reshape(v[:,:,2],(1,-1))
for i in range(200):
    v = test_lm_img[i]
    test_v_lm[i,:] = np.reshape(v[:,:,2],(1,-1))

# Do the PCA on the training images V channel
mean_train_lm = train_v_lm.mean(axis=0)
train_v_lm = train_v_lm - mean_train_lm
principles_v_lm,e_v_lm = pca(train_v_lm)
principles_v_lm = normalize(principles_v_lm,axis=1,norm='l1')

# Reconstruct the Testing images Landmarks by the top 10 eigen-warpings
recon_test_lm = np.zeros((200,68*2))
recon_test_lm_reshape = []

for i in range(200):
    recon_test_lm[i,:],_ = reconst_face(test_lm[i,:],principles_lm,10,mean_lm)
for i in range(200):
    recon_test_lm_reshape.append(np.reshape(recon_test_lm[i,:],(68,2)))

# Reconstruct the Testing images by the Top 50 eigen-faces
recon_test_mean_im = np.zeros((200,128*128))
recon_test_mean_im_reshape = []

for i in range(200):
    recon_test_mean_im[i,:],_ = reconst_face(test_v_lm[i,:],principles_v_lm,50,mean_train_lm)
for i in range(200):
    recon_test_mean_im_reshape.append(np.reshape(recon_test_mean_im[i,:],(128,128)))

# Plot the Reconstructed Faces
fig=plt.figure(figsize=(8, 10))
columns = 5
rows = 4
test_im = test_lm_img
for i in range(1, columns*rows+1):
    test_im[i-1][:,:,2] = recon_test_mean_im_reshape[i-1]
    img = skim.color.hsv2rgb(image_normalize(warp(test_im[i-1],mean_lm_pl,recon_test_lm_reshape[i-1]),128,128,2))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
# Plot the corresponding original faces
w=10
h=20
fig=plt.figure(figsize=(8, 10))
columns = 5
rows = 4
test_im = test_lm_img
for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(skim.color.hsv2rgb(img_test[i-1]))
plt.show()

# Plot the error

## 4
random.seed(100)
fig=plt.figure(figsize=(10, 10))
columns = 5
rows = 10
n_lm = 10
n_v = 50
for i in range(1, columns*rows+1):
    coeff_lm = np.random.normal(np.sqrt(0,e_value_s[:n_lm]),n_lm)
    coeff_v = np.random.normal(0,np.sqrt(e_v_lm[:n_v]),n_v)
    recon_lm = mean_lm+np.dot(coeff_lm[:n_lm],principles_lm.T[:n_lm,:])
    recon_lm_v = mean_train_lm+np.dot(coeff_v[:n_v ],principles_v_lm.T[:n_v ,:])
    img = np.zeros([128,128,3])
    img[:,:,2] = np.reshape(recon_lm_v,(128,128))
    img = warp(img,mean_lm_pl,np.reshape(recon_lm,(68,2)))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img[:,:,2],cmap ='gray')
plt.show()


## Part 3

# Set up the Fisherface Recognizer
recognizer = cv2.face.FisherFaceRecognizer_create()

# Load the Original Dataset and Lable the Data; Male = 1, Female = -1
images_male = [mpimg.imread(file) for file in glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/Project_1/male_images/*.jpg')]
images_female = [mpimg.imread(file) for file in glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/Project_1/female_images/*.jpg')]

train_male = random.sample(range(412),330)
train_female = random.sample(range(588),470)
img_train_male = [skim.color.rgb2hsv(images_male[i]) for i in train_male]
img_test_male = [skim.color.rgb2hsv(images_male[i]) for i in range(412) if i not in train_male]
img_train_female = [skim.color.rgb2hsv(images_female[i]) for i in train_female]
img_test_female = [skim.color.rgb2hsv(images_female[i]) for i in range(588) if i not in train_female]

train_image = img_train_male+img_train_female
test_image = img_test_male+img_test_female

train_img_v = []
test_img_v = []

for i in range(800):
    train_img_v.append(train_image[i][:,:,2])

for i in range(200):
    test_img_v.append(test_image[i][:,:,2])

landmarks_male = [scipy.io.loadmat(file) for file in glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/Project_1/male_landmarks/*.mat')]
landmarks_female = [scipy.io.loadmat(file) for file in glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/Project_1/female_landmarks/*.mat')]
lm_list_male = []
for i in range(412):
    lm_list_male.append(np.array(list(landmarks_male[i].values()),dtype=object)[3])
lm_list_male = np.array(lm_list_male)
lm_list_female = []
for i in range(588):
    lm_list_female.append(np.array(list(landmarks_female[i].values()),dtype=object)[3])
lm_list_female = np.array(lm_list_female)
lm_train_male = [lm_list_male[i] for i in train_male]
lm_test_male = [lm_list_male[i] for i in range(412) if i not in train_male]
lm_train_female = [lm_list_female[i] for i in train_female]
lm_test_female = [lm_list_female[i] for i in range(588) if i not in train_female]

train_landmark = lm_train_male+lm_train_female
test_landmark = lm_test_male+lm_test_female

label_train = np.zeros(800,dtype=int)
label_test = np.zeros(200,dtype=int)
label_train[0:330]=label_train[0:330]+1
label_train[330:800]=label_train[330:800]-1
label_test[0:82]=label_test[0:82]+1
label_test[82:200]=label_test[82:200]-1

#recognizer.train(train_img_v,label_train)
#FisherMean = recognizer.getMean()
#plt.imshow(np.reshape(FisherMean,(128,128)))
#plt.show()

#predictedLable = []
#for i in range(200):
#    predictedLable.append(recognizer.predict(test_img_v[i]))
#predictedLable = np.asarray(predictedLable)
#np.sum(predictedLable[:,0] == label_test)/2

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#clf = LinearDiscriminantAnalysis(store_covariance=True)
#train_img_v = np.asarray(train_img_v)
#train_img_v = np.reshape(train_img_v, (800,-1))
#clf.fit(train_img_v,label_train)
#w = clf.transform(train_img_v)
#w = clf.fit_transform(train_img_v,label_train)
#clf.coef_

#def calculate_boundary(X,MU_k,MU_l, SIGMA,pi_k,pi_l): 
#    return (np.log(pi_k / pi_l) - 1/2 * (MU_k + MU_l).T @ np.linalg.inv(SIGMA)@(MU_k - MU_l) + X.T @ np.linalg.inv(SIGMA)@ (MU_k - MU_l)).flatten()[0]

#mu_male, mu_female = clf.means_
#sigma = clf.covariance_
#pi_k,pi_l = clf.priors_