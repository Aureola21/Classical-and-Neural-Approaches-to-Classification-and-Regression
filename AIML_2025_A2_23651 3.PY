from re import M
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), 'A2'))
import Oracle_Assignment_2
import pickle #to cache data
import cvxopt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
#optimizer
import torch.optim as optim
import random
#Confusion Matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler

#----------------QUESTION1-------------------
cache_file="Q1data.pkl"
if os.path.exists(cache_file):
    print("Loading cached data...")
    with open(cache_file, "rb") as f:
        train_data,test_data = pickle.load(f)
else:
    print("No cache file found")
    train_data,test_data=Oracle_Assignment_2.q1_get_cifar100_train_test(23651)

    with open(cache_file, "wb") as f:
        pickle.dump((train_data, test_data), f)

print(f"Train Data Size: {len(train_data)}, Test Data Size: {len(test_data)}")
print(f"Type of train_data[0][0]: {type(train_data[0][0])}")
print(f"Type of train_data[0][1]: {type(train_data[0][1])}")
# It is not a numpy array! So ig, we need a bit of conversion here

# converting each feature and label to numpy array
train_data = [[np.array(x[0]), np.array(x[1])] for x in train_data]
test_data = [[np.array(x[0]), np.array(x[1])] for x in test_data]
# print(f"Sample Train Feature Shape: {train_data[0][0].shape}, Label Shape: {train_data[0][1].shape}") 

# Train DATA
train_features = np.array([np.array(x[0]) for x in train_data]) 
train_labels =  np.array([x[1] for x in train_data])           
print(f"Train Features Shape: {train_features.shape}")#(N, 432)
print(f"Train Labels Shape: {train_labels.shape}")#(N,)
# print(train_features)

# Test DATA
test_features = np.array([np.array(x[0]) for x in test_data])
test_labels = np.array([x[1] for x in test_data])
print(f"Test Features Shape: {test_features.shape}")#(N, 432)
print(f"Test Labels Shape: {test_labels.shape}")#(N,)
#labels are 1 and -1
# print(test_features)
# print(test_labels)

# # #-------------------------TASK1---------------------------
#Perception Algorithm

def perceptron_train(features, labels, max_iter=10000):
    Error_Counts_rate={}
    N, D = features.shape
    w = np.zeros(D)
    #print(w.shape)
    k=0
    b=0
    while k<max_iter:
        error_count=0
        for j in range(N):
            x=features[j]
            y=labels[j]
            if y*(np.dot(w.T,x)+b)<=0:
                w+=y*x
                b+=y
                error_count+=1
        # print("Iteration",k,"Error Count",error_count)
        Error_Counts_rate[k]=error_count/N
        k+=1
        if error_count==0:
            print("The Perceptron Algorithm has converged after",k,"iterations")
            return w,b,Error_Counts_rate
    print("The Perceptron Algorithm has not converged after",max_iter,"iterations")
    return w,b,Error_Counts_rate

w,b,ErrCountDict=perceptron_train(train_features,train_labels)

#-----------DELIVERABLE 1
#Plottinf Error Count vs Iteration
def plot_error_count_vs_iteration(ErrRateDict,filename):
    plt.scatter(ErrRateDict.keys(),ErrRateDict.values())
    plt.xlabel("Iteration")
    plt.ylabel("Misclassificatin Rate")
    plt.title(" Misclassification Rate Vs Iteration")
    plt.savefig(f"{filename}.png")
    plt.close()

plot_error_count_vs_iteration(ErrCountDict,"Perceptron_MiscRate_vs_Iteration")

# # #-------------------------TASK2---------------------------

#lINEAR kernel SVM

def lin_K(x,z):
    return np.dot(x,z.T)

# #primal form
# #problem: min_x (1/2)x^TPx + q^Tx
# #Constrainsts: Ax<=b
# #We have the problem: 
# #min_w,b (1/2)w^Tw + C*sum(Eta_i: i=1 to n)
# #x would be [w,b,Eta_1,.....,Eta_n]

def primal_SVM(features,labels,C=10):
    N,D=features.shape
    #For quad prog solving, we fist form the required matrices

    #P=[I 0 0
    #   0 0 0
    #   0 0 0] This is quadratic Regularization term, only w is regularized
    P = np.zeros((D + 1 + N, D + 1 + N))
    P[:D, :D] = np.eye(D)  # Regularizing only w, not b or slack variables
    P = cvxopt.matrix(P)


    #q=[0
    #   0
    #  C.I] Linear cost term, last N elements are C to penalise slack variables

    q = np.hstack([np.zeros(D + 1), C * np.ones(N)])  # Penalizing slack variables with C
    q = cvxopt.matrix(q)

    #A
    #First block enforces y_i(w^Tx_i+b)>=1, second block enforces Eta_i>=0

    bias_col = -labels.reshape(-1, 1)  # Bias term
    A1 = np.hstack([-np.dot(np.diag(labels), features), bias_col, -np.eye(N)])# SVM Constraints: y_i(w^Tx_i+b) >= 1 - ξ
    A2 = np.hstack([np.zeros((N, D + 1)), -np.eye(N)])  # ξ_i >= 0 (Slack Variables)
    A = np.vstack([A1, A2])
    A=cvxopt.matrix(A)

   

    b = np.hstack([-np.ones(N), np.zeros(N)])  # First N elements: -1, Last N elements: 0
    b = cvxopt.matrix(b)


    solution = cvxopt.solvers.qp(P, q, A,b)
    w_b = np.array(solution['x'])[:D + 1].flatten()  # Extract w and b
    w, b = w_b[:-1], w_b[-1]

    return w, b


start_time_primal=time.time()
w_primal,b_primal=primal_SVM(train_features,train_labels)
end_time_primal=time.time()
# print(f"Time taken for Primal SVM: {end_time_primal-start_time_primal}")
print(w_primal.shape)
#number of support vectors


#dual form
def dual_alpha(Kernal_Matrix,features,labels,C=10):
    N,D=features.shape

    #P=[Y_i*Y_j*K(x_i,x_j)]
    #size--> NXN
    K=Kernal_Matrix*np.outer(labels,labels)
    P=cvxopt.matrix(K)

    #q=[-1_N]
    #size--> NX1
    q=cvxopt.matrix(-np.ones(N))

    #A=[Y]
    #size--> 1XN
    A=cvxopt.matrix(labels.reshape(1,-1))

    #b=[0]
    #size--> 1X1   
    b=cvxopt.matrix(0.0)

    #G=[-I_N   0_N
    #    0_N   -I_N]
    G = cvxopt.matrix(np.vstack([np.eye(N), -np.eye(N)]))

    #h
    h = cvxopt.matrix(np.hstack([C * np.ones(N),np.zeros(N)]))

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha=np.ravel(solution['x'])

    return alpha

def dual_svm_lin(Kernel_Matrix,features,labels,C=10):
    N,D=features.shape
    alpha=dual_alpha(Kernel_Matrix,features,labels,C)

    #Finding support vectors
    indices = np.where((alpha > 0)&(alpha<C))[0]
    #print(indices)
    support_vectors = features[indices]
    support_labels = labels[indices]
    support_alpha = alpha[indices]

    w = np.sum(support_alpha[:, None] * support_labels[:, None] * support_vectors, axis=0)
    bias = np.mean(support_labels - np.dot(support_vectors, w))

    print(f"Number of support vectors: {len(support_vectors)}")
    print(f"Support Vectors Shape: {support_vectors.shape}")
    print(f"Support Labels Shape: {support_labels.shape}")
    print(f"Support Alpha Shape: {support_alpha.shape}")
    print(f"Weight Shape: {w.shape}")
    print(f"Bias: {bias}")
    return w, bias

start_time_dual=time.time()
lin_K_Mat=lin_K(train_features,train_features)
w_dual,b_dual=dual_svm_lin(lin_K_Mat,train_features,train_labels)
end_time_dual=time.time()

print()
print(f"Time taken for Primal SVM: {end_time_primal-start_time_primal}")
print(f"Time taken for Dual SVM: {end_time_dual-start_time_dual}")
print()
print("For Primal")
print(f"Weight: {w_primal}")
print(f"Bias: {b_primal}")
print()
print("For Dual")
print(f"Weight: {w_dual}")
print(f"Bias: {b_dual}")



# Non-Seperablity
#lINEAR kernel SVM
#number of support vectors
#Primal
#we need to isolate those points which have a high slack, as in the primal SVM, slack variables 
#represent how much a point violates the margin constraint
#slack variable>0 ----> y(w^Tx+b)<1 i.e. point is inside the margin and hence misclassified

def svm_non_seperable(features,labels,w,b,C=0.1):
    #print(w.shape)
    w=w.flatten()
    # print(f"Weight Shape: {w.shape}")
    margins=labels*(np.dot(features,w)+b)
    nonsep_indices=np.where(margins<1)[0]
    print(f"Number of Non-Seperable Points: {len(nonsep_indices)}")
    misclassified_indices=np.where(margins<0)[0]
    print(f"Number of Misclassified Points: {len(misclassified_indices)}")
    return nonsep_indices,misclassified_indices
print()
print("primal case")
indices_nonsep_primal,indices_misc_primal=svm_non_seperable(train_features,train_labels,w_primal,b_primal)
print()
print("Outlier Incides:", indices_nonsep_primal)
print("misclassified inidices:",indices_misc_primal)
# #Dual
# #In the dual case, we need to isolate those points which have alpha>0 and alpha<C
print("dual case")
indices_nonsep_dual,indices_misc_dual=svm_non_seperable(train_features,train_labels,w_dual,b_dual)
print("Outlier Incides:", indices_nonsep_primal)
print("misclassified inidices:",indices_misc_primal)

#submitting the indices for DUAL SVM
with open("inseperable_23651.csv","wb") as f:
    np.savetxt(f,indices_misc_dual,delimiter=",")
print()


# #-------------------------TASK3---------------------------
#Gaussian Kernel SVM
def gaussian_kernel(X,Z,gamma):
    N=X.shape[0]
    M=Z.shape[0]
    K=np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j]=np.exp(-gamma*np.linalg.norm(X[i]-Z[j])**2)
    return K

def kernal_svm(kernal_matrix,features,labels,C=10):
    N,D=features.shape
    alpha=dual_alpha(kernal_matrix,features,labels,C)

    supp_indices = np.where((alpha > 1e-7) & (alpha < C))[0]
    supp_vectors = features[supp_indices]
    supp_labels = labels[supp_indices]
    supp_alphas = alpha[supp_indices]

    Kernal_SV = kernal_matrix[np.ix_(supp_indices, supp_indices)]
    # bias=np.mean(supp_labels - np.sum(supp_alphas * supp_labels * Kernal_SV, axis=1))
    # print("shape of support labels",supp_labels.shape)
    # print("of support alphas",supp_alphas.shape)
    bias = np.mean(supp_labels - np.dot(Kernal_SV, (supp_alphas * supp_labels)))

    print(f"Number of support vectors: {len(supp_vectors)}")
    print(f"Support Vectors Shape: {supp_vectors.shape}")
    return supp_vectors, supp_labels, supp_alphas, bias

def predict_lin(test_features, w, b):
    decision_values = np.dot(test_features, w) + b
    return np.sign(decision_values)

def predict_gauss(features,supp_vectors,supp_labels,supp_alphas,bias,gamma):
    N,D=features.shape
    Kernal_SV=gaussian_kernel(features,supp_vectors,gamma)
    # prediction=np.sum(supp_alphas*supp_labels*Kernal_SV,axis=1)+bias
    prediction = np.dot(Kernal_SV, (supp_alphas * supp_labels)) + bias
    prediction=np.sign(prediction)
    return prediction

def accuracy(y_true, y_pred):
    correct=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            correct+=1
    return (correct/len(y_true))*100

def misclassification_rate(y_true, y_pred):
    return 100 - accuracy(y_true, y_pred)

#hyperParam Tuning----------------
c = 12
gamma=10
print("C:",c)
print("Gamma:",gamma)
#so basically I tried a bunch of values for C and gamma, and these values gave me the best results :)

#Need to choose Gamma in such a way that "Training set Error" becomes zero(almost) as we want data to become linearly seperable
gauss_ker_mat=gaussian_kernel(train_features,train_features,gamma)
supp_vectors_guass,supp_labels_guass,supp_alphas_guass,bias_guass=kernal_svm(gauss_ker_mat,train_features,train_labels,c)

y_pred_train_gauss=predict_gauss(train_features,supp_vectors_guass,supp_labels_guass,supp_alphas_guass,bias_guass,gamma)
print(f"Accuracy of Gaussian Kernel SVM on Training Data: {accuracy(train_labels,y_pred_train_gauss)}")
print(f"Misclassification Rate of Gaussian Kernel SVM on Training Data: {misclassification_rate(train_labels,y_pred_train_gauss)}")
print()

# --------------------------------ON TEST SET NOW
# gauss_ker_mat already computed
# supp_vectors_guass,supp_labels_guass,supp_alphas_guass,bias_guass already computed

y_pred_gauss=predict_gauss(test_features,supp_vectors_guass,supp_labels_guass,supp_alphas_guass,bias_guass,gamma)

print(f"Accuracy of Gaussian Kernel SVM: {accuracy(test_labels,y_pred_gauss)}")
print(f"Misclassification Rate of Gaussian Kernel SVM: {misclassification_rate(test_labels,y_pred_gauss)}")

# #-------------------------TASK4---------------------------

#Linear SVM primal
print("Linear SVM Primal")
#data with removed non-seperable points
train_features_sep_prim=np.delete(train_features,indices_nonsep_primal,axis=0)
train_labels_sep_prim=np.delete(train_labels,indices_nonsep_primal,axis=0)

#run perceptron on this data
print("On removing all outliers:")
w_primal_sep,b_primal_sep,ErrCountDict_sep=perceptron_train(train_features_sep_prim,train_labels_sep_prim)
plot_error_count_vs_iteration(ErrCountDict_sep,"PerceptronPrimalSep_ErrorCount_vs_Iteration")
print()

#misclassified
print("On removing all misclassified points:")
train_features_misc_prim=np.delete(train_features,indices_misc_primal,axis=0)
train_labels_misc_prim=np.delete(train_labels,indices_misc_primal,axis=0)
#run perceptron on this data
w_primal_misc,b_primal_misc,ErrCountDict_misc=perceptron_train(train_features_misc_prim,train_labels_misc_prim)
plot_error_count_vs_iteration(ErrCountDict_misc,"PerceptronPrimalMisc_ErrorCount_vs_Iteration")
print()

#Linear SVM dual
print("Linear SVM Dual")
#data with removed non-seperable points
train_features_sep_dual=np.delete(train_features,indices_nonsep_dual,axis=0)
train_labels_sep_dual=np.delete(train_labels,indices_nonsep_dual,axis=0)
print()

#run perceptron on this data
print("On removing all outliers:")
w_dual_sep,b_dual_sep,ErrCountDict_dual=perceptron_train(train_features_sep_dual,train_labels_sep_dual)
plot_error_count_vs_iteration(ErrCountDict_dual,"PerceptronDualSep_ErrorCount_vs_Iteration")
print()

#misclassified
train_features_misc_dual=np.delete(train_features,indices_misc_dual,axis=0)
train_labels_misc_dual=np.delete(train_labels,indices_misc_dual,axis=0)

#run perceptron on this data
print("On removing all misclassified points:")
w_dual_misc,b_dual_misc,ErrCountDict_misc=perceptron_train(train_features_misc_dual,train_labels_misc_dual)
plot_error_count_vs_iteration(ErrCountDict_misc,"PerceptronDualMisc_ErrorCount_vs_Iteration")

#------------------------------------------------------------

#------------------QUESTION2-------------------

if os.path.exists("q2_data"):
    print("Data already exists in folder q2_data")
else:
    print("Loading Data...")
    Oracle_Assignment_2.q2_get_mnist_jpg_subset(23651)

#Dataset is of size 10*1000 images, with each image of size 28*28, and 10 classes,
#each class having 1000 images, class labels being 0-9

#Loading the data:
transform=transforms.Compose([ transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
dataset=datasets.ImageFolder("q2_data",transform=transform)

print(f"Dataset Size: {len(dataset)}")
print("Classes: ",dataset.classes)
print(f"Image Shape: {dataset[0][0].shape}")

#Splitting the data into train and test (choosing 80-20 split)
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

#DataLoader for batch processing
train_data_loader=DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader=DataLoader(test_data, batch_size=64, shuffle=True)

#------------------TASK1-------------------

# MULTI-LAYER PERCEPTRON

class MLP(nn.Module):
    def __init__(self,input_size=784,hidden_size=128,output_size=10):
        super(MLP, self).__init__()
        self.layer1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.layer2=nn.Linear(hidden_size, output_size)
        self.softmax=nn.Softmax(dim=1) #To output class conditionals

    def forward(self, x):
        x = torch.flatten(x, 1) # Flatten the input (28x28 -> 784)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

MLP_model=MLP()

#Loss Function And Optimizer
loss_fxn=nn.NLLLoss()
optimiser=optim.SGD(MLP_model.parameters(), lr=0.001)#Adam was giving better accuracy :)

#training the model!!!!
epochs=10

for epoch in range(epochs):
    MLP_model.train()
    runningLoss=0.0
    for image,labels in train_data_loader:
        optimiser.zero_grad()
        output=MLP_model(image)
        log_output = torch.log(output)
        loss=loss_fxn(log_output,labels)#Because NLLLoss requires log probabilities
        loss.backward()
        optimiser.step()
        runningLoss+=loss.item()
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {runningLoss:.4f}/{len(train_data_loader)}")

print("MLP is hopefully trained") #fingers_crossedXX

#Testing Time
MLP_model.eval() #evaluation mode
y_mlp_true=[]
y_mlp_pred=[]
correct=0
total=0

with torch.no_grad():
    for image,labels in test_data_loader:
        output=MLP_model(image)

        #not using log output here as torch.max outputs class with max prob
        _, predicted = torch.max(output, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        y_mlp_true.extend(labels.numpy())
        y_mlp_pred.extend(predicted.numpy())

print(f"Accuracy of MLP:{correct}/{total} = {correct/total}")

#check an image pridiction
img,true_label=random.choice(test_data)
output = MLP_model(img.unsqueeze(0))
_, predicted = torch.max(output, 1)

print()
print(f"Predicted: {predicted.item()}, Actual: {true_label}")
plt.imshow(img.squeeze(0), cmap='gray')
plt.title(f"Predicted: {predicted.item()}, Actual: {true_label}")
plt.axis('off')
plt.savefig("MLP_Prediction_img.png")
print()


#------------------TASK2-------------------

#Convulational Neural Network
#Input is 28x28x1
#output is class probabilities

class CNN(nn.Module):
    def __init__(self,num_channels=1,num_classes=10):
        super(CNN, self).__init__()
        #first set of CONV=>RELU=>POOL 
        #input: 28x28x1
        #32 filters, kernel=3x3, maxpool=2x2
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
        #28x28x32
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        #14x14x32
        #second set of CONV=>RELU=>POOL
        #64 filters, kernel=3x3, maxpool=2x2
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        #14x14x64
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        #7x7x64
        #Fully connected layer
        #input: 64*7*7, neurons: 128
        self.fc1=nn.Linear(64*7*7,128)
        self.relu3=nn.ReLU()
        #neurons: 10
        #output layer
        self.fc2=nn.Linear(128,10)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.relu3(x)
        x=self.fc2(x)
        x=self.softmax(x)
        return x
    
# Device Configuration (GPU if available)
# UGCL machine was crashing on running CNN !!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

CNN_model=CNN().to(device)

#Loss Function And Optimizer
loss_fxn_cnn=nn.NLLLoss()
optimiser_cnn=optim.Adam(CNN_model.parameters(), lr=0.001)

#training the model!!!!
epochs_cnn=10

for epoch in range(epochs_cnn):
    CNN_model.train()
    runningLoss=0.0
    for image,labels in train_data_loader:
        image, labels = image.to(device), labels.to(device)
        optimiser_cnn.zero_grad()
        output=CNN_model(image)
        output=torch.log(output)
        loss=loss_fxn_cnn(output,labels)
        loss.backward()
        optimiser_cnn.step()
        runningLoss+=loss.item()
    print(f"Epoch: {epoch+1}/{epochs_cnn}, Loss: {runningLoss:.4f}/{len(train_data_loader)}")

print("CNN is trained!!!!!")

#Testing Time
CNN_model.eval() #evaluation mode
correct_cnn=0
total_cnn=0
ypred_cnn=[]
ytrue_cnn=[]

with torch.no_grad():
    for image,labels in test_data_loader:
        image, labels = image.to(device), labels.to(device)
        output=CNN_model(image)

        #not using log output here as torch.max outputs class with max prob
        _, predicted = torch.max(output, 1)
        total_cnn+=labels.size(0)
        correct_cnn+=(predicted==labels).sum().item()

        #for confusion matrix
        ypred_cnn.extend(predicted.cpu().numpy())
        ytrue_cnn.extend(labels.cpu().numpy())

print()
print("length of Train Data:",len(train_data))
print("Lenght of Test Data:",len(test_data))

print(f"Accuracy of CNN:{correct_cnn}/{total_cnn} = {correct_cnn/total_cnn}")
print()

#------------------TASK3-------------------

#Principal Component Analysis

#I already had loaded dataset as tensor in previous task
#for PCA, I would first convert it to numpy array

train_features=np.array([x[0].numpy().flatten() for x in train_data])
train_labels=np.array([x[1] for x in train_data])

test_features=np.array([x[0].numpy().flatten() for x in test_data])
test_labels=np.array([x[1] for x in test_data])

print(f"Train Features Shape: {train_features.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Features Shape: {test_features.shape}")
print(f"Test Labels Shape: {test_labels.shape}")

train_mean=np.mean(train_features,axis=0)

#Centering the data
train_features_centered=train_features-train_mean
test_features_centered=test_features-train_mean
#Covariance Matrix
cov_matrix=np.cov(train_features_centered,rowvar=False)

#Eigen Values and Eigen Vectors
eig_val,eig_vec=np.linalg.eig(cov_matrix)

#Sorting the eigen values and eigen vectors in descending order of eigen values
sort_indices=np.argsort(eig_val)[::-1]
eig_val=eig_val[sort_indices]
eig_vec=eig_vec[:,sort_indices]

#reducing the dimensionality
#Taking first 100 eigen vectors
eig_vec_200=eig_vec[:,:200]
print(f"Eigen Vectors Shape: {eig_vec_200.shape}")
#Projecting the data
train_features_pca=np.dot(train_features_centered,eig_vec_200)
test_features_pca=np.dot(test_features_centered,eig_vec_200)

print(f"Train Features PCA Shape: {train_features_pca.shape}")
print(f"Test Features PCA Shape: {test_features_pca.shape}")

# -----------------------Deliverable 1-----------------------
#Reconstructing any image of my choice :)
fav_num=21

recons_img=np.dot(train_features_pca[fav_num],eig_vec_200.T)+train_mean
recons_img=np.real(recons_img)#I was getting warning of complex numbers, so converting to real numbers
recons_img=recons_img.reshape(28,28)
plt.figure(figsize=(8, 4))

#original image
plt.subplot(1, 2, 1)
plt.imshow(train_data[fav_num][0].squeeze(0), cmap='gray')
plt.title("Original Image")
plt.axis('off')
#reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(recons_img, cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')
plt.savefig("OgVSRecons.png")
plt.close()

#-----------------------TASK4-----------------------

#Training MLP on PCA data

#FIRST OF ALL:
#converting back to tensor
#first converting to real numbers, as there was some warning of complex numbers
train_features_pca = np.real(train_features_pca)
test_features_pca = np.real(test_features_pca)

train_features_pca=torch.tensor(train_features_pca,dtype=torch.float32)
test_features_pca=torch.tensor(test_features_pca,dtype=torch.float32)
train_labels=torch.tensor(train_labels,dtype=torch.long)
test_labels=torch.tensor(test_labels,dtype=torch.long)

dataset_pca_train=torch.utils.data.TensorDataset(train_features_pca,train_labels)
dataset_pca_test=torch.utils.data.TensorDataset(test_features_pca,test_labels)

train_data_loader_pca=DataLoader(dataset_pca_train, batch_size=64, shuffle=True)
test_data_loader_pca=DataLoader(dataset_pca_test, batch_size=64, shuffle=True)

# print("Data is ready for MLP-PCA")
# print(len(train_data_loader_pca),len(test_data_loader_pca))

#MLP on PCA data
print()
MLP_model_pca=MLP(200) #200 features from PCA

#Loss Function And Optimizer
loss_fxn_pcaMLP=nn.NLLLoss()
optimiser_pcaMLP=optim.SGD(MLP_model_pca.parameters(), lr=0.001)

#training the model on data with reduced dimensionality!!!!
epochs_pcaMLP=10

for epoch in range(epochs_pcaMLP):
    MLP_model_pca.train()
    runningLoss=0.0
    for img ,label in train_data_loader_pca:
        optimiser_pcaMLP.zero_grad()
        output=MLP_model_pca(img)
        log_output = torch.log(output)#Because NLLLoss requires log probabilities
        loss=loss_fxn_pcaMLP(log_output,label)
        loss.backward()
        optimiser_pcaMLP.step()
        runningLoss+=loss.item()
    print(f"Epoch: {epoch+1}/{epochs_pcaMLP}, Loss: {runningLoss:.4f}/{len(train_data_loader_pca)}")

print("MLP on PCA data is trained")

#Testing Time
MLP_model_pca.eval() #evaluation mode
correct_pcaMLP=0
total_pcaMLP=0

ypred_pcaMLP=[]
ytrue_pcaMLP=[]

with torch.no_grad():
    for img,label in test_data_loader_pca:
        output=MLP_model_pca(img)

        #not using log output here as torch.max outputs class with max prob
        #code feels like such a repetition, but I am too lazy to refactor it
        _, predicted = torch.max(output, 1)
        total_pcaMLP+=label.size(0)
        correct_pcaMLP+=(predicted==label).sum().item()

        #for confusion matrix
        ypred_pcaMLP.extend(predicted.cpu().numpy())
        ytrue_pcaMLP.extend(label.cpu().numpy())


print(f"Accuracy of MLP on PCA data:{correct_pcaMLP}/{total_pcaMLP} = {correct_pcaMLP/total_pcaMLP}")


#-----------------------TASK5-----------------------

#Logistic Regression on PCA data
#using numpy for this task


#Logistic Regression
#We will use softmax as activation function(mUlti-class classification)
class LogisticRegression:
    def __init__(self, input_size=200, output_size=10, lr=0.001):
        self.lr = lr
        self.W = np.zeros((input_size, output_size))
        self.b = np.zeros(output_size)

    def softmax(self,x):
        expX=np.exp(x)
        return expX/np.sum(expX,axis=1,keepdims=True)
    
    def cross_entropy(self,predicted_label,true_label):
        return -np.mean(np.sum(true_label*np.log(predicted_label),axis=1))
    
    def train(self,features,labels,epochs=10):
        N=len(features)

        for epoch in range(epochs):
            logits=np.dot(features,self.W)+self.b
            predicted_labels=self.softmax(logits) #class conditionals
            loss=self.cross_entropy(predicted_labels,labels)#predicted_labels should be one hot encoded

            #Gradient Descent (Full Batch)
            #gradient calculation: Backward pass

            dW=np.dot(features.T,(predicted_labels-labels))/N
            db=np.sum(predicted_labels-labels,axis=0)/N

            #update weights
            self.W-=self.lr*dW
            self.b-=self.lr*db

            print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss:.4f}")
            

    def predict(self,features):
        output=np.dot(features,self.W)+self.b
        predicted_labels=self.softmax(output)
        return predicted_labels
    
    def accuracy(self,true_labels,predicted_labels):
        predicted_labels=np.argmax(predicted_labels,axis=1)
        true_labels=np.argmax(true_labels,axis=1)
        return np.mean(predicted_labels==true_labels)
    
    def misclassification_rate(self,true_labels,predicted_labels):
        return 1-self.accuracy(true_labels,predicted_labels)
    
    def one_hot_encode(self,labels):
        one_got_labels=np.zeros((labels.shape[0],10))
        one_got_labels[np.arange(labels.shape[0]),labels]=1
        return one_got_labels


log_reg=LogisticRegression()

#One hot encoding the labels
train_labels_one_hot=log_reg.one_hot_encode(train_labels)
test_labels_one_hot=log_reg.one_hot_encode(test_labels)

#Training the model
log_reg.train(train_features_pca.numpy(),train_labels_one_hot,epochs=10)

#Testing the model
predicted_labels=log_reg.predict(test_features_pca.numpy())
accuracy_log_reg=log_reg.accuracy(test_labels_one_hot,predicted_labels)
misclassification_rate_log_reg=log_reg.misclassification_rate(test_labels_one_hot,predicted_labels)

#For confusion matrix
ypred_log_reg=np.argmax(predicted_labels,axis=1)
ytrue_log_reg=np.argmax(test_labels_one_hot,axis=1)

print(f"Accuracy of Logistic Regression on PCA data: {accuracy_log_reg:.4f}")
print(f"Misclassification Rate of Logistic Regression on PCA data: {misclassification_rate_log_reg:.4f}")
print()

#One Vs Rest Approach
#We will train 10 models, one for each class
#For each model, we will consider that class as positive and all other classes as negative
#We will then predict the class with highest probability

class LogisticRegression_OvR:

    def __init__(self, input_size=200, num_classes=10, lr=0.001):
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = 10
        self.W = np.zeros((num_classes, input_size))  # 10x200
        self.b = np.zeros(num_classes) #10
    

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def bin_cross_entropy(self,predicted_label,true_label):
        return -np.mean(true_label*np.log(predicted_label)+(1-true_label)*np.log(1-predicted_label))
    

    def train(self,features,labels,epochs=10):
        N=len(features)
        for epoch in range(epochs):
            for c in range(self.num_classes):
                true_labels=(labels==c).astype(int)
                logits=np.dot(features,self.W[c])+self.b[c]
                predicted_labels=self.sigmoid(logits)
                loss=self.bin_cross_entropy(predicted_labels,true_labels)

                #Gradient Descent (Full Batch)
                #gradient calculation: Backward pass

                dW=np.dot(features.T,(predicted_labels-true_labels))/N
                db=np.sum(predicted_labels-true_labels)/N

                #update weights
                self.W[c]-=self.lr*dW
                self.b[c]-=self.lr*db

            print(f"Epoch: {epoch+1}/{epochs}, Class: {c}, Loss: {loss:.4f}")
    

    def predict(self,features):
        output=np.dot(features,self.W.T)+self.b
        predicted_labels=self.sigmoid(output)
        return predicted_labels
    
    def accuracy(self,true_labels,predicted_labels):
        predicted_labels=np.argmax(predicted_labels,axis=1)
        return np.mean(predicted_labels==true_labels)
    
    def misclassification_rate(self,true_labels,predicted_labels):
        return 1-self.accuracy(true_labels,predicted_labels)


    

log_reg_ovr=LogisticRegression_OvR()
train_labels_ovr=train_labels.numpy()
test_labels_ovr=test_labels.numpy()

#Training the model
log_reg_ovr.train(train_features_pca.numpy(),train_labels_ovr,epochs=10)

#Testing the model
predicted_labels_ovr=log_reg_ovr.predict(test_features_pca.numpy())
accuracy_log_reg_ovr=log_reg_ovr.accuracy(test_labels_ovr,predicted_labels_ovr)
misclassification_rate_log_reg_ovr=log_reg_ovr.misclassification_rate(test_labels_ovr,predicted_labels_ovr)

#for confusion matrix
ypred_log_reg_ovr=np.argmax(predicted_labels_ovr,axis=1)
ytrue_log_reg_ovr=test_labels_ovr

print(f"Accuracy of Logistic Regression OvR on PCA data: {accuracy_log_reg_ovr:.4f}")
print(f"Misclassification Rate of Logistic Regression OvR on PCA data: {misclassification_rate_log_reg_ovr:.4f}")
print()

#-----------------------DELIVERABLE 2-----------------------
# Print confusion matrix for all the multi class classification models and compare them using following
#metrics : Accuracy, precision, recall and F1 score for each class and compare the results.

def print_metrics(model_name,conf_mat):
    print(f"Metrics for {model_name}")
    num_classes = conf_mat.shape[0]

    #Per class metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    accuracy_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        TP = conf_mat[i][i] #TRue pos for class i
        FP = np.sum(conf_mat[:, i]) - TP #falsepos
        FN = np.sum(conf_mat[i, :]) - TP#falseneg
        TN = np.sum(conf_mat) - TP - FP - FN #true neg

        precision_per_class[i] = TP / (TP + FP)
        recall_per_class[i] = TP / (TP + FN)
        f1_per_class[i] = 2 * precision_per_class[i] * recall_per_class[i] / (precision_per_class[i] + recall_per_class[i])
        accuracy_per_class[i] = (TP + TN) / (TP + TN + FP + FN) #TN Inflates Per-Class Accuracy

    #Overall metrics
    precision_overall = np.mean(precision_per_class)
    recall_overall = np.mean(recall_per_class)
    f1_overall = np.mean(f1_per_class)
    #overall accuracy(Cannot be calculated from per class metrics)
    accuracy_overall = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

    print(f"Overall Precision: {precision_overall:.4f}")
    print(f"Overall Recall: {recall_overall:.4f}")
    print(f"Overall F1 Score: {f1_overall:.4f}")
    print(f"Overall Accuracy: {accuracy_overall:.4f}")

    print("Per Class Metrics:")
    for i in range(num_classes):
        print(f"Class {i}: Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1 Score: {f1_per_class[i]:.4f}, Accuracy: {accuracy_per_class[i]:.4f}")

    print()

print()
print("Confusion Matrix for MLP of Task1:")
conf_mat_mlp=confusion_matrix(y_mlp_true,y_mlp_pred)
print(conf_mat_mlp)
print_metrics("MLP",conf_mat_mlp)

print("Confusion Matrix for CNN of Task2")
conf_mat_cnn=confusion_matrix(ytrue_cnn,ypred_cnn)
print(conf_mat_cnn)
print_metrics("CNN",conf_mat_cnn)

print("Task3 was PCA, so no confusion matrix")
print()

print("Confusion Matrix for MLP on PCA data of Task4")
conf_mat_mlp_pca=confusion_matrix(ytrue_pcaMLP,ypred_pcaMLP)
print(conf_mat_mlp_pca)
print_metrics("MLP on PCA Data",conf_mat_mlp_pca)

print("Confusion Matrix for Logistic Regression on PCA data of Task5")
conf_mat_log_reg=confusion_matrix(ytrue_log_reg,ypred_log_reg)
print(conf_mat_log_reg)
print_metrics("Logistic Regression on PCA Data",conf_mat_log_reg)

print("Confusion Matrix for Logistic Regression OvR on PCA data of Task5")
conf_mat_log_reg_ovr=confusion_matrix(ytrue_log_reg_ovr,ypred_log_reg_ovr)
print(conf_mat_log_reg_ovr)
print_metrics("Logistic Regression OvR on PCA Data",conf_mat_log_reg_ovr)
print()


#-----------------------DELIVERABLE 3-----------------------

#We need to plot the ROC curve for each class for the Logistic Regression OvR model
#Then find the AUC for each class
#Finally, return the average AUC score


def TPR_FPR(y_true, y_scores,class_num):
    thresholds=np.sort(y_scores)[::-1]
    fpr=[]
    tpr=[]

    for threshold in thresholds:
        y_pred=(y_scores>=threshold).astype(int)

        TP=np.sum((y_pred==1)&(y_true==1))
        FP=np.sum((y_pred==1)&(y_true==0))
        TN=np.sum((y_pred==0)&(y_true==0))
        FN=np.sum((y_pred==0)&(y_true==1))

        fpr.append(FP/(FP+TN))
        tpr.append(TP/(TP+FN))
    

    return fpr,tpr

def auc_score(fpr,tpr):
    auc=0
    for i in range(1,len(fpr)):
        if fpr[i]!=fpr[i-1]:
            auc+=0.5*(tpr[i]+tpr[i-1])*(fpr[i]-fpr[i-1])

    return auc

def plot_roc(fpr, tpr, auc_score, class_num):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Class {class_num}, AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Class {class_num}")
    plt.legend()
    plt.savefig(f"ROC_Curve_{class_num}.png")
    plt.close()

#For each class
avg_auc=0
for i in range(10):
    y_true=(test_labels_ovr==i).astype(int)
    y_scores=predicted_labels_ovr[:,i]
    fpr,tpr=TPR_FPR(y_true,y_scores,i)
    auc=auc_score(fpr,tpr)
    plot_roc(fpr,tpr,auc,i)
    avg_auc+=auc
    print(f"Class {i}: AUC Score: {auc:.4f}")

avg_auc/=10 
print()
print(f"Average AUC Score: {avg_auc:.4f}")
print()

#-----------------------QUESTION3-----------------------

# QUESTION 3.1---------------------------------------
# -----------------------TASK1-----------------------
#Querrying the oracle to obtain the data

cache_file_q3_1 = "q3_data_1.pkl"
cache_file_q3_2 = "q3_data_2.pkl"
if os.path.exists(cache_file_q3_1) and os.path.exists(cache_file_q3_2):
    with open(cache_file_q3_1, "rb") as file1:
        data_1_q3 = pickle.load(file1)
    with open(cache_file_q3_2, "rb") as file2:
        data_2_q3 = pickle.load(file2)

else:
    print("Loading Data...")
    data_1_q3 = Oracle_Assignment_2.q3_linear_1(23651)
    data_2_q3 = Oracle_Assignment_2.q3_linear_2(23651)
    with open(cache_file_q3_1, "wb") as file1:
        pickle.dump(data_1_q3, file1)
    with open(cache_file_q3_2, "wb") as file2:
        pickle.dump(data_2_q3, file2)

X_train_1,y_train_1,X_test_1,y_test_1=data_1_q3
X_train_2,y_train_2,X_test_2,y_test_2=data_2_q3

print(type(X_train_1))
print(type(y_train_1))
#these are lists, so first converting them to numpy arrays

X_train_1=np.array(X_train_1)
y_train_1=np.array(y_train_1)
X_test_1=np.array(X_test_1)
y_test_1=np.array(y_test_1)

X_train_2=np.array(X_train_2)
y_train_2=np.array(y_train_2)
X_test_2=np.array(X_test_2)
y_test_2=np.array(y_test_2)

print("Dataset 1 ")
print("Shape of X train",X_train_1.shape)
print("shape of y train",y_train_1.shape)
print()
print("Dataset 2 ")
print("Shape of X train",X_train_2.shape)
print("shape of y train",y_train_2.shape)
# print(X_test.shape)
# print(y_test.shape)
print()
print("Data Loaded Successfully.......")
print()

#-----------------------TASK2-----------------------

# Linear Regression (using Ordinary Least Squares)

# For Train Dataset 1: w_1_ols
# For Train Dataset 2: w_2_ols

def LinearRegression(X,y):
    N,D=X.shape
    X_with_bias=np.hstack((X,np.ones((N,1)))) #(N,D+1)
    # print(X_with_bias)
    # print(X_with_bias.shape)
    X=X_with_bias
    # w_ols=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    #with pseudoinverse
    XTX=np.dot(X.T,X)
    XTX_inv=np.linalg.pinv(XTX)
    w_ols=np.dot(XTX_inv,np.dot(X.T,y))
    # print("Weight:", w_ols[:-1].flatten())
    # print("Bias:", w_ols[-1,0])
    return w_ols

print("Weights for Dataset 1 using OLS: ")
w_1_ols=LinearRegression(X_train_1,y_train_1)
print("Shape of Weights: ",w_1_ols.shape)
print()
print("Weights for Dataset 2 using OLS: ")
w_2_ols=LinearRegression(X_train_2,y_train_2)
print("Shape of Weights: ",w_2_ols.shape)
print()


#Ridge Regression (With Lamda=1)
def RidgeRegression(X,y,lamda=1):
    N,D=X.shape
    X_with_bias=np.hstack((X,np.ones((N,1)))) #(N,D+1), bias wd be at the end
    lamda_I=np.eye(D+1,D+1)
    lamda_I[-1,-1]=0 #bias term should not be regularized
    lamda_I*=lamda
    X=X_with_bias
    w_rr=np.dot(np.linalg.inv(np.dot(X.T,X)+lamda_I),np.dot(X.T,y))
    # print("Weight:", w_rr[:-1].flatten())
    # print("Bias:", w_rr[-1,0])
    return w_rr

print("Weights for Dataset 1 using Ridge Regression: ")
w_1_rr=RidgeRegression(X_train_1,y_train_1)
print("Shape of Weights: ",w_1_rr.shape)
print()
print("Weights for Dataset 2 using Ridge Regression: ")
w_2_rr=RidgeRegression(X_train_2,y_train_2)
print("Shape of Weights: ",w_2_rr.shape)
print()


#-----------------------TASK3-----------------------

#Calculating the mean squared error for w_1_ols and w_1_rr on the test data and train data of dataset 1
#Calculating the mean squared error for w_2_ols and w_2_rr on the test data and train data of dataset 2

#I am adding the ridge penalty to the cost function
#for OLS, lamda=0

def MeanSquaredError(X,y,w,lamda=0):
    N,D=X.shape
    # print(len(y)==N)
    X_with_bias=np.hstack((X,np.ones((N,1)))) #(N,D+1)
    X=X_with_bias
    y_pred=np.dot(X,w)
    mse=0
    for i in range(N):
        # print(f"Predicted: {y_pred[i,0]}, Actual: {y[i,0]}")
        mse+=(y_pred[i,0]-y[i])**2
    mse/=N
    ridge_penalty=lamda*np.sum(w[:-1]**2)
    print("shape of w: ",w.shape)
    mse+=ridge_penalty
    # print(mse)
    return mse


print()
mse_1_ols=MeanSquaredError(X_test_1,y_test_1,w_1_ols)
mse_1_rr=MeanSquaredError(X_test_1,y_test_1,w_1_rr,lamda=1)
mse_2_ols=MeanSquaredError(X_test_2,y_test_2,w_2_ols)
mse_2_rr=MeanSquaredError(X_test_2,y_test_2,w_2_rr,lamda=1)

train_mse_1_ols=MeanSquaredError(X_train_1,y_train_1,w_1_ols)
train_mse_1_rr=MeanSquaredError(X_train_1,y_train_1,w_1_rr,lamda=1)
train_mse_2_ols=MeanSquaredError(X_train_2,y_train_2,w_2_ols)
train_mse_2_rr=MeanSquaredError(X_train_2,y_train_2,w_2_rr,lamda=1)

#-----------------------DELIVERABLE 1-----------------------

#Can we do OLS if X doesn't have full rank?

#ANSWER: No, we cannot do OLS if X doesn't have full rank.
#WHY?
#w_ols=(X.T*X)^-1*X.T*y
#Because we need to calculate the inverse of X.T*X
#and if X doesnt have a full rank, then X.T*X will not be invertible
#and hence we cannot calculate the weights using OLS
#If X.T*X is singular, we get undefined or unstable solutions

#For Dataset 2, we observed that the MSE loss using OLS was very high.
print()
print("MSE for Dataset 2 using OLS: ")
print(f"MSE: {mse_2_ols[0]:.4f}")
print("This means, OLS provides unstable solutions for Dataset 2")
#Lets check the rank of the matrix X_train_2
rank_train_2=np.linalg.matrix_rank(X_train_2)
print("Rank of X_train_2: ",rank_train_2)
print("Number of features in X_train_2: ",X_train_2.shape[1])
print("Since rank is less than number of features, X_train_2 doesn't have full rank")
#Lets check if X.T*X is singular (X=X_train_2)
XTX=np.dot(X_train_2.T,X_train_2)
det_XTX=np.linalg.det(XTX)
print("Determinant of X.T*X: ",det_XTX)
print()
#While on the other hand, we have for Dataset 1
rank_train_1=np.linalg.matrix_rank(X_train_1)
no_features_train_1=X_train_1.shape[1]
print("Rank of X_train_1: ",rank_train_1)
print("Number of features in X_train_1: ",no_features_train_1)
print("Since rank is equal to number of features, X_train_1 has full rank")
print("so, OLS can be applied to Dataset 1")
print()
#-----------------------DELIVERABLE 2-----------------------

#MSE on test data 1
#Already computed mse_1_ols and mse_1_rr

print("MSE for Dataset 1: ")
print("Using OLS: ")
print(f"MSE on train data: {train_mse_1_ols[0]:.4f}")
print(f"MSE on test data: {mse_1_ols[0]:.4f}")
print()

print("Using Ridge Regression: (with added penalty in cost function)")
print(f"MSE on train data: {train_mse_1_rr[0]:.4f}")
print(f"MSE on test data: {mse_1_rr[0]:.4f}")
print()

#need to print w_1_ols and w_1_rr
print("Weights and bias of w_1_ols: ")
print(w_1_ols[:-1].flatten())
print(w_1_ols[-1,0])
print()
print("Weights and bias of w_1_rr: ")
print(w_1_rr[:-1].flatten())
print(w_1_rr[-1,0])

# #-----------------------DELIVERABLE 3-----------------------

#MSE on train data 2
#Already computed mse_2_ols and mse_2_rr
lamda=1

print("MSE for Dataset 2: ")
print("Using OLS: ")
print(f"MSE on train data: {train_mse_2_ols[0]:.4f}")
print("Giving stable solutions when calculated pseudo-inverse using pinv")
print(f"MSE on test data: {mse_2_ols[0]:.4f}")
print()
print("Using Ridge Regression: (with added penalty in cost function)")
print(f"MSE on train data: {train_mse_2_rr[0]:.4f}")
print(f"MSE on test data: {mse_2_rr[0]:.4f}")
print()

#need to SAVE w_2_ols and w_2_rr in csv files
np.savetxt("w_ols_23651.csv",w_2_ols,delimiter=",")
np.savetxt("w_rr_23651.csv",w_2_rr,delimiter=",")

#------------------------------------------------------------

#QUESTION 3.2---------------------------------------

#Support Vector Regression

#Querrying the Oracle to find the stock assigned
assigned_stock=Oracle_Assignment_2.q3_stocknet(23651)
print("Assigned Stock: ",assigned_stock)

#Loading the data
print()
stock_data=f"{assigned_stock}.csv"
print("Loading Data for Stock: ",assigned_stock)
data=pd.read_csv(stock_data)
print("Data Loaded Successfully")
# print(data)
# print(data.shape)

#Extract closing prices
closing_prices=data["Close"].values.reshape(-1,1) #reshaping cause standard scalar expects 2D data
# print(closing_prices)
#normalizing usind standardscaler from sklearn
scaler=StandardScaler()
d=scaler.fit_transform(closing_prices)#Nx1
# print(d)
print(d.shape)

#Preparing the data X
def prepare_X(t,d):
    # #t is the time window
    # N=len(d)
    # #form a(N-t)xN matrix
    # X=np.zeros((N-t,t))
    # #ITH ROW CONTAINS cloasing data of days from i to i+t-1
    # for i in range(N-t):
    #     X[i]=d[i:i+t].flatten()
    # return X
    N = len(d)
    return np.array([d[i:i+t].flatten() for i in range(N - t)])  # Use list comprehension


# Obtaining y by removing first t elements from d
def prepare_y(t,d):
    return np.array([d[i + t] for i in range(len(d) - t)]).flatten()

print("debuggingg...")
t = 7  # Time window
X_train = prepare_X(t, d)
y_train = prepare_y(t, d)
for i in range(5):
    print(f"X[{i}] → {X_train[i]}")
    print(f"y[{i}] → {y_train[i]}\n")

print("Shape of X_train:", X_train.shape)  # Should be (N-t, t)
print("Shape of y_train:", y_train.shape)  # Should be (N-t,)
print()
# X_i is the ith row of X
# y_i is the ith element of y
# Need to use SVR to fit a function f:R^t->R such that f(X_i)=y_i

#defining the loss function (e-sensitive)
def svr_loss_fxn(y_pred,y_true,epsilon=1e-7):
    abs_diff=np.abs(y_pred-y_true)
    loss_indices=[]
    for i in range(len(abs_diff)):
        if abs_diff[i]>epsilon:
            loss_indices.append(i)
    loss=np.mean(abs_diff[loss_indices])
    return loss


# ----------------------------------TASK1----------------------------------
# Dual Formulation of SVR, using LINEAR KERNAL
# train svm and predit it for t in [7,30,90]

# So optimization problem is to minimize the sum of this loss function over all training examples
# as well as the slack variables
# so we need to modify our dual_alpha and dual_svm functions used in question 1

# maximize alpha, alpha_star:
# -1/2 * (alpha - alpha_star)^T * K * (alpha - alpha_star) - epsilon * sum(alpha + alpha_star) + y^T * (alpha - alpha_star)
# subject to:
# 0 <= alpha_i, alpha_star_i <= C
# sum(alpha_i - alpha_star_i) = 0

def dual_alpha_svr(Kernel_Matrix, features, labels, epsilon, C):
    K = Kernel_Matrix
    print("Min eigenvalue of K:", np.min(np.linalg.eigvalsh(K))) #PSD check
    N, D = features.shape

    U=np.hstack([np.eye(N),-np.eye(N)])
    P=U.T@K@U
    P=cvxopt.matrix(P)

    q=np.zeros(2*N)
    for i in range(N):
        q[i]=-labels[i]+epsilon
        q[N+i]=labels[i]+epsilon

    q=cvxopt.matrix(q)
    A=np.zeros((1,2*N))
    for i in range(N):
        A[0,i]=1
        A[0,N+i]=-1
    A=cvxopt.matrix(A)
    b=cvxopt.matrix(0.0)

    G=np.zeros((4*N,2*N))
    h=np.zeros(4*N)

    for i in range(N):
        G[i,i]=-1
        G[N+i,i]=1
        G[2*N+i,N+i]=-1
        G[3*N+i,N+i]=1

    for i in range(N):
        h[i]=0
        h[N+i]=C
        h[2*N+i]=0
        h[3*N+i]=C

    G=cvxopt.matrix(G)
    h=cvxopt.matrix(h)

    # Debugging ranks
    print("Final Rank of P:", np.linalg.matrix_rank(np.array(P)))
    print("Final Rank of A:", np.linalg.matrix_rank(np.array(A)))
    print("Final Rank of G:", np.linalg.matrix_rank(np.array(G)))

    print("Shape of A:", A.size)
    print("Shape of b:", b.size)
    print("Shape of G:", G.size)
    print("Shape of h:", h.size)

    # Solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x'])
    alpha_pos = alphas[:N]
    alpha_neg = alphas[N:]
    # print(alpha_pos.shape)

    # Compute support vectors
    support_indices=[]
    for i in range(N):
        if C>alpha_pos[i]>0 or C>alpha_neg[i]>0:
            support_indices.append(i)
    support_indices=np.array(support_indices)

    return alpha_pos, alpha_neg, support_indices

def dual_svm_svr(kernal_matrix, features, labels,epsilon,C,ker=None):
    N,D=features.shape
    
    alpha_pos,alpha_neg,indices= dual_alpha_svr(kernal_matrix, features, labels,epsilon,C)
    print("First 10 alpha values:", alpha_pos.flatten()[:10])
    print("First 10 alpha_star values:", alpha_neg.flatten()[:10])

    #find support vectors
    support_vectors = features[indices]
    support_labels = labels[indices]

    # print("Number of support vectors:", len(support_vectors))
    # print("Support vector indices:", indices[:10])

    # Compute w from support vectors
    # w = (alpha_pos - alpha_neg) @ features
    w = (alpha_pos - alpha_neg).reshape(1, -1) @ features
    w = w.flatten()
    print(w.shape)
    #calculate b
    b = np.mean(labels[indices] - np.sum((alpha_pos[indices] - alpha_neg[indices]) * kernal_matrix[indices], axis=0).reshape(-1, 1))

    print(f"Number of support vectors: {len(support_vectors)}")
    print(f"Support Vectors Shape: {support_vectors.shape}")
    print(f"Support Labels Shape: {support_labels.shape}")
    print(f"Weight Shape: {w.shape}")
    print(f"Bias: {b}")

    if ker is None:
        return w,b
    return alpha_pos,alpha_neg, support_vectors, b

def predict_svr_linear(X_test, w, b):
    return np.dot(X_test, w) + b

print("Mean of normalized data:", np.mean(d))
print("Standard deviation of normalized data:", np.std(d))

#for deliverables
X_test_dict = {}
y_test_dict = {}
y_pred_dict = {}

for t in [7, 30, 90]:
    print(f"Time Window: {t}")
    eps=1e-3
    c=10
    print(f"Training with C={c}, epsilon={eps}")
    #Splitting Dataset into 50:50 for training and testing
    train_data_size=int(0.5*len(d))
    X_train=prepare_X(t,d[:train_data_size])
    y_train=prepare_y(t,d[:train_data_size])

    test_data_size=len(d)-train_data_size
    X_test=prepare_X(t,d[train_data_size:])
    y_test=prepare_y(t,d[train_data_size:])

    print("Shape of X_train: ",X_train.shape)
    print("Shape of y_train: ",y_train.shape)
    print("Shape of X_test: ",X_test.shape)
    print("Shape of y_test: ",y_test.shape)
    

    #training SVM:
    # X_train_normalized = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    # kernel_matrix = np.dot(X_train_normalized, X_train_normalized.T)
    kernel_matrix = np.dot(X_train, X_train.T)
    # kernel_matrix = np.dot(X_train, X_train.T)  # Linear Kernel
    print("Kernel Matrix Sample:", kernel_matrix[:5, :5])

    w_svr, b_svr = dual_svm_svr(kernel_matrix, X_train, y_train, eps,c)
    print("Weight Vector (w):", w_svr)
    print("Bias (b):", b_svr)

    #testing SVM
    y_pred_svr=predict_svr_linear(X_test,w_svr,b_svr)
    loss = svr_loss_fxn(y_pred_svr, y_test, eps)
    print(f"Loss: {loss}")
    # print("Predicted Values: ")
    # print(y_pred_svr[0:5])
    # print("Actual Values: ")
    # print(y_test[0:5])
    print()
    X_test_dict[t] = X_test
    y_test_dict[t] = y_test
    y_pred_dict[t] = y_pred_svr


# ----------------------------------TASK2----------------------------------

# Now solving the same problem using SVR with RBF Kernel

# I have already defined the guassian kernel in the previous solution
# So, I will use that kernel here

def gaussian_kernel(X,Z,gamma):
    N=X.shape[0]
    M=Z.shape[0]
    K=np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j]=np.exp(-gamma*np.linalg.norm(X[i]-Z[j])**2)
    return K

def predict_svr_gauss(X_train,X_test,alpha_pos,alpha_neg, b, gamma):
    K_test = gaussian_kernel(X_test, X_train, gamma)
    y_pred= np.dot(K_test, alpha_pos - alpha_neg) + b
    return y_pred.flatten()

#For deliverables
y_pred_dict_rbf={}
eps=1e-3
c=10
print(f"Training with C={c}, epsilon={eps}")
print()
for t in [7,30,90]:
    print(f"Time Window: {t}")
    #Splitting Dataset into 50:50 for training and testing
    train_data_size=int(0.5*len(d))
    X_train=prepare_X(t,d[:train_data_size])
    y_train=prepare_y(t,d[:train_data_size])

    test_data_size=len(d)-train_data_size
    X_test=prepare_X(t,d[train_data_size:])
    y_test=prepare_y(t,d[train_data_size:])

    print("Shape of X_train: ",X_train.shape)
    print("Shape of y_train: ",y_train.shape)
    print("Shape of X_test: ",X_test.shape)
    print("Shape of y_test: ",y_test.shape)

    y_pred_dict_rbf[t]={}

    #training SVM:
    for gamma in [1,0.1,0.01,0.001]:
        print(f"Training with gamma={gamma}")
        print()
        kernel_matrix = gaussian_kernel(X_train,X_train,gamma)  # RBF Kernel
        alpha, alpha_star, support_vectors,bias = dual_svm_svr(kernel_matrix, X_train, y_train, eps,c,ker=gaussian_kernel)
        #testing SVM
        y_pred_svr=predict_svr_gauss(X_train,X_test, alpha, alpha_star, bias,gamma)
        print("Predicted Values: ")
        print(y_pred_svr[0:3])
        print("Actual Values: ")
        print(y_test[0:3])

        W=alpha-alpha_star
        loss=svr_loss_fxn(y_pred_svr,y_test,eps)
        print(f"Loss: {loss}")

        y_pred_dict_rbf[t][gamma] = y_pred_svr
        
        print()

#----------------------------------DELIVERABLES--------------------------------

def plot_ques_3_lin(X_test,y_test,y_pred,t,name):
    avg_prices = np.mean(X_test, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Closing Price', marker='o', markersize=3)
    plt.plot(y_pred, label='Predicted Closing Price', marker='x', markersize=3)
    plt.plot(avg_prices, label=f'Average Price (Previous {t} Days)', linestyle='--', marker='.', markersize=3)
    # for gamma in Gamma_list:
    #     plt.plot(y_pred_rbf[gamma], label=f'Predicted Closing Price (RBF Kernel, γ = {gamma})', linestyle='-', marker='.', markersize=3)

    plt.title('Actual vs. Predicted Closing Prices and Average Price')
    plt.xlabel('Days')
    plt.ylabel('Normalized Closing Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"q3_plot_{t}_{name}.png")
    plt.close()
def plot_ques_3_rbf(X_test,y_pred,Gamma_list,t,name):
    avg_prices = np.mean(X_test, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Closing Price', marker='o', markersize=3)
    plt.plot(avg_prices, label=f'Average Price (Previous {t} Days)', linestyle='--', marker='.', markersize=3)
    for gamma in Gamma_list:
        plt.plot(y_pred[gamma], label=f'Predicted Closing Price (RBF Kernel, γ = {gamma})', linestyle='-', marker='.', markersize=3)

    plt.title('Actual vs. Predicted Closing Prices and Average Price')
    plt.xlabel('Days')
    plt.ylabel('Normalized Closing Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"q3_plot_{t}_{name}.png")
    plt.close()

#for linear kernel
for t in [7, 30, 90]:
    X_test = X_test_dict[t]
    y_test = y_test_dict[t]
    y_pred = y_pred_dict[t]
    plot_ques_3_lin(X_test, y_test, y_pred,t,"linear")

#for rbf kernel
Gamma_list=[1,0.1,0.01,0.001]
for t in [7, 30, 90]:
    X_test = X_test_dict[t]
    y_test = y_test_dict[t]
    y_pred = y_pred_dict_rbf[t]
    plot_ques_3_rbf(X_test, y_pred, Gamma_list, t, "rbf")

print()

# -------------------------------END-------------------------------------

# PS: This assignment was fun :)