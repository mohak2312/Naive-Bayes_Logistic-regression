
import numpy as np
from sklearn.model_selection import train_test_split                                    # Library for sliting the data into train and test set
from collections import Counter                                                         # Library to count nuber of samples for each class
from sklearn.metrics import recall_score,precision_score,accuracy_score                 # Library for results
from sklearn.metrics import confusion_matrix                                            # Library for calculate the confusion matrix
from sklearn.linear_model import LogisticRegression                                     # Library for logistic regression model
import math

def get_dataset():                                                                      # Create dataset 
    data=[]
    with open("spambase.data",'r') as file:                                             # read the spambase file to extract the feaures and labels
        d1=file.read()
        d=d1.splitlines()
        for values in d:
            x=values.split(",")
            x=[float(i) for i in x]
            data.append(x)
    data_frame= pd.DataFrame(data)
    y=data_frame[len(data[0])-1].values.tolist()                                        # create a lable dataset for samples
    X=data_frame.loc[:,0:len(data[0])-2].values.tolist()                                # create a feature dataset for samples            
    return X,y,data

def split_train_test(X,y,data):                                                         # function for spliting the data into half training and half test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(len(data)/2), train_size=len(data)-int(len(data)/2),random_state=42)

    return  X_train, X_test, y_train, y_test

def get_prob(training_set):                                                             # calculate the prior probablity for training set
    prob=[]
    c=Counter(training_set)                                                             # Count the number of samles for each class
    prob.append(c[0]/len(training_set))
    prob.append(c[1]/len(training_set))
    print(prob)
    return prob

def seprate_data(X_train,y_train):                                                      # Separate the samples according to the class
    train_one=[]
    train_zero=[]
    for c,i in enumerate(y_train):
        
        if(i==1):
            train_one.append(X_train[c])                                                # create a dataset for class 1(spam)
        else:
            train_zero.append(X_train[c])                                               # create a dataset for class 0(not spam)
    return train_one,train_zero


def cal_mean_std(train_one,train_zero):                                                 # calculate the mean and standard deviation for training set
    train_one=np.array(train_one)                                                       # convert class 1 and clas 0 dataset into numpy array
    train_zero=np.array(train_zero)
    mean_one=np.mean(train_one,axis=0)                                                  # calculate the mean for both of the class
    mean_zero=np.mean(train_zero,axis=0)
    std_one=np.std(train_one,axis=0)                                                    # calculate the stadard deviation for both class
    std_zero=np.std(train_zero,axis=0)
    std_one=std_one+0.0001                                                              # Adding small value epsilon=0.0001 to std for each class
    std_zero=std_zero+0.0001                                                                        
   
    return mean_one,mean_zero,std_one,std_zero


def cal_prob(x,mean,stdev):                                                             # calculate the probablity density functiion
    exponent= np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
    return (1 / (np.sqrt(2*math.pi) * stdev)) * exponent

def predict(X_test,y_test,prob,mean_one,mean_zero,std_one,std_zero):                    # Function for testing the test dataset to predict th class
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    y_predict=[]
    for i in range(len(X_test)):
        Class=np.zeros(2)
        pdf_zero=cal_prob(X_test[i],mean_zero,std_zero)                                 # calculate the probablity density finction for class 0
        pdf_zero=np.append(pdf_zero,prob[0])
        
        Class[0]=np.sum(np.log(pdf_zero))                                               # calculate the log of product of 
        pdf_one=cal_prob(X_test[i],mean_one,std_one)                                    # calculate the probablity density finction for class 1
        pdf_one=np.append(pdf_one,prob[1])
    
        Class[1]=np.sum(np.log(pdf_one))                                                # calculate the log of product
        y_predict.append(np.argmax(Class))                                              # find the maximum probability for class
    return y_predict

def cal_results(y_test,predict_lable):                                                  # calculate the precision,recall, accuracy and confusion matrix
    Precision = precision_score(y_test,predict_lable)
    Recall = recall_score(y_test,predict_lable)
    Accuracy = accuracy_score(y_test,predict_lable)
    c=confusion_matrix(y_test, predict_lable)
    return Precision,Recall,Accuracy,c
    

def Logestic_regression(X_train, X_test, y_train, y_test):                              # finction for classification using logistic regression
                                                                                        # use logistic regression class for th classification                                                                                       #
    clf = LogisticRegression(random_state=0, solver='lbfgs',max_iter=3000 ,multi_class='multinomial').fit(X_train, y_train)
    predicted_label=clf.predict(X_test)                                                 # predict the class 
    Precision,Recall,Accuracy,c=cal_results(y_test,predicted_label)                     # calculate the results
    print("Precision :",Precision)
    print("Recall :",Recall)
    print("Accuracy :",Accuracy)
    print("Confusion Matrix :\n",c)


print("|----------------- Experiment 1 -------------------|")
X,y,data=get_dataset()                                                                  # get the dataset from file
X_train, X_test, y_train, y_test=split_train_test(X,y,data)                             # split dataset into test set and train set
prob=get_prob(y_train)                                                                  # calculate the prior probablity for training set
train_one,train_zero=seprate_data(X_train,y_train)                                      # seerate the dataset according to each class 
mean_one,mean_zero,std_one,std_zero=cal_mean_std(train_one,train_zero)                  # calculate the mean and standard deviation for traing set
predict_lable=predict(X_test,y_test,prob,mean_one,mean_zero,std_one,std_zero)           # Predic the class for test data
Precision,Recall,Accuracy,c=cal_results(y_test,predict_lable)                           # calculate the results
print("Precision :",Precision)
print("Recall :",Recall)
print("Accuracy :",Accuracy)
print("Confusion Matrix :\n",c)
print("|----------------- Experiment 2 -------------------|")
Logestic_regression(X_train, X_test, y_train, y_test)                                   # calssify the data using logistic regression

    

