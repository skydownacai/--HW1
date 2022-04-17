import numpy as np
import minst_read as m
from model import NN_Classfier
from random import shuffle
from tqdm import tqdm
import pandas as pd




def Metric(pred: np.array, truth: np.array, truth_val : int = 1):

    TP = ((pred == truth_val) & (truth == truth_val)).sum()
    TN = ((pred != truth_val) & (truth != truth_val)).sum()
    FN = ((pred != truth_val) & (truth == truth_val)).sum()
    FP = ((pred == truth_val) & (truth != truth_val)).sum()

    p = 0 if TP == FP == 0 else TP / (TP + FP)
    r = 0 if TP == FN == 0 else TP / (TP + FN)
    F1 = 0 if p == r == 0 else 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return (p, r, F1, acc)


def get_minist_data():
    train_X = []
    train_Y = []
    train_label = []
    test_X  = []
    test_Y  = []
    test_label = []

    train_images = m.load_train_images()
    train_labels = m.load_train_labels()
    test_images =  m.load_test_images()
    test_labels =  m.load_test_labels()

    for i in range(len(train_images)):
        train_X.append(train_images[i].reshape(1,784)/256)
        y = np.zeros((1,10))
        y[0,int(train_labels[i])] = 1
        train_Y.append(y)
        train_label.append(int(train_labels[i]))

    for i in range(len(test_images)):
        test_X.append(test_images[i].reshape(1,784)/256)
        y = np.zeros((1,10))
        y[0,int(test_labels[i])] = 1
        test_Y.append(y)
        test_label.append(int(test_labels[i]))

    train_X = np.vstack(train_X)
    train_Y = np.vstack(train_Y)
    test_X = np.vstack(test_X)
    test_Y = np.vstack(test_Y)

    train_label = np.array(train_label).reshape(-1,1)
    test_label = np.array(test_label).reshape(-1,1)

    return (train_X,train_Y,train_label,test_X,test_Y,test_label)


def extract_info(predictions, labels, losses):

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    losses = np.vstack(losses)

    accuracy = (predictions == labels).sum() / len(predictions)
    avg_loss = np.sum(losses)/len(losses)

    return accuracy,avg_loss




def Train(Network : NN_Classfier,train_epoch : int, batch_size :int, lr_rate : float, Lambda : float):

    '''
    Tranning the model in the minist dataset
    '''
    train_X,train_Y,train_label,test_X,test_Y,test_label = get_minist_data()

    train_history = {"Train_Loss":[],"Test_Loss":[],"Train_Accuracy":[],"Test_Accuracy":[]}

    best_test_accuracy = 0

    model_name = "Model_%d_%.4f_%.4f" % (Network.hidden_size,lr_rate,Lambda)

    for epoch in range(train_epoch):

        # trainning entire data
        train_idxs = list(range(len(train_X)))
        shuffle(train_idxs)
        p = 0 
        Network.set_train_mode()

        predictions = []
        labels = []
        losses = []

        for c in tqdm(range(int(len(train_idxs)/batch_size) + 1),desc = "Train Epoch %d" % epoch):
            p = c * batch_size
            if p >= len(train_idxs): break

            batch_idxs = train_idxs[p : p + batch_size]
            batch_X = train_X[batch_idxs]
            batch_Y = train_Y[batch_idxs]
            batch_label = train_label[batch_idxs]

            #update 
            Network.zero_grad()
            prob = Network.forward(batch_X)
            Network.backward(batch_Y,Lambda)
            Network.step(lr_rate)

            #record
            loss = Network.loss_fn(batch_Y,prob)
            batch_prediction = np.argmax(prob,axis= -1).reshape(-1,1)
            predictions.append(batch_prediction)
            labels.append(batch_label)
            losses.append(loss)
            
        accuracy,avg_loss = extract_info(predictions,labels,losses)
        print("Train Epoch %d" % epoch,"Accuracy:%.4f" % accuracy,"Avg_losses:%.4f" % avg_loss)

        train_history["Train_Loss"].append(avg_loss)
        train_history["Train_Accuracy"].append(accuracy)

        #test
        test_idxs = list(range(len(test_X)))
        predictions = []
        labels = []
        losses = []

        Network.set_eval_mode()

        for c in tqdm(range(int(len(test_idxs)/batch_size) + 1),desc = "Test Epoch %d" % epoch):
            p = c * batch_size
            if p >= len(test_idxs): break

            batch_idxs = test_idxs[p : p + batch_size]
            batch_X = test_X[batch_idxs]
            batch_Y = test_Y[batch_idxs]
            batch_label = test_label[batch_idxs]

            #update 
            prob = Network.forward(batch_X)

            #record
            loss = Network.loss_fn(batch_Y,prob)
            batch_prediction = np.argmax(prob,axis= -1).reshape(-1,1)
            predictions.append(batch_prediction)
            labels.append(batch_label)
            losses.append(loss)

        accuracy,avg_loss = extract_info(predictions,labels,losses)
        print("Test Epoch %d" % epoch,"Accuracy:%.4f" % accuracy,"Avg_losses:%.4f" % avg_loss)

        if accuracy > best_test_accuracy:
            best_test_accuracy = accuracy
            Network.save_model("models/%s_Best.m" % model_name)
        
        train_history["Test_Loss"].append(avg_loss)
        train_history["Test_Accuracy"].append(accuracy)

        Network.save_model("models/%s_latest.m" % model_name)

        df = pd.DataFrame(train_history)
        df.to_csv("models/%s_history.csv" % model_name)


