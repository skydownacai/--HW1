import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import NN_Classfier


from sklearn.metrics import accuracy_score


def show_train_history():
    files = os.listdir("models")
    train_history_dfs= []
    for file in files:
        if file.endswith("_history.csv"):
            parameters = list(map(float,file.lstrip("Model_").rstrip("_history.csv").split("_")))
            train_history_dfs.append((str(tuple(parameters)),pd.read_csv("models/" + file)))
    
    loss_dict = {"epoch" : [i for i in range(1,51)]}
    accuracy_dict = {"epoch" : [i for i in range(1,51)]}
    for  parameters,train_history_df in train_history_dfs[:-1]:
        for column in train_history_df.columns:
            if "Unnamed" in column : continue
            if "Loss" in column:
                loss_dict[parameters + "_" + column] = list(train_history_df[column].values)
            else:
                accuracy_dict[parameters + "_" + column] = list(train_history_df[column].values)

    loss_df = pd.DataFrame(loss_dict)
    loss_df.set_index("epoch",inplace= True)
    
    accuracy_df = pd.DataFrame(accuracy_dict)
    accuracy_df.set_index("epoch",inplace= True)

    #show loss curve
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("epoch vs loss")
    sns.lineplot(data = loss_df)
    plt.savefig("loss.png")

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy vs loss")
    sns.lineplot(data = accuracy_df)
    plt.savefig("accuracy.png")

def show_parameter(filepath):
    m = NN_Classfier(hidden_size=300,input_dim=784,output_dim=64).load_model(filepath)
    W_0 = m.parameters[0]
    W_1 = m.parameters[1]
    W_0 = pd.DataFrame(W_0)
    plt.figure()
    plt.title("Heatmap of W0")
    sns.heatmap(data = W_0)
    plt.savefig("W0.png")

    plt.figure()
    plt.title("Heatmap of W1")
    sns.heatmap(data = W_1)
    plt.savefig("W1.png")




if __name__ == "__main__":
    show_train_history()
    show_parameter("models/Model_300_0.1000_0.0000_best.m")

    