from train import *

#Initialize the network
for hidden_size in [300,600]:
    nn = NN_Classfier(input_dim = 784, hidden_size = hidden_size, output_dim = 10)
    for lr_rate in [1e-2,1e-1]:
        for Lambda in [0,1e-4]:
            nn.initial_parameters()
            hyper_parameteres = (hidden_size,lr_rate,Lambda)
            if hyper_parameteres in [(300,1e-2,0),(300,1e-2,1e-4)]:continue
            print("Train",hyper_parameteres)
            Train(nn, train_epoch = 50, batch_size = 500,lr_rate = lr_rate, Lambda = Lambda)
