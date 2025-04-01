import torch
from torch import nn
from d2l import torch as d2l

def alexnet():

    net=nn.Sequential(
        nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(96,256,kernel_size=5,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(256,384,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(384,384,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(384,256,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Flatten(),
        nn.Linear(6400,4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096,10)
    )

    return net

def dataset():
    batch_size=128
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,
                                                    resize=224)

    return batch_size,train_iter,test_iter

def train():
    net=alexnet()
    batch_size,train_iter,test_iter=dataset()
    lr,numepochs=0.01,10
    d2l.train_ch6(net,
                train_iter=train_iter,
                test_iter=test_iter,
                num_epochs=numepochs,
                lr=lr,
                device=d2l.try_gpu())
    
if __name__=='__main__':
    train()