import dataloader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix

class EEGNet(nn.Module):
    def __init__(self, activation=None, dropout=0.65):
        super(EEGNet, self).__init__()

        if not activation:
            # Default activation function is ELU
            activation = nn.ELU

        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=(1, 51),
                stride=(1,1), 
                padding=(0, 25), 
                bias=False
            ),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, 4), 
                stride=(1, 4), 
                padding=0
            ),
            nn.Dropout(p=dropout)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            )
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, 
                      out_features=2, 
                      bias=True
            )
        )

    def forward(self, input):
        x = self.firstConv(input)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)

        # Flatten
        x = x.view(-1, self.classify[0].in_features)

        self.output = self.classify(x)

        return self.output


class DeepConvNet(nn.Module):
    def __init__(self, activation=None, dropout=0.1):
        super(DeepConvNet, self).__init__()

        sizes = [25, 50, 100, 200]
        self.convSizes = [convSizes for convSizes in zip(sizes[:-1], sizes[1:])]

        if not activation:
            # Default activation function is ELU
            activation = nn.ELU

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=25, 
                kernel_size=(1, 5),                
                stride=(1,1), 
                padding=(0, 0), 
                bias=False
            ),
            nn.Conv2d(
                in_channels=25, 
                out_channels=25, 
                kernel_size=(2, 1),                
                stride=(1,1), 
                padding=(0, 0), 
                bias=False
            ),
            nn.BatchNorm2d(25), # C * out_features(?)
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        for convID in range(len(self.convSizes)):
            self.add_module(
                "conv" + str(convID+1),
                nn.Sequential(
                    nn.Conv2d(
                    in_channels=self.convSizes[convID][0],
                    out_channels=self.convSizes[convID][1],
                    kernel_size=(1, 5),
                    stride=(1,1), 
                    padding=(0, 0), 
                    bias=True
                    ),
                    nn.BatchNorm2d(self.convSizes[convID][1]), # C * out_features(?)
                    activation(),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                    nn.Dropout(p=dropout)
                )
            )
        
        self.classify = nn.Sequential(
            nn.Linear(
                in_features=8600, 
                out_features=2, 
                bias=True
            )
        )


    def forward(self, input):
        x = self.conv0(input)
        for convID in range(1, len(self.convSizes)+1):
            x = getattr(self, 'conv'+str(convID))(x)

        x = x.view(-1, self.classify[0].in_features)
        self.output = self.classify(x)
        
        return self.output


def trainAndTestModels(train_data, train_label, test_data, test_label,
              models, epochs, batch_size, learning_rate, optimizer=optim.Adam, loss_function=nn.CrossEntropyLoss()):
    accuracyOfTrainingPerModels = {}
    
    for key, model in models.items():
        # Initialize the trainAcc array as zeros
        trainAcc = np.zeros(epochs)
        testAcc = np.zeros(epochs)

        # train the models
        for epoch in range(epochs):
            train_inputs = torch.Tensor(train_data).to(device)
            train_labels = torch.Tensor(train_label).to(device).long().view(-1)

            optimizer(model.parameters(), lr=learning_rate).zero_grad()
            outputs = model.forward(train_inputs)
            loss = loss_function(outputs, train_labels)
            print("Epochs[%3d/%3d] Loss : %f" % (epoch, epochs, loss))
            loss.backward()

            # Pick the maximum value of each row and return its index
            trainAcc[epoch] = (torch.max(outputs, 1)[1] == train_labels).sum().item() * 100 / len(train_label)
            
            accuracyOfTrainingPerModels.update([(key+'_train', trainAcc)])
            # Update the parameters
            optimizer(model.parameters(), lr=learning_rate).step()
            
            # test the models
            with torch.no_grad():
                test_inputs = torch.Tensor(test_data).to(device)
                test_labels = torch.Tensor(test_label).to(device).long().view(-1)

                outputs = model.forward(test_inputs)
                testAcc[epoch] = (torch.max(outputs, 1)[1] == test_labels).sum().item() * 100 / len(test_label)
                
                accuracyOfTrainingPerModels.update([(key+'_test', testAcc)])

        plotConfusionMatrix(key, test_labels.to(torch.device('cpu')).numpy(), torch.max(outputs, 1)[1].to(torch.device('cpu')).numpy())

    return accuracyOfTrainingPerModels


def plotTheResult(network, accuracy):
    plt.figure(figsize=(8,4.5))
    plt.title('Activation function comparision (' + network + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.ylim(50, 100)

    for key, value in accuracy.items():
        plt.plot(
            value, 
            '--' if 'test' in key else '-',
            label=key
        )
        plt.legend(loc='lower right')

    #plt.show()


def plotConfusionMatrix(title, test_y, pred_y, normalize=True):
    cm = confusion_matrix(test_y, pred_y)
    np.set_printoptions(precision=2)
    
    plt.figure()
    classes = ['0', '1']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    filename = title + "_ConfusionMatrix"
    plt.savefig(filename + ".png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    models = {
        "ELU" : EEGNet(nn.ELU).to(device),
        "ReLU" : EEGNet(nn.ReLU).to(device),
        "LeakyReLU" : EEGNet(nn.LeakyReLU).to(device)
    }

    EEGNetAcc = trainAndTestModels(train_data, train_label, test_data, test_label, 
        models, epochs=500, batch_size=64, learning_rate=1e-3)

    for key, value in EEGNetAcc.items():
    	print(key, np.amax(value))

    plotTheResult("EEGNet", EEGNetAcc)


    models = {
        "ELU" : DeepConvNet(nn.ELU).to(device),
        "ReLU" : DeepConvNet(nn.ReLU).to(device),
        "LeakyReLU" : DeepConvNet(nn.LeakyReLU).to(device)
    }

    DeepConvNetAcc = trainAndTestModels(train_data, train_label, test_data, test_label, 
        models, epochs=500, batch_size=64, learning_rate=1e-3)

    for key, value in DeepConvNetAcc.items():
    	print(key, np.amax(value))

    plotTheResult("DeepConvNet", DeepConvNetAcc)