
## Misc tools
import sys

## Data science tools
import torch
from torch import optim, cuda
import torch.nn as nn
import pandas

## IC Project modules
import IC_Input as I
from IC_Net import Net


class IC_NetManipulation:

    def __init__(self,dataSets, classes):
        self.net, self.optimizer, self.criterion = self.createNet()
        self.useGPU()

        self.dataLoadTrain = dataSets[0]
        self.dataLoadVal = dataSets[1]
        self.dataLoadTest = dataSets[2]
        self.classes = classes

    def trainNetOneEpoch(self, epoch):
        print('\n___________________________________________________________epoch: %d' % (epoch + 1))

        loss_train_tot = 0
        acc_train_tot = 0
        datLen = 0

        self.net.train()

        for i, data in enumerate(self.dataLoadTrain, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            batchLen = list(outputs.shape)[0]
            datLen += batchLen

            loss_train_tot += batchLen * loss.item()
            loss_train = loss_train_tot / datLen

            _, pred = torch.max(outputs, dim=1)
            ans = pred.eq(labels.data.view_as(pred))
            acc = torch.mean(ans.type(torch.FloatTensor))
            acc_train_tot += batchLen * acc.item()
            acc_train = acc_train_tot / datLen

            a_stat = 1
            if i % a_stat == 0 and i != 0:  # print every a_stat batches
                print(f'\rbatch nr: {i + 1},\ttraining loss: \t\t{loss_train:.8f}', end='')
                sys.stdout.flush()
        print(f'\n\t\ttraining accuracy:\t{100 * acc_train:.2f} %')

        with torch.no_grad():
            loss_val_tot = 0
            acc_val_tot = 0
            datLen = 0
            self.net.eval()
            for i, data in enumerate(self.dataLoadVal, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                # print statistics
                batchLen = list(outputs.shape)[0]
                datLen += batchLen

                loss_val_tot += batchLen * loss.item()
                loss_val = loss_val_tot / datLen

                _, pred = torch.max(outputs, dim=1)
                ans = pred.eq(labels.data.view_as(pred))
                acc = torch.mean(ans.type(torch.FloatTensor))
                acc_val_tot += batchLen * acc.item()
                acc_val = acc_val_tot / datLen

                a_stat = 1
                if i % a_stat == 0:  # print every a_stat batches
                    print(f'\rbatch nr: {i + 1},\tvalidation loss: \t{loss_val:.8f}', end='')
                    sys.stdout.flush()
            print(f'\n\t\tvalidation accuracy:\t{100 * acc_val:.2f} %')
        return [loss_train, acc_train, loss_val, acc_val]

    def finalTest(self):
        print(f'\n___________________________________________________________test set')
        stats2 = []
        predClass = []
        labelsClass = []
        with torch.no_grad():
            loss_test_tot = 0
            acc_test_tot = 0
            datLen = 0
            self.net.eval()
            for i, data in enumerate(self.dataLoadTest, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                # print statistics

                batchLen = list(outputs.shape)[0]
                datLen += batchLen

                loss_test_tot += batchLen * loss.item()
                loss_test = loss_test_tot / datLen

                _, pred = torch.max(outputs, dim=1)
                predClass.extend([self.classes[i] for i in pred.cpu().numpy()])
                labels = labels.data.view_as(pred)
                labelsClass.extend([self.classes[i] for i in labels.cpu().numpy()])
                ans = pred.eq(labels)
                acc = torch.mean(ans.type(torch.FloatTensor))
                acc_test_tot += batchLen * acc.item()
                acc_test = acc_test_tot / datLen

                a_stat = 1
                if i % a_stat == 0:  # print every 2 mini-batches
                    print(f'\rbatch nr: {i + 1},\ttest loss: \t\t{loss_test:.8f}', end='')
                    sys.stdout.flush()

                stats2.append([loss_test, acc_test])

            print(f'\n\t\ttest accuracy:\t\t{100 * acc_test:.2f} %')

            data = {'y_Predicted': predClass, 'y_Actual': labelsClass}
            df = pandas.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
            confusion_matrix = pandas.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'],
                                               colnames=['Predicted'])
            print('\nConfusion matrix of test set\n')
            print(confusion_matrix)
            confusion_matrix.to_csv(result_path + 'testSetConfusionMatrix.csv')

    def createNet(self):
        net = Net(I.n_Class)
        print('\nNeural network created')
        optimizer = optim.SGD(net.parameters(),lr=I.lr, momentum=I.momentum)
        criterion = nn.CrossEntropyLoss()
        return net, optimizer, criterion

    def useGPU(self):
        if cuda.is_available():
            #multi_gpu = False
            self.device='cuda'
            self.net = self.net.to(self.device)
        else:
            self.device='cpu'
        print(f'Used device: {self.device}')