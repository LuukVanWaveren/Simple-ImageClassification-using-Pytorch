
###=============================================================Libraries===###

## Misc tools
import os
import random
from timeit import default_timer as timer  ## Timing utility

## Data science tools
import pandas
from PIL import Image  ## Image manipulations
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt  ## Visualizations

## IC Project modules
import IC_ImageManipulation as IM
import IC_Input as I
from IC_NetManipulation import IC_NetManipulation as NM


class ImageClassification:

    def __init__(self):

        ##_________________________________________________________Setup images___##
        print()

        self.manageImgFolders()
        self.classes = self.detectImgClassNames()

        print(f'\nAll included classes: {self.classes}')
        print()

        #show transformation on a few images as a test, if requested
        if I.testTrans == True and I.diag == True:
            self.testTransformRandomImage()

    def startLearning(self):

        ##_________________________________________________________Create datasets___##

        dataTrain = datasets.ImageFolder(root=I.tempImg_path + 'train\\', transform=I.t1)
        dataVal = datasets.ImageFolder(root=I.tempImg_path + 'val\\', transform=I.t2)
        dataTest = datasets.ImageFolder(root=I.tempImg_path + 'test\\', transform=I.t2)

        dataLoadTrain=DataLoader(dataTrain, batch_size=I.batch_size, shuffle=True)
        dataLoadVal=DataLoader(dataVal, batch_size=I.batch_size, shuffle=True)
        dataLoadTest=DataLoader(dataTest, batch_size=I.batch_size, shuffle=True)

        dataSets = [dataLoadTrain, dataLoadVal, dataLoadTest]

        ##_____________________________________Create convolutional neural network___##

        netM = NM(dataSets, self.classes)

        ##___________________________________________Start training and validating___##


        print('=======================================================================================================')
        print(f'\n\n\nStarting training and validation for {I.epochs} epochs with batch size: {I.batch_size}')

        stats=[]
        start = timer()

        for epoch in range(I.epochs):  # loop over the dataset multiple times
            stats.append(netM.trainNetOneEpoch(epoch))

        print(f'\nFinished Training with time duration of: {timer()-start:0.2f} s')

        stats = pandas.DataFrame(stats, columns=['loss_train', 'acc_train', 'loss_val', 'acc_val'])
        stats.index +=1


        ##___________________________________________Show results___##

        self.showTrainValGraph(stats)

        if I.testClassification:
            IM.testClassImage(I.testClassification_n, netM.net, I.n_Class, self.classes, netM.device, dataLoadTest, I.mean, I.std)

        if I.finalTestMode:
            netM.finalTest()

    @staticmethod
    def manageImgFolders():
        if IM.findFolder(I.tempImg_path):
            if I.regen:
                print('Folders are being regenerated, set "regen" to False to stop regeneration')
                IM.delete_folder(I.tempImg_path)
            else:
                print("""To regenerate folders with images: delete the existing folders
                ,change destination path or set "regen" to True""")

        if not IM.findFolder(I.tempImg_path):
            IM.create_imgfolder(I.img_path_main + '101_ObjectCategories\\', I.tempImg_path + I.cats[0] + '\\', I.i_class, 0, 0.5)
            IM.create_imgfolder(I.img_path_main + '101_ObjectCategories\\', I.tempImg_path + I.cats[1] + '\\', I.i_class, 0.5, 0.25)
            IM.create_imgfolder(I.img_path_main + '101_ObjectCategories\\', I.tempImg_path + I.cats[2] + '\\', I.i_class, 0.75, 0.25)

    @staticmethod
    def detectImgClassNames():
        classes = os.listdir(I.tempImg_path + I.cats[0] + '\\')
        classes.sort()
        return classes

    def testTransformRandomImage(self):
        cat = random.choice(I.cats)
        classRand = random.choice(self.classes)
        images_src = os.listdir(I.tempImg_path + cat + '\\' + classRand)
        img_path_test = I.tempImg_path + cat + '\\' + classRand + '\\' + random.choice(images_src)
        print(f'Testing transform on: {img_path_test}\n')
        IM.testTransform(img=Image.open(img_path_test), n_img=9, t=I.t1, mean=I.mean, std=I.std, title='t1')
        IM.testTransform(img=Image.open(img_path_test), n_img=9, t=I.t2, mean=I.mean, std=I.std, title='t2')

    @staticmethod
    def showTrainValGraph(stats):
        plt.figure(figsize=(8, 6))
        for label in ['loss_train', 'loss_val']:
            plt.plot(stats[label], label=label)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.show()