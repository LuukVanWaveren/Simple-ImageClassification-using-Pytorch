
from torchvision import transforms

###______________________________________Settings and variables for model manipulation___###

# Number of class image folders to use from source path, starting from the first folder.
# n_Class <= available class folders. changing n_Class requires a regen for at least 1 run.
n_Class = 2
# Regenerates custom train, validation and test image folders !When true, the image folder should not be open with file browser!
regen = True

# Shows model tests and model stats and doesn't train or test data
diag = False
# Shows image transformations on random images when diag == True
testTrans = True

# Neural network learn settings
batch_size = 5
epochs = 100
lr = 0.0008
momentum = 0.9

# test a single random classification of trained network on an image from a specified class?
testClassification = True
# what class should be tested if testClassification==True? testClassification_n <= (available class folders -1).
testClassification_n = 1

# Neural network is tested on final dataset to conclude its final performance
finalTestMode = True


###______________________________________________________Other settings and variables___###

img_path_main = "Images\\"
result_path = "Results\\"
tempImg_path = img_path_main + 'ScriptImg_temp\\'

i_class = list(range(0, n_Class)) #Indexes of the image classes in the image folder that will be used

#mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # Imagenet standards
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


###______________________________________________________Don't change these settings and variables___###

cats = ['train', 'val', 'test']

t1 = transforms.Compose([
    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=280),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

t2 = transforms.Compose([
    transforms.Resize(size=300),
    transforms.CenterCrop(size=280),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])