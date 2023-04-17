
## Misc tools
import os
import shutil
import sys
import time

## Data science tools
import math
import numpy as np
import pandas
import torch
import matplotlib.pyplot as plt  ## Visualizations


def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)
    print()
    while os.path.isdir(path):
        ans = input('Deleting folders unsuccessful, retry deleting? (y/n) ')
        if ans == 'n':
            sys.exit('Script terminated')
        shutil.rmtree(path, ignore_errors=True)
        for w in range(6):
            print('.', end=" ")
            time.sleep(0.5)
    print('Folders deleted')

def findFolder(path):
    if os.path.isdir(path):
        print('Image folders found: ' + path + '\n')
        return True
    return False

def create_imgfolder(src_path, dest_path, i_class, start, size, max=40):
    """Creates a new folder using the images in all image class folders available at the source path from folder 1 to i_class.
    From each class folder, images are copied from start to start+size where start and size are percentages to be used for each class"""

    classes_src = os.listdir(src_path)
    for i in i_class:
        class1 = classes_src[i]
        os.makedirs(dest_path + class1)
        imgs = os.listdir(src_path + class1)

        if len(imgs) > max:
            imgAmount = max
        else:
            imgAmount = len(imgs)

        s = round(imgAmount * start) + 1
        a = round(imgAmount * size)
        for i in range(s, s + a):
            shutil.copy(src_path + class1 + '\\image_%04d.jpg' % i, dest_path + class1 + '\\image_%04d.jpg' % i)
    print(f'Folders with images created: {dest_path}')

def imshow(image, ax=None, title=None, mean=None, std=None):
    """show image for Tensor or normal image."""

    if ax is None:
        fig, ax = plt.subplots()

    if torch.is_tensor(image):
        image = image.numpy().transpose((1, 2, 0))

        # Reverse the preprocessing steps
        image = std * image + mean

        # Clip the image pixel values
        image = np.clip(image, 0, 1)
    elif mean is not None and std is not None:
        print("Unused arg input: mean and/or std")
    ax.title.set_text(title)
    ax.imshow(image)
    plt.axis('off')
    return ax, image

def testTransform(img, n_img, t, mean, std, title):
    """Create a transform and show resulting images as a preview. Shows the
    exact amount of transformed images according to n_img, whenever the
    squareroot of n_img is an integer"""

    fig1 = plt.figure(figsize=(10, 10), constrained_layout=True)
    fig1.suptitle(f'testTransform images with transform {title}', fontsize=16)

    s = int(math.sqrt(n_img))
    gs = fig1.add_gridspec(s + 2, s)
    imshow(img, fig1.add_subplot(gs[1, :]), title='source')
    print(f'\n{s ** 2} transformed images created for {title}:')
    for i in range(s ** 2):
        y = int(math.floor(i / s) + 2)
        x = int(i % s)
        img_t = t(img)
        name = 'img%02d' % i

        imshow(img_t, fig1.add_subplot(gs[y, x]), title=name, mean=np.array(mean), std=np.array(std))
        print([name, img_t.size()])
    plt.show()

def testClassImage(n, model, topk, classes, device, dataLoad, mean, std):
    """Picks one of the images from specified class n and shows the trained networks classification on this image.
    The class with the highest positive number is the networks answer for this image"""

    print(f'\nTesting random image from class: {classes[n]}')

    label = None
    while label != n:
        image, label = [x[0] for x in iter(dataLoad).next()]

    inputs = image.unsqueeze(0)
    inputs = inputs.to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(inputs)
        topk, topclass = outputs.cpu().topk(topk, dim=1)
        topclass = [classes[i] for i in topclass.numpy()[0]]
        topk = topk.numpy()[0]

    fig2 = plt.figure(figsize=(10, 3), constrained_layout=True)
    gs = fig2.add_gridspec(1, 3)
    imshow(image, ax=fig2.add_subplot(gs[0, 0]), title=classes[n], mean=np.array(mean), std=np.array(std))

    predTest = pandas.DataFrame({'predict': topk}, index=topclass)
    predTest.sort_values('predict').plot.barh(color='yellow', edgecolor='k', ax=fig2.add_subplot(gs[0, 1:3]))
    plt.show()