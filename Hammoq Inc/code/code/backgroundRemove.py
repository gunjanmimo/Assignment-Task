import cv2
import numpy as np
import matplotlib.pyplot as plt

def backgroundRemoval(imagepath):
    #Load the Image
    img = cv2.imread(imagepath)

    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width*0.7),int(height*0.7)), interpolation = cv2.INTER_AREA) 
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Create a mask holder
    mask = np.zeros(img.shape[:2],np.uint8)

    #Grab Cut the object
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #Hard Coding the Rect… The object must lie within this rect.
    rect = (10,10,width-30,height-30)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = img*mask[:,:,np.newaxis]

    #Get the background
    background = img - img1

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

    #Add the background and the image
    final = background + img1

    #To be done – Smoothening the edges….
    fig = plt.figure()
    fig.set_figwidth(18)
    a=fig.add_subplot(1, 2, 1)
    a.set_title('Original Image')
    plt.imshow(img)
    plt.axis('off')
    a=fig.add_subplot(1, 2, 2)
    a.set_title('Background Removed')
    plt.imshow(final)
    plt.axis('off')
    return final