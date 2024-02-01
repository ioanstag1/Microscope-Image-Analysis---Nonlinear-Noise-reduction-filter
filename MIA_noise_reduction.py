import cv2
import numpy as np

#print the image with salt and pepper noise
originalimg=cv2.imread('5.png')
cv2.namedWindow('Original Image')
cv2.imshow('Original Image',originalimg)
cv2.waitKey(0)


#function for clearing salt and pepper noise
def medianfilter(filename):
 img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
 rows, cols = img.shape
 img_new = np.zeros([rows, cols])
 kernel = np.zeros((3, 3))
 krows, kcols = kernel.shape
 medianrow = krows // 2
 mediancol = kcols // 2
 # create the  kernel
 for i in range(1,rows - 1):
        for j in range(1,cols - 1 ):
            for r in range(krows):
                for c in range(kcols):
                    kernel[r, c] = img[i + r-medianrow, j + c-mediancol]
            # find the median value of the kernel
            k = krows * kcols  # size of kernel
            kernel1D = kernel.reshape(1, k)  # reshape the kernel to sort it
            sorted_kernel = np.sort(kernel1D)
            # find the median value and put it on the mid of the kernel
            med = k // 2
            medianvalue = sorted_kernel[(0, med)]
            img_new[i, j] = medianvalue  # put the pixel of median in the new img

 smoothedimg = img_new.astype(np.uint8)
 cv2.imwrite('smoothedimg.png',smoothedimg)
 cv2.namedWindow('medianfilter')
 cv2.imshow('medianfilter', smoothedimg)
 cv2.waitKey(0)
 
 #call the function for smoothing
medianfilter('5.png')

#read the smoothed img
img=cv2.imread('smoothedimg.png')
# Make a grayescale copy
new_img = cv2.imread('smoothedimg.png', cv2.IMREAD_GRAYSCALE)

#create the integral image
def integralimg(image,x0,y0,x1,y1):
 m, n = image.shape
 Summed_Area_Table = np.zeros([m+1, n+1])
 sum = 0
 for x in range(1, m+1):
     for y in range(1, n+1):
             sum = Summed_Area_Table[x - 1, y] + Summed_Area_Table[x, y - 1] - Summed_Area_Table[x - 1, y - 1] + image[x-1, y-1]
             Summed_Area_Table[x, y] = sum

 # calculate the sum for [x0,y0]-[x1,y1]->bounding box
 A = Summed_Area_Table[y0, x0]
 B = Summed_Area_Table[y1, x0]
 C = Summed_Area_Table[y0, x1]
 D = Summed_Area_Table[y1, x1]
 s = A + D - C - B
 return(s)

#create the binary img
_, binary = cv2.threshold(new_img, 20, 255, cv2.THRESH_BINARY )
cv2.namedWindow('binaryimg')
cv2.imshow('binaryimg', binary)
#cv2.imwrite('binaryimg.png',binary)
cv2.waitKey(0)

#create a function that colors the labels with unique color
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.namedWindow('labeled', cv2.WINDOW_NORMAL)
    cv2.imshow('labeled', labeled_img)
    #cv2.imwrite('labeled.png',labeled_img)


#find the connected components
num_cc, labeled, stats, centroids = cv2.connectedComponentsWithStats(binary)
#call the function to color the labels
imshow_components(labeled)
cv2.waitKey(0)


for i in range(1,num_cc):
    area = stats[i, cv2.CC_STAT_AREA]
    # check component area to be large enough
    if area > 50:
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        (cX, cY) = centroids[i]#contains the center of the component
        x1=x0+w
        y1=y0+h


        print("--Region",str(i),"--")
        print("Area(px):", area)
        # total pixels of bounding box can be found
        # by Multiply the width of the bounding box by the height of the bounding box
        total_pixels_of_bounding_box = h*w
        print("Bounding Box Area:", total_pixels_of_bounding_box)
        #call the function for integral image
        sumofboundinbox=integralimg(new_img,x0,y0,x1,y1)
        #mean gray level if the sum/the total pixels of the bounding box
        print('Mean gray level in bounding box:', sumofboundinbox / total_pixels_of_bounding_box)
        print('')

    # draw bounding boxes over contours
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    # label the bounding boxes with their number i
        cv2.putText(img, "{}".format(i), (int(cX), int(cY)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow('All connected components with bounding box and labeled', img)
#cv2.imwrite('All connected components with bounding box and labeled.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

