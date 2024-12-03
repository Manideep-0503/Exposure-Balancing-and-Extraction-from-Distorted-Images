import cv2
import numpy as np


drawing = False
which_one = 'rect'
ix, iy = -1, -1
rect = None
rect_defined = False

#Now for resizing the image of highier quality for displaying it properly in a window
def resize_image(image, max_length=600):
    height, width = image.shape[:2]
    if max(height, width) > max_length:
        scale = max_length / max(height, width)
        adjusted_width = int(width * scale)
        adjusted_height = int(height * scale)
        return cv2.resize(image, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
    return image

# Image reading
img = cv2.imread('Messi-kick-Ball-wallpaper (1).jpg') 
img = resize_image(img, max_length=600) 
original_img = img.copy()
img_copy = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # We have to initialise the mask for grab cut processing


# Everytime this function gets called when ever there is some mouse movement on the image, 
def draw(mouse_work, x, y, flags, param):
    global drawing, which_one, ix, iy, mask, img, img_copy, rect, rect_defined

    if mouse_work == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif mouse_work == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if which_one == 'fg':  # Probable foreground marking
                cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            elif which_one == 'rect':
                img_copy = img.copy()
                cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
                cv2.imshow('Image', img_copy)
            elif which_one == 'bg':  # Probable background marking
                cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    elif mouse_work == cv2.EVENT_LBUTTONUP:
        drawing = False
        if which_one == 'rect':
            rect = (ix, iy, x, y)
            rect_defined = True
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 2)


cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw)
print("- f for probable foreground")
print("- b for probable background")
print("- g for applying grabcut")
print("- r to reset")
print("- q to quit and save")

#showing the image continuosly untill the code is exited and the image detects if there is any waitkey
while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key == ord('r'):  # Reset
        img = original_img.copy()
        img_copy = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        rect = None
        rect_defined = False
        print("Reset.")
    elif key == ord('f'):  
        which_one = 'fg'
        print("Mode: Mark probable foreground")
    elif key == ord('b'): 
        which_one = 'bg'
        print("Mode: Mark probable background")
    elif key == ord('g'): 
        if rect_defined:
            back = np.zeros((1, 65), np.float64)
            fore = np.zeros((1, 65), np.float64)
            x1, y1, x2, y2 = rect
            cv2.grabCut(original_img, mask, (x1, y1, x2 - x1, y2 - y1), back, fore, 5, cv2.GC_INIT_WITH_RECT)
        
        back = np.zeros((1, 65), np.float64)
        fore = np.zeros((1, 65), np.float64)
        cv2.grabCut(original_img, mask, None, back, fore, 5, cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        result = original_img * mask2[:, :, np.newaxis]

        img = original_img.copy()
        print("GrabCut applied.")

        cv2.imshow('Segmented', result)
    elif key == ord('q'):
        cv2.imwrite('grab_cut_output.jpg', result)
        print("Image saved as grab_cut_output.jpg. Exiting...")
        break  

cv2.destroyAllWindows()

