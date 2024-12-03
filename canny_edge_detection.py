import cv2
import numpy as np

#resizing the image if necessary
def resize_image(image, max_length=1000):
    height, width = image.shape[:2]
    if max(height, width) > max_length:
        scale = max_length / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

# initializing variables
drawing = False  
eraser_thickness = 15
erased_coords = []

def erase_object(event, x, y, flags, param):
    global drawing, erased_coords
    if event == cv2.EVENT_LBUTTONDOWN: 
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing: 
        erased_coords.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:  
        drawing = False

# Loading the image
image = cv2.imread('Messi-kick-Ball-wallpaper (1).jpg')
image = resize_image(image, max_length=1000)
original = image.copy()

#selecting region of interest
roi = cv2.selectROI("Select Region", image, showCrosshair=True, fromCenter=False)

# Extracting the selected region
x, y, w, h = map(int, roi)
selected_region = image[y:y+h, x:x+w]

#blurring and detecting the edges of the selected region of interest
gray = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow("canny_edges_detected", edges)
print("press any key to continue")
cv2.waitKey(0)


#Now we can use dialation and closing operations on the obtained edges to make them connect
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
cv2.imshow("after dialiation", closed_edges)
print("press any key to continue")
cv2.waitKey(0)

#now since the largest contour will be connected, we can proceed to create a mask
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# Extracting the object
extracted_object = cv2.bitwise_and(selected_region, selected_region, mask=mask)

while True:
    cv2.imshow("Final Output", extracted_object)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  
        cv2.imwrite('final_output.jpg', extracted_object)
        print("Image saved as final_output.jpg. Exiting...")
        break
    elif key == ord('e'):  
        print("Erasing mode activated. Draw on the output to erase.")
        cv2.setMouseCallback("Final Output", erase_object)
        #everytime anything is erased, we update it again, showing the results live
        while True:
            for coord in erased_coords:
                cv2.circle(mask, coord, eraser_thickness, 0, -1) 
                cv2.circle(extracted_object, coord, eraser_thickness, (0, 0, 0), -1)  
            
            cv2.imshow("Final Output", extracted_object)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                cv2.imwrite('final_output.jpg', extracted_object)
                print("Image saved as final_output.jpg. Exiting...")
                break
        break

cv2.destroyAllWindows()
