# Exposure-Balancing-and-Extraction-from-Distorted-Images

---

#### **Project Title**
Exposure Balancing and Extraction from Distorted Images

---

#### **Authors**
- Avunuri Manideep  
- Damireddy Mohith Reddy  
- Satyam Kumar  
- Yash Raj  

---

#### **Abstract**
This project focuses on addressing real-world image distortions caused by factors like tilted perspectives and poor lighting conditions. The primary aim is to extract focused objects from distorted images using a three-step approach:
1. **Homography using Direct Linear Transformation (DLT)**: Corrects distortions to align the image to the desired perspective.  
2. **Exposure Correction**: Enhances the visibility of objects using techniques such as adaptive gamma correction.  
3. **Object Extraction using GrabCut**: Accurately isolates the focused object from the background.  

---

#### **Key Features**
1. **Homography and Perspective Transformation**:
   - Establishes a relationship between source and target planes using a 3×3 homography matrix.
   - Employs Direct Linear Transformation (DLT) to compute the matrix and uses bilinear interpolation for accurate warping.

2. **Exposure Correction Techniques**:
   - **Histogram Equalization**: Enhances intensity distribution across color channels.
   - **Gamma Correction**: Adjusts brightness using an efficient LUT (Lookup Table).
   - **Log Transformation**: Compresses high intensities and expands low ones.
   - **Contrast Stretching**: Enhances visibility by spreading pixel intensity over a full range.
   - **Adaptive Gamma Correction (AGC)**: Dynamically adjusts brightness for low and high-intensity areas while preventing over-enhancement.

3. **GrabCut Algorithm**:
   - A graph-based segmentation method to separate the foreground object from the background for precise object extraction.

---

#### **Methodology**
1. **Homography for Perspective Transformation**:
   - Calculates the 3×3 matrix using DLT and SVD (Singular Value Decomposition).
   - Transforms images to rectify distortions (e.g., tilt, rotation).

2. **Exposure Correction**:
   - Implements various techniques to optimize image brightness and contrast, including histogram equalization and AGC.

3. **Object Extraction**:
   - Utilizes the GrabCut algorithm to iteratively refine and isolate the focused object.

---

#### **Results**
The approach successfully corrected distortions and enhanced visibility, enabling accurate object extraction. Results demonstrated robust handling of real-world distortions and visibility challenges.

---

#### **How to Use**
1. **Setup**:
   - Clone the repository and install dependencies (e.g., OpenCV, NumPy).
   - Provide the distorted image as input.

2. **Execution**:
   - Run the main script to process the image through homography correction, exposure adjustment, and object extraction.

3. **Output**:
   - The final output will be a distortion-free image with the focused object accurately isolated.

---

#### **Contributions**
- **Avunuri Manideep**: Research on Homography and User Interface (UI).
- **Satyam Kumar**: Research on Homography, Exposure Methods, and Adaptive Gamma Correction.
- **Damireddy Mohith Reddy**: Object extraction with the GrabCut algorithm.
- **Yash Raj**: Research on Exposure Methods, including Adaptive Gamma Correction, and UI.

---

#### **References**
1. S.-C. Huang, F.-C. Cheng, and Y.-S. Chiu, “Efficient contrast enhancement using adaptive gamma correction with weighting distribution,” IEEE Trans. Image Process., vol. 22, no. 6, pp. 2320–2331, Jun. 2013.  
2. J. Smith, “Estimating the homography matrix with the direct linear transform (DLT),” Medium, Jun. 2021.  
3. A. Author, “Image processing with Python: Image warping using homography matrix,” Medium, Oct. 2020.  

---

This project showcases innovative approaches for handling image distortions and enhancing visibility, making it a robust solution for object extraction in challenging environments.
