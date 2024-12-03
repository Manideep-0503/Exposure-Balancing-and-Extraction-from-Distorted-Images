import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import matplotlib.pyplot as plt

if not os.path.exists('static'):
    os.makedirs('static')

# --- Image Processing Functions ---
# Histogram Equalization
def histogram_equalization(img_in):
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])

    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')

    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
    img_out = cv2.merge((img_b, img_g, img_r))

    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))

    return img_out, equ

# Gamma Correction
def gamma_correction(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# Log Transformation
def log_transformation(img):
    c = 255 / (np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img)
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    return log_transformed

# Contrast Stretching
def contrast_stretching(img_in):
    img_float = img_in.astype(np.float32)
    min_pixel = np.min(img_float)
    max_pixel = np.max(img_float)
    img_stretched = (img_float - min_pixel) / (max_pixel - min_pixel) * 255
    img_stretched = np.clip(img_stretched, 0, 255).astype(np.uint8)
    return img_stretched

# Adaptive Gamma Correction
def adaptive_gamma_correction(image, weighting_param=1.0):
    img_float = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    hist, bins = np.histogram(v_channel.flatten(), 256, [0, 256])
    pdf = hist / (v_channel.size)
    cdf = np.cumsum(pdf)
    cdf_normalized = cdf / cdf[-1]
    cdf_weighted = np.power(cdf_normalized, weighting_param)
    enhanced_v = np.interp(v_channel.flatten(), bins[:-1], cdf_weighted * 255.0)
    enhanced_v = enhanced_v.reshape(v_channel.shape)
    hsv[:, :, 2] = enhanced_v
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image
def plot_histograms(image, enhanced_image, title):
    """Plots histograms for the original and enhanced images.

    Args:
      image: The original image.
      enhanced_image: The enhanced image.
      title: The title for the histogram of the enhanced image.
    """
    plt.figure(figsize=(12, 6))

    # Histogram for original image
    plt.subplot(1, 2, 1)
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
    plt.hist(image[..., 1].ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
    plt.hist(image[..., 2].ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
    plt.title("Histogram of Original Image")
    plt.xlim([0, 256])
    plt.legend()

    # Histogram for enhanced image
    plt.subplot(1, 2, 2)
    plt.hist(enhanced_image.ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
    plt.hist(enhanced_image[..., 1].ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
    plt.hist(enhanced_image[..., 2].ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
    plt.title(title)
    plt.xlim([0, 256])
    plt.legend()

    plt.savefig(f'static/{title}.png')  # Save in 'static' folder
    plt.close()

# --- Homography Functions ---
def data_normalization(points):
    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)
    T = np.array([[1/std_dev[0], 0, -mean[0]/std_dev[0]],
                  [0, 1/std_dev[1], -mean[1]/std_dev[1]],
                  [0, 0, 1]])
    normalized_points = (T @ np.vstack((points.T, np.ones((1, points.shape[0]))))).T[:, :2]
    return normalized_points, T

def find_homography(points_source, points_target):
    points_source, T = data_normalization(np.array(points_source))
    points_target, Tp = data_normalization(np.array(points_target))
    A = []
    for i in range(points_source.shape[0]):
        x, y = points_source[i]
        x_t, y_t = points_target[i]
        A_for_one_point = [
            [-x, -y, -1, 0, 0, 0, x_t * x, x_t * y, x_t],
            [0, 0, 0, -x, -y, -1, y_t * x, y_t * y, y_t],
        ]
        A.extend(A_for_one_point)
    A = np.array(A)
    Htemp = np.zeros((3, 3))
    U, D, Vt = np.linalg.svd(A)
    Htemp[0, :] = Vt[-1][0:3]
    Htemp[1, :] = Vt[-1][3:6]
    Htemp[2, :] = Vt[-1][6:9]
    H = np.linalg.inv(Tp) @ Htemp @ T
    return H / H[-1, -1]

def warpPerspective(source_image, H, output_shape):
    w, h = output_shape[:2]
    y, x = np.indices((h, w))
    dst_hom_pts = np.stack((x.ravel(), y.ravel(), np.ones(y.size)))
    src_hom_pts = np.dot(np.linalg.inv(H), dst_hom_pts)
    src_hom_pts /= src_hom_pts[2]
    src_hom_pts = np.round(src_hom_pts).astype(int)
    src_x = np.clip(src_hom_pts[0], 0, source_image.shape[1] - 1)
    src_y = np.clip(src_hom_pts[1], 0, source_image.shape[0] - 1)
    warped_image = np.zeros((h, w, source_image.shape[2]), dtype=source_image.dtype)
    for ch in range(source_image.shape[2]):
        warped_image[y.ravel(), x.ravel(), ch] = source_image[src_y, src_x, ch]
    return warped_image

# --- Streamlit UI ---
st.title("Image Processing and Homography Point Selection")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    image = image.convert("RGB")  # Ensure compatibility
    image_np = np.array(image)

    # Resize the image to fit within max width and height (adjust values here)
    max_width = 800
    max_height = 600
    width, height = image.size
    aspect_ratio = height / width

    if width > max_width or height > max_height:
        if width > height:
            new_width = max_width
            new_height = int(new_width * aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height / aspect_ratio)
    else:
        new_width, new_height = width, height

    resized_image = image.resize((new_width, new_height))
    resized_image_np = np.array(resized_image)

    # Display the resized image
    st.image(resized_image, caption="Uploaded Image", use_container_width=True)

    # Dynamically set canvas size to match the resized image size
    canvas_width = resized_image.width
    canvas_height = resized_image.height

    # Source Points Selection
    st.markdown("### Select four points in same order for both source and destination")
    st.markdown("### Select Source Points")
    source_canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=resized_image,  # Use resized image
        update_streamlit=True,
        width=canvas_width,
        height=canvas_height,
        drawing_mode="point",
        key="source_canvas",
    )

    # Extract source points
    source_points = []
    if source_canvas.json_data:
        source_points = [
            (int(obj["left"]), int(obj["top"]))
            for obj in source_canvas.json_data["objects"]
            if obj["type"] == "circle"
        ]
        st.write("Source Points:", source_points)

    # Destination Points Selection
    st.markdown("### Select Destination Points")
    dest_canvas = st_canvas(
        fill_color="rgba(0, 165, 255, 0.3)",  # Transparent fill
        stroke_width=2,
        stroke_color="#0000FF",
        background_image=resized_image,  # Use resized image
        update_streamlit=True,
        width=canvas_width,
        height=canvas_height,
        drawing_mode="point",
        key="dest_canvas",
    )

    # Extract destination points
    dest_points = []
    if dest_canvas.json_data:
        dest_points = [
            (int(obj["left"]), int(obj["top"]))
            for obj in dest_canvas.json_data["objects"]
            if obj["type"] == "circle"
        ]
        st.write("Destination Points:", dest_points)

    # Apply homography
    if st.button("Apply Homography"):
        if len(source_points) == 4 and len(dest_points) == 4:
            H = find_homography(source_points, dest_points)
            
            # Use custom warpPerspective function
            transformed_image = warpPerspective(resized_image_np, H, resized_image_np.shape)

            # Resize the transformed image to fit within max width and height
            transformed_image_pil = Image.fromarray(transformed_image)

            # Display the transformed image
            st.image(transformed_image_pil, caption="Transformed Image", use_container_width=True)
        else:
            st.error("Please select exactly 4 source and 4 destination points.")

st.title("Image Processing for Exposure Correction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Select processing functionality
    functionality = st.selectbox("Select Functionality", [
        "None",
        "Histogram Equalization",
        "Gamma Correction",
        "Log Transformation",
        "Contrast Stretching",
        "Adaptive Gamma Correction"
    ])

    if functionality != "None":
        if functionality == "Histogram Equalization":
            processed_img, _ = histogram_equalization(img)
            plot_histograms(img, processed_img, "Histogram of Equalized Image")
            st.image("static/Histogram of Equalized Image.png", caption="Histograms", use_column_width=True)
        elif functionality == "Gamma Correction":
            gamma_value = st.slider("Gamma Value", 0.1, 3.0, 1.0, 0.1)
            processed_img = gamma_correction(img, gamma=gamma_value)
            plot_histograms(img, processed_img, f"Histogram of Gamma Corrected Image (gamma={gamma_value})")
            st.image(f"static/Histogram of Gamma Corrected Image (gamma={gamma_value}).png", caption="Histograms", use_column_width=True)
        elif functionality == "Log Transformation":
            processed_img = log_transformation(img)
            plot_histograms(img, processed_img, "Histogram of Log Transformed Image")
            st.image("static/Histogram of Log Transformed Image.png", caption="Histograms", use_column_width=True)
        elif functionality == "Contrast Stretching":
            processed_img = contrast_stretching(img)
            plot_histograms(img, processed_img, "Histogram of Contrast Stretched Image")
            st.image("static/Histogram of Contrast Stretched Image.png", caption="Histograms", use_column_width=True)
        elif functionality == "Adaptive Gamma Correction":
            processed_img = adaptive_gamma_correction(img)
            plot_histograms(img, processed_img, "Histogram of Adaptive Gamma Corrected Image")
            st.image("static/Histogram of Adaptive Gamma Corrected Image.png", caption="Histograms", use_column_width=True)

        # Display processed image
        st.image(processed_img, caption="Processed Image", use_column_width=True)