import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def find_pupil(image):
    if peak_detected(image):
        edges = edge_pupil_center_detection(image)
        if edges is not None:
            ellipsis, center = edges
            return ellipsis, center
        else:
            # Thresholding and coarse positioning
            return None
    else:
        # Thresholding and coarse positioning
        return None


# Returns True if there is a peak in the histogram of an image
def peak_detected(image, th=200, mu=10):
    # Calculate the histogram of the image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Calculate the average intensity of the image
    avg_intensity = np.mean(image)

    # If there is a peak in the histogram, return True
    for i in range(256):
        if hist[i] > th and hist[i] > mu * avg_intensity:
            return True

    return False


# Returns the ellipsis that best fits the pupil with its center, or None if no pupil is found
def edge_pupil_center_detection(image):
    # Apply Gaussian filtering for smoothing
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(smoothed, 50, 100)  # Adjust the threshold as needed

    # Find the contours of the edges
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the ellipses that best fit the contours
    ellipses = []
    for contour in contours:
        if len(contour) >= 30:
            ellipse = cv2.fitEllipse(contour)

            # Unpack the ellipse parameters
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse

            # Convert the parameters to integer values
            center = (int(center_x), int(center_y))
            axes = (int(major_axis / 2), int(minor_axis / 2))

            # Calculate the size of the square based on a scale factor
            scale_factor = 0.4
            square_size = int(min(axes) * scale_factor)

            # Define the coordinates of the square
            x = center[0] - square_size // 2
            y = center[1] - square_size // 2

            # Extract the square region of interest (ROI) from the image
            roi = image[y : y + square_size, x : x + square_size]

            # Calculate the percentage of black pixels within the ROI
            black_pixels = np.sum(roi < 45)
            total_pixels = roi.size
            if total_pixels == 0:
                black_pixels_percentage = 0
            else:
                black_pixels_percentage = black_pixels / total_pixels

            # If the percentage of black pixels is greater than 90%, then the ellipse is a pupil
            if black_pixels_percentage > 0.9:
                ellipses.append(ellipse)

    # If no ellipses are found, return None. Otherwise, return the largest ellipse with its center
    if len(ellipses) == 0:
        return None
    else:
        return find_largest_ellipsis(ellipses)


# Returns the largest ellipse with its center
def find_largest_ellipsis(ellipses):
    # Initialize variables for tracking the largest ellipse
    largest_area = 0
    largest_ellipse = None
    center = None

    for ellipse in ellipses:
        # Unpack the ellipse parameters
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse

        # Convert the parameters to integer values
        c = (int(center_x), int(center_y))
        axes = (int(major_axis / 2), int(minor_axis / 2))

        # Calculate the area of the ellipse contour
        contour_area = np.pi * (axes[0] / 2) * (axes[1] / 2)

        # Check if the current ellipse has a larger area than the previous largest ellipse
        if contour_area > largest_area:
            largest_area = contour_area
            largest_ellipse = ellipse
            center = c

    return largest_ellipse, center


# Draw the ellipsis on the image
def draw_ellipsis(image: np.ndarray, ellipsis):
    if ellipsis is not None:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.ellipse(image, ellipsis, (255, 0, 0), 2)
    return image


def process_image(plots_dir: str, image_path: str):
    image = cv2.imread(image_path, 0)

    if find_pupil(image) is not None:
        ellipsis, center = find_pupil(image)
        image = draw_ellipsis(image, ellipsis)
        msg = f"pupil center : {center}"
    else:
        msg = "no pupil detected"

    bottom_left_coord = (10, image.shape[0] - 10)
    image = cv2.putText(
        image,
        msg,
        bottom_left_coord,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    plt.imsave(f"{plots_dir}/{Path(image_path).stem}.png", image)


def sumatorias(image, div=10, debug=0):
    image = ~image
    ejeX = []
    ejeY = []
    x=0
    y=0
    while(x < len(image)):
        suma = 0
        while(y<len(image[0])):
            valor = image[x,y]
            if(valor > 200):
                suma = suma + valor
            elif(valor < -1):
                suma = suma + 255-valor
            y = y + div
        ejeX.append(suma)
        x = x + div
        y = 0

    while(y < len(image[0])):
        suma = 0
        while(x<len(image)):
            valor = image[x,y]
            if(valor > 200):
                suma = suma + valor
            elif(valor < -1):
                suma = suma + 255-valor
            x = x + div
        ejeY.append(suma)
        y = y + div
        x = 0
    i = 0
    j = 0
    if(debug == 1):
        plt.figure()
        plt.imshow(image,cmap='gray')
        plt.figure()
        plt.plot(ejeX)
        plt.plot(ejeY)
    coordX = ejeY.index(max(ejeY))*div
    coordY = ejeX.index(max(ejeX))*div
    return (coordX, coordY)


def ExCuSe(img: str):
    pupil = find_pupil(img)
    if pupil is not None:
        return pupil[1]
    else:
        return sumatorias(img, 10)