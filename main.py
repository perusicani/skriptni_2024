import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Creating own scale
def define_color_temperature_scale():
    """
    Define a general mapping from colors to temperatures.
    The mapping is based on a typical thermal color gradient.
    """
    # Define the color to temperature mapping based on the typical thermal gradient
    color_temperature_pairs = [
        ([255, 255, 255], 43),  # White: hottest
        ([255, 0, 0], 41),      # Red
        ([255, 165, 0], 38),    # Orange
        ([255, 255, 0], 36),    # Yellow
        ([0, 255, 0], 33),      # Green
        ([0, 0, 255], 31),      # Blue
        ([128, 0, 128], 29)     # Purple: coldest
    ]
    
    # Convert to numpy arrays for further processing
    colors = np.array([color for color, _ in color_temperature_pairs])
    temperatures = np.array([temp for _, temp in color_temperature_pairs])
    
    return colors, temperatures

def map_image_to_temperature(image_rgb, colors, temperatures):
    """
    Map each pixel in the image to a temperature based on the closest color in the thermal scale.
    """
    # Flatten the image into an array of pixels
    pixel_values = image_rgb.reshape((-1, 3))
    
    # Create a KDTree for fast lookup of the nearest color
    tree = KDTree(colors)
    
    # Map each pixel to the closest color and retrieve the corresponding temperature
    _, indices = tree.query(pixel_values)
    mapped_temperatures = temperatures[indices].reshape(image_rgb.shape[:2])
    
    return mapped_temperatures


def find_contours_and_label_own_scale(image_rgb, mapped_temperatures):
    """
    Find contours and label them based on the temperature mapping.
    """
    # Convert the mapped temperatures to grayscale for boundary detection
    gray_mapped = cv2.normalize(mapped_temperatures, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Use the Sobel operator to find the gradients
    grad_x = cv2.Sobel(gray_mapped, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_mapped, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize and threshold the magnitude to get binary edges
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours and label them with the corresponding temperatures
    contoured_image = image_rgb.copy()
    for contour in contours:
        cv2.drawContours(contoured_image, [contour], -1, (0, 0, 0), 1)  # Black contours with thickness 1
        
        # Compute the mean temperature within the contour
        mask = np.zeros_like(gray_mapped)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_temperature = cv2.mean(mapped_temperatures, mask=mask)[0]
        
        # Find the center of the contour to place the text
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Add the temperature text to the contour
            cv2.putText(contoured_image, f'{int(mean_temperature)}C', (cX - 20, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return contoured_image

# Creating own scale

def map_temperature_to_color(scale_image, temperature_values):
    """
    Map temperatures to colors on the gradient scale.
    :param scale_image: The gradient scale image.
    :param temperature_values: List of temperatures corresponding to points on the gradient.
    :return: A list of mapped colors corresponding to each temperature.
    """
    height, width, _ = scale_image.shape
    row = height // 2  # Sample from the middle row
    num_temperatures = len(temperature_values)

    # Map each temperature to a color sampled from the gradient
    sampled_colors = []
    positions = []
    for i in range(num_temperatures):
        x = int(i * ((width - 1) / (num_temperatures - 1)))  # Ensure x is within bounds
        sampled_colors.append(scale_image[row, x])
        positions.append(x)  # Store the position for plotting later
    
    return np.array(sampled_colors), positions

def map_image_to_sampled_colors(image_rgb, sampled_colors):
    # Flatten the image into an array of pixels
    pixel_values = image_rgb.reshape((-1, 3))
    
    # Create a KDTree for fast lookup of the nearest sampled color
    tree = KDTree(sampled_colors)
    
    # Map each pixel to the closest color in the sampled gradient
    _, indices = tree.query(pixel_values)
    mapped_image = sampled_colors[indices].reshape(image_rgb.shape)
    
    return mapped_image

def find_contours_and_label(image_rgb, mapped_image, temperature_values, sampled_colors, positions):
    # Convert the mapped image to grayscale for boundary detection
    gray_mapped = cv2.cvtColor(mapped_image, cv2.COLOR_RGB2GRAY)

    # Use the Sobel operator to find the gradients
    grad_x = cv2.Sobel(gray_mapped, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_mapped, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize and threshold the magnitude to get binary edges
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours and label them with the corresponding temperatures
    contoured_image = image_rgb.copy()
    for contour in contours:
        cv2.drawContours(contoured_image, [contour], -1, (0, 0, 0), 1)  # Black contours with thickness 1
        
        # Compute the mean color within the contour
        mask = np.zeros_like(gray_mapped)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(image_rgb, mask=mask)[:3]
        
        # Find the closest temperature by comparing mean color to sampled colors
        tree = KDTree(sampled_colors)
        _, idx = tree.query(mean_color)
        temperature = temperature_values[idx]
        
        # Find the center of the contour to place the text
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Add the temperature text to the contour
            cv2.putText(contoured_image, f'{temperature}C', (cX - 20, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return contoured_image

image_path = '' # Placeholders that need to be filed with test functions
scale_image_path = '' # Placeholders that need to be filed with test functions
temperature_values = [] # Placeholders that need to be filed with test functions

def runTest():
    image_path = 'test.png'
    scale_image_path = 'scale.png'
    # Define the temperature values corresponding to the gradient scale
    temperature_values = [43, 41, 38, 36, 33, 31, 29, 25]
    return image_path, scale_image_path, temperature_values

def runTest2():
    image_path = 'test2.png'
    scale_image_path = 'scale.png'
    # Define the temperature values corresponding to the gradient scale
    temperature_values = [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,  31, 30, 29, 28]
    return image_path, scale_image_path, temperature_values

def runBalticsTest():
    image_path = 'baltic_test.png'
    scale_image_path = 'baltic_test_scale.png'
    temperature_values = [20, 17.5, 15, 12.5, 10, 7.5, 5]
    return image_path, scale_image_path, temperature_values

def runAfricaTest():
    image_path = 'africa_test.png'
    scale_image_path = 'africa_scale.png'
    temperature_values = [17.5, 17, 16, 15, 14, 13, 12, 11, 10, 9.5]
    return image_path, scale_image_path, temperature_values

image_path, scale_image_path, temperature_values = runAfricaTest()

print(image_path)
print(scale_image_path)
print(temperature_values)

# Assert variables have been set
assert image_path != ""
assert scale_image_path != ""
assert temperature_values != []

# Load the main image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the gradient scale image
scale_image = cv2.imread(scale_image_path)
scale_image_rgb = cv2.cvtColor(scale_image, cv2.COLOR_BGR2RGB)


# ------------------ Scale included
# # Map the temperature values to their respective colors on the gradient
# sampled_colors, positions = map_temperature_to_color(scale_image_rgb, temperature_values)

# # Map the image colors to the closest color in the sampled gradient
# mapped_image = map_image_to_sampled_colors(image_rgb, sampled_colors)

# # Find contours and label them with temperatures
# contoured_and_labeled_image = find_contours_and_label(image_rgb, mapped_image, temperature_values, sampled_colors, positions)

# # Plot the results
# plt.figure(figsize=(10, 20))

# # Display the gradient scale with temperature labels
# plt.subplot(4, 1, 1)
# plt.imshow(cv2.cvtColor(scale_image, cv2.COLOR_BGR2RGB))
# plt.title('Gradient Scale')
# for temp, pos in zip(temperature_values, positions):
#     plt.axvline(x=pos, color='red', linestyle='--', linewidth=1)  # Vertical lines
#     plt.text(pos, scale_image.shape[0] // 2 - 10, f'{temp}C', color='black', fontsize=8, ha='center')

# # Display the original image
# plt.subplot(4, 1, 2)
# plt.imshow(image_rgb)
# plt.title('Original Image')

# # Display the mapped image
# plt.subplot(4, 1, 3)
# plt.imshow(mapped_image)
# plt.title('Mapped Image')

# # Display the contoured and labeled image
# plt.subplot(4, 1, 4)
# plt.imshow(contoured_and_labeled_image)
# plt.title('Contoured & Labeled Image')

# plt.tight_layout()
# plt.show()
# ------------------ Scale included

# ------------------- Own scale
# Define the color-temperature scale
colors, temperatures = define_color_temperature_scale()

# Map the image colors to temperature values
mapped_temperatures = map_image_to_temperature(image_rgb, colors, temperatures)

# Find contours and label them with temperatures
contoured_and_labeled_image = find_contours_and_label_own_scale(image_rgb, mapped_temperatures)

# Plot the results
plt.figure(figsize=(10, 15))

# Display the original image
plt.subplot(3, 1, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

# Display the mapped temperatures (as a grayscale image)
plt.subplot(3, 1, 2)
plt.imshow(mapped_temperatures, cmap='hot')
plt.title('Mapped Temperature Image')

# Display the contoured and labeled image
plt.subplot(3, 1, 3)
plt.imshow(contoured_and_labeled_image)
plt.title('Contoured & Labeled Image')

plt.tight_layout()
plt.show()
# -------------------- Own scale

