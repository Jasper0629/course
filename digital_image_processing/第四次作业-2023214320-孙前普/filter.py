import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
input_image = cv2.imread('Sample.png', cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform of the image
f_transform = np.fft.fft2(input_image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Define the cutoff frequencies
D0_values = [10, 20, 40]

# Initialize a figure to display the results
plt.figure(figsize=(15, 5))

for i, D0 in enumerate(D0_values):
    # Ideal Lowpass Filter
    H_ideal = np.zeros_like(input_image, dtype=np.float32)
    center_x, center_y = input_image.shape[1] // 2, input_image.shape[0] // 2
    for y in range(input_image.shape[0]):
        for x in range(input_image.shape[1]):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance <= D0:
                H_ideal[y, x] = 1

    # Apply the ideal lowpass filter to the frequency domain image
    ideal_filtered = f_transform_shifted * H_ideal
    ideal_filtered = np.fft.ifftshift(ideal_filtered)
    ideal_filtered_image = np.fft.ifft2(ideal_filtered)
    ideal_filtered_image = np.abs(ideal_filtered_image).astype(np.uint8)

    rows, cols = input_image.shape
    center_x, center_y = cols // 2, rows // 2
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    H_gaussian = np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * D0 ** 2))

    # Apply the Gaussian lowpass filter to the frequency domain image
    gaussian_filtered = f_transform_shifted * H_gaussian
    gaussian_filtered = np.fft.ifftshift(gaussian_filtered)
    gaussian_filtered_image = np.fft.ifft2(gaussian_filtered)
    gaussian_filtered_image = np.abs(gaussian_filtered_image).astype(np.uint8)


    # Display the original and filtered images
    plt.subplot(3, 5, i * 5 + 1)
    plt.title(f'Original Image (D0={D0})')
    plt.imshow(input_image, cmap='gray')

    plt.subplot(3, 5, i * 5 + 2)
    plt.title(f'Ideal Filtered Image (D0={D0})')
    plt.imshow(ideal_filtered_image, cmap='gray')

    plt.subplot(3, 5, i * 5 + 3)
    plt.title(f'Gaussian Filtered Image (D0={D0})')
    plt.imshow(gaussian_filtered_image, cmap='gray')

    # Analyze the ringing effect by plotting the difference between the original and filtered images
    ringing_ideal = input_image - ideal_filtered_image
    ringing_gaussian = input_image - gaussian_filtered_image

    plt.subplot(3, 5, i * 5 + 4)
    plt.title(f'Ringing_idead (D0={D0})')
    plt.imshow(ringing_ideal, cmap='gray')
    plt.colorbar()

    plt.subplot(3, 5, i * 5 + 5)
    plt.title(f'Ringing_gaussian (D0={D0})')
    plt.imshow(ringing_gaussian, cmap='gray')
    plt.colorbar()

plt.tight_layout()
plt.savefig("output/ideal_gaussian.png")
