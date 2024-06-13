import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

# Step 1: Load and display the original image
image_raw = imread("image.png")
print("Original image shape:", image_raw.shape)  # Dimensions of the image (height, width, channels)

# Displaying the original image
plt.figure(figsize=[12, 8])
plt.imshow(image_raw)
plt.title("Original Image")
plt.show()

# Create and print the vector with image dimensions and number of color channels
image_dimensions = np.array([image_raw.shape[0], image_raw.shape[1], image_raw.shape[2]])
print("Image dimensions and number of color channels:", image_dimensions)

# Step 2: Convert to grayscale and display the grayscale image
image_sum = image_raw.sum(axis=2)  # Sum across the color channels
image_bw = image_sum / image_sum.max()  # Normalize to the range [0, 1]
print("Grayscale image shape:", image_bw.shape)  # Dimensions of the grayscale image

# Displaying the grayscale image
plt.figure(figsize=[12, 8])
plt.imshow(image_bw, cmap=plt.cm.gray)
plt.title("Grayscale Image")
plt.show()

# Step 3: Apply PCA to the grayscale image
pca = PCA()
pca.fit(image_bw)

# Cumulative variance
var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

# Number of components to explain 95% variance
k = np.argmax(var_cumu >= 95) + 1  # Adding 1 as np.argmax returns the first index where condition is met
print("Number of components explaining 95% variance:", k)

# Plot cumulative explained variance
plt.figure(figsize=[10, 5])
plt.plot(var_cumu)
plt.axhline(y=95, color='r', linestyle='--')
plt.axvline(x=k, color='k', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.show()

# Step 4: Reconstruct the grayscale image using the identified number of components
ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

# Displaying the reconstructed image
plt.figure(figsize=[12, 8])
plt.imshow(image_recon, cmap=plt.cm.gray)
plt.title(f"Reconstructed Image with {k} Components (95% Variance)")
plt.show()

# Step 5: Reconstruct and display the grayscale image using different numbers of components
def plot_at_k(k):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
    plt.imshow(image_recon, cmap=plt.cm.gray)
    plt.title(f"Components: {k}")

ks = [10, 25, 50, 100, 150, 250]

plt.figure(figsize=[15, 9])

for i in range(len(ks)):
    plt.subplot(2, 3, i + 1)
    plot_at_k(ks[i])

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()