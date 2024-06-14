import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

# крок 1: вивести початкове кольорове зображення та вектор
image_raw = imread("image.png")
print("Original image shape:", image_raw.shape)  # розміри зображення в пікселях та кількість основних каналів кольорів (height, width, channels)

plt.figure(figsize=[12, 8])
plt.imshow(image_raw)
plt.title("Original Image")
plt.show()

# Create and print the vector with image dimensions and number of color channels
image_dimensions = np.array([image_raw.shape[0], image_raw.shape[1], image_raw.shape[2]])
print("Image dimensions and number of color channels:", image_dimensions)

# крок 2: Перетворити зображення в чорно-біле та вивести розмір зображення і кількість каналів кольорів
image_sum = image_raw.sum(axis=2)  # Sum across the color channels
print(image_sum.shape)
image_bw = image_sum / image_sum.max()  # Normalize to the range [0, 1]
print("Grayscale image shape:", image_bw.shape)  # Dimensions of the grayscale image

plt.figure(figsize=[12, 8])
plt.imshow(image_bw, cmap=plt.cm.gray)
plt.title("Grayscale Image")
plt.show()

# крок 3: Застосувати PCA для матриці компонентів image_bw
pca = PCA()
pca.fit(image_bw)

# cumulative variance
var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100
k = np.argmax(var_cumu >= 95) + 1
print("Number of components explaining 95% variance:", k)

plt.figure(figsize=[10, 5])
plt.plot(var_cumu)
plt.axhline(y=95, color='r', linestyle='--')
plt.axvline(x=k, color='k', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.show()

# крок 4: Провести реконструкцію чорно-білого зображення, використовуючи обмежену кількість компонентів, знайдену в попередньому кроці
ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

# Вивести отримане зображення
plt.figure(figsize=[12, 8])
plt.imshow(image_recon, cmap=plt.cm.gray)
plt.title(f"Reconstructed Image with {k} Components (95% Variance)")
plt.show()


# крок 5: Проведіть реконструкцію зображення для різної кількості компонент та виведіть відповідні результати


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