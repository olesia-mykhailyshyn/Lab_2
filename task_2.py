import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

# крок 1: вивести початкове кольорове зображення та вектор
print("Task 1:")
image_raw = imread("image.png")
print("Original image shape (height, width, channels):", image_raw.shape)  # розміри зображення в пікселях та кількість основних каналів кольорів (height, width, channels)
#channels = 4 -- red,green,blue, alpha(прозорість)

plt.imshow(image_raw)
plt.title("Original Image")
plt.show()
print()

# крок 2: Перетворити зображення в чорно-біле та вивести розмір зображення і кількість каналів кольорів
print("Task 2:")
image_sum = image_raw.sum(axis=2) #сума по каналах кольорів
# кожен піксель у вихідному зображенні представляється одним числовим значенням, яке є сумою значень червоного, зеленого і синього каналів для цього пікселя
print(image_sum.shape)
image_bw = image_sum / image_sum.max() #нормалізація для отримання значень від 0 до 1 -- інтенсивність кожного пікселя
print(image_bw.max())
print("Grayscale image shape (height, width, channels = 1 (because it is grayscale image) ):", image_bw.shape)

plt.imshow(image_bw, cmap=plt.cm.gray) #plt.cm.gray це колірна карта, яка перетворює значення інтенсивності пікселів у відповідні відтінки сірого кольору на зображенні
plt.title("Grayscale Image")
plt.show()
print()

# крок 3: Застосувати PCA для матриці компонентів image_bw
print("Task 3:")
pca = PCA()
pca.fit(image_bw)
# за допомогою fit() PCA обчислює головні компоненти на основі введеної матриці image_bw
# PCA шукає напрямки (головні компоненти), які пояснюють найбільшу варіацію в даних

# дисперсія
var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu >= 95) + 1
print("Number of components explaining 95% variance:", k)

plt.plot(var_cumu)
plt.axhline(y=95, color='r', linestyle='--')
plt.axvline(x=k, color='k', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.show()
print()

# крок 4: Провести реконструкцію чорно-білого зображення, використовуючи обмежену кількість компонентів, знайдену в попередньому кроці
print("Task 3:")
ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
#fit_transform для зменшення розмірності вхідних даних і inverse_transform для отримання відтвореного зображення у вихідному просторі ознак

# Вивести отримане зображення
plt.imshow(image_recon, cmap=plt.cm.gray)
plt.title(f"Reconstructed Image with {k} Components (95% Variance)")
plt.show()
print()


# крок 5: Проведіть реконструкцію зображення для різної кількості компонент та виведіть відповідні результати


def plot_at_k(k):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
    plt.imshow(image_recon, cmap=plt.cm.gray)
    plt.title(f"Components: {k}")


components = [10, 25, 50, 100, 150, 250]

for i in range(len(components)):
    plt.subplot(2, 3, i + 1)
    plot_at_k(components[i])

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()