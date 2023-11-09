# COMP257 - Unsupervised & Reinforcement Learning (Section 002)
# Assignment 4 - Gaussian Mixture Models
# Name: Wai Lim Leung
# ID  : 301276989
# Date: 08-Nov-2023

from sklearn.datasets import fetch_olivetti_faces
from scipy.ndimage import rotate
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Part 1 - Retrieve & Load Olivetti Faces
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
of_data = olivetti_faces.data
of_target = olivetti_faces.target
of_images = olivetti_faces.images
of_labels = olivetti_faces.target
print()
print("Part 1 - Olivetti Faces")
print("Data Shape  :", of_data.shape)
print("Target Shape:", of_target.shape)

fig = plt.figure(figsize=(6, 2))
for i in range(3):
    face_image = of_data[i].reshape(64, 64)
    position = fig.add_subplot(1, 3, i + 1)
    position.imshow(face_image, cmap='gray')
    position.set_title(f"Person {of_target[i]}")
    position.axis('off')
plt.tight_layout()
plt.show()

# Part 2 - PCA Preserving 99% Variance
pca = PCA(n_components=0.99, whiten=True, random_state=42)
faces_data_pca = pca.fit_transform(of_data)
print()
print("Part 2 - PCA Preserving 99% Variance")
print(faces_data_pca)

# Part 3 - GMM Reduce Dateset Dimension
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
gmm.fit(faces_data_pca)
print()
print("Part 3 - GMM Reduce Dateset Dimension")
print("Weights: ", gmm.weights_)
print("Means  : ", gmm.means_)

# Part 4 - Most suitable Covariance Type dataset
print()
print("Part 4 - Most suitable Covariance Type dataset")
print(gmm.covariances_)

# Part 5 - Determine Min Number of Clusters
n_clusters = np.arange(1, 21)
aics = []
bics = []

# Fit GMM into different clusters
for n in n_clusters:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(faces_data_pca)
    aics.append(gmm.aic(faces_data_pca))
    bics.append(gmm.bic(faces_data_pca))

min_aic = np.min(aics)
min_bic = np.min(bics)
best_n_clusters_aic = n_clusters[np.argmin(aics)]
best_n_clusters_bic = n_clusters[np.argmin(bics)]

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(n_clusters, aics, label='AIC')
plt.xlabel('No of Clusters')
plt.ylabel('AIC')
plt.title('AIC / No of Clusters')
plt.axvline(x=best_n_clusters_aic, color='r', linestyle='--')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_clusters, bics, label='BIC')
plt.xlabel('No of Clusters')
plt.ylabel('BIC')
plt.title('BIC / No of Clusters')
plt.axvline(x=best_n_clusters_bic, color='r', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

print()
print("Part 5 - Determine Min Number of Clusters")
print("AIC: ", best_n_clusters_aic)
print("BIC: ", best_n_clusters_bic)

# Part 6 - Plot Graph from Part 2 to 4
print()
print("Part 6 - Plot Graph from Part 2 to 4")

# Part 6 - Plot the Results from Part 2 to 4
pca_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.bar(range(len(pca_variance)), pca_variance, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Principal Component & Ratio')
plt.show()

pca_for_plot = PCA(n_components=2)
faces_data_pca_2 = pca_for_plot.fit_transform(of_data)
gmm_for_plot = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
gmm_for_plot.fit(faces_data_pca_2)
gmm_labels = gmm_for_plot.predict(faces_data_pca_2)
plt.figure(figsize=(8, 4))
plt.scatter(faces_data_pca_2[:, 0], faces_data_pca_2[:, 1], c=gmm_labels, cmap='viridis', marker='.')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM in PCA Reduced Space')
plt.colorbar(label='Cluster Label')
plt.show()

silhouette_avg = silhouette_score(faces_data_pca, gmm.predict(faces_data_pca))
print("Silhouette Score Average:", silhouette_avg)

# Part 7 - Hard Clustering for Each Instance
hard_clustering = gmm.predict(faces_data_pca)
print()
print("Part 7 - Hard Clustering for Each Instance")
print(hard_clustering)

# Part 8 - Output the Soft Clustering for Each Instance
soft_clustering = gmm.predict_proba(faces_data_pca)
print()
print("Part 8 - Hard Clustering for Each Instance")
print(soft_clustering)

# Part 9 - Generate New Faces & Visualize
print()
print("Part 9 - Generate New Faces & Visualize")
n_samples = 10
new_faces_pca, _ = gmm.sample(n_samples)
new_faces_original = pca.inverse_transform(new_faces_pca)

fig, axes = plt.subplots(1, n_samples, figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(new_faces_original[i].reshape(olivetti_faces.images[0].shape), cmap='gray')
    ax.set_title(f"Sample Face {i+1}")
plt.tight_layout()
plt.show()

# Part 10 - Rotate, Flip & Darken
def rotate_image(image, angle):
    return rotate(image, angle, reshape=False)

def flip_image(image):
    return image[:, ::-1]

def darken_image(image, factor):
    return image * factor

example_face = new_faces_original[0].reshape(64, 64)
rotated_face = rotate_image(example_face, 90)
flipped_face = flip_image(rotated_face)
darkened_face = darken_image(example_face, 50)

fig, axes = plt.subplots(1, 4, figsize=(12, 3), subplot_kw={'xticks':[], 'yticks':[]})
axes[0].imshow(example_face, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(rotated_face, cmap='gray')
axes[1].set_title('Rotated')
axes[2].imshow(flipped_face, cmap='gray')
axes[2].set_title('Flipped')
axes[3].imshow(darkened_face, cmap='gray')
axes[3].set_title('Darkened')
plt.tight_layout()
plt.show()

print()
print("Part 10 - Rotate, Flip & Darken")

# Part 11 - Score Samples Method
def modify_images(images):
    modified_images = []
    for image in images:
        rotated = rotate(image, 90, reshape=False)
        flipped = np.fliplr(rotated)
        darkened = flipped * 50
        modified_images.append(darkened.ravel())
    return np.array(modified_images)

modified_new_faces = modify_images(new_faces_original.reshape(-1, 64, 64))
modified_new_faces_pca = pca.transform(modified_new_faces)
original_scores = gmm.score_samples(faces_data_pca)
generated_faces_scores = gmm.score_samples(new_faces_pca)
modified_faces_scores = gmm.score_samples(modified_new_faces_pca)

print()
print("Part 11 - Score Samples Method")
print("Average Score for Original Images: ", np.mean(original_scores))
print("Average Score for Generated Faces: ", np.mean(generated_faces_scores))
print("Average Score for Modified Faces : ", np.mean(modified_faces_scores))

threshold = np.mean(original_scores) - 3 * np.std(original_scores)
print()
print("Threshold for Anomaly Detection  : ", threshold)
print("Anomalies in Generated Faces     : ", np.sum(generated_faces_scores < threshold))
print("Anomalies in Modified Faces      : ", np.sum(modified_faces_scores < threshold))

fig, axes = plt.subplots(1, n_samples, figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(modified_new_faces[i].reshape(64, 64), cmap='gray')
    ax.set_title(f"Modified {i+1}")
plt.tight_layout()
plt.show()