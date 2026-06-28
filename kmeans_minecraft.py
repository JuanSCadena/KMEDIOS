import os, sys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IMG_DIR = '/Users/sebascadena/KMEANS/archive/minecraft_block_textures/images'
OUT_DIR = '/Users/sebascadena/KMEANS/output'
os.makedirs(OUT_DIR, exist_ok=True)

def load_images(img_dir, target_size=(16, 16), max_imgs=None):
    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if max_imgs:
        files = files[:max_imgs]
    images, names = [], []
    for f in files:
        fp = os.path.join(img_dir, f)
        img = Image.open(fp).convert('RGB')
        img = img.resize(target_size, Image.NEAREST)
        images.append(np.array(img, dtype=np.float32))
        names.append(f)
    return np.array(images), names

def images_to_features(images):
    n, h, w, c = images.shape
    return images.reshape(n, h * w * c)

def plot_centroids(centroids, h, w, c, k, out_path):
    n = len(centroids)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i in range(n):
        centroid = centroids[i].reshape(h, w, c)
        centroid = np.clip(centroid / 255.0, 0, 1)
        axes[i].imshow(centroid)
        axes[i].set_title(f'Cluster {i}', fontsize=9)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'K-Means Centroids (k={k})', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_centroids_compact(km, images, labels, k, h, w, c, out_path, names):
    counts = np.bincount(labels.astype(int), minlength=k)
    n_samples = min(3, int(counts.min()))
    if n_samples < 1:
        n_samples = 1

    grid_cols = int(np.ceil(np.sqrt(k)))
    grid_rows = int(np.ceil(k / grid_cols))

    cell_w = 3 + n_samples * 1.2
    cell_h = 2.5
    fig, axes = plt.subplots(grid_rows, grid_cols,
                             figsize=(grid_cols * cell_w, grid_rows * cell_h))
    if grid_rows * grid_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(k):
        ax = axes[i]
        idxs = np.where(labels == i)[0]
        centroid = km.cluster_centers_[i].reshape(h, w, c)
        centroid = np.clip(centroid / 255.0, 0, 1)

        inset_gs = ax.inset_axes([0.02, 0.35, 0.3, 0.6])
        inset_gs.imshow(centroid, interpolation='nearest')
        inset_gs.axis('off')
        inset_gs.set_title(f'C{i}', fontsize=7, pad=1)

        for j in range(n_samples):
            if j < len(idxs):
                idx = int(idxs[j])
                img = images[idx].reshape(h, w, c).astype(np.uint8)
                lx = 0.35 + j * 0.22
                ins = ax.inset_axes([lx, 0.35, 0.2, 0.6])
                ins.imshow(img, interpolation='nearest')
                ins.axis('off')

        ax.text(0.5, 0.12, f'Cluster {i}  n={len(idxs)}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, fontweight='bold')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(k, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Clusters y Muestras (k={k})', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_grid(images, labels, k, h, w, c, out_path, names):
    counts = np.bincount(labels.astype(int), minlength=k)
    n_samples = min(5, int(counts.min()))
    if n_samples < 1:
        n_samples = 1
    cols = 1 + n_samples
    fig, axes = plt.subplots(k, cols, figsize=(cols * 2, k * 2))
    if k == 1:
        axes = axes.reshape(1, -1)
    for i in range(k):
        idxs = np.where(labels == i)[0]
        axes[i, 0].text(0.5, 0.5, f'Cluster {i}\n(n={len(idxs)})',
                        transform=axes[i, 0].transAxes,
                        ha='center', va='center', fontsize=8)
        axes[i, 0].axis('off')
        for j in range(n_samples):
            ax = axes[i, j + 1]
            if j < len(idxs):
                idx = int(idxs[j])
                img = images[idx].reshape(h, w, c).astype(np.uint8)
                ax.imshow(img)
                nm = names[idx].replace('.png', '')
                ax.set_title(nm, fontsize=5)
            ax.axis('off')
    plt.suptitle(f'Cluster Samples (k={k})', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_pca(images, labels, k, names, out_path):
    n = len(images)
    feat = images.reshape(n, -1)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feat)
    var_expl = pca.explained_variance_ratio_
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, k))
    for i in range(k):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=f'Cluster {i} (n={mask.sum()})',
                   alpha=0.6, s=10, edgecolors='none')
    ax.set_xlabel(f'PC1 ({var_expl[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({var_expl[1]:.1%} var)')
    ax.set_title('PCA Projection of Minecraft Textures')
    ax.legend(markerscale=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return pca, coords

def elbow_method(features, k_range, out_path):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(features)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, 'bo-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return list(k_range), inertias

def main():
    print("Loading images...")
    images, names = load_images(IMG_DIR)
    print(f"Loaded {len(images)} images with shape {images[0].shape}")
    n, h, w, c = images.shape
    features = images_to_features(images)
    print(f"Feature matrix shape: {features.shape}")

    print("Computing elbow curve...")
    k_range = range(2, 21)
    ks, inertias = elbow_method(features, k_range,
                                 os.path.join(OUT_DIR, 'elbow.png'))

    k_values = [8, 16, 24]
    results = {}
    for k in k_values:
        print(f"Running K-Means with k={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(features)
        results[k] = {'km': km, 'labels': labels}

        plot_centroids(km.cluster_centers_, h, w, c, k,
                       os.path.join(OUT_DIR, f'centroids_k{k}.png'))
        plot_centroids_compact(km, images, labels, k, h, w, c,
                               os.path.join(OUT_DIR, f'samples_k{k}.png'), names)

    print("Generating PCA visualization for k=16...")
    pca, coords = plot_pca(images, results[16]['labels'], 16, names,
                           os.path.join(OUT_DIR, 'pca_clusters.png'))

    print(f"All outputs saved to {OUT_DIR}/")
    print("K-means clustering complete!")

if __name__ == '__main__':
    main()
