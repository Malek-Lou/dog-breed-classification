#### Import necessary libraries
# General libraries
import os
import random # Random number generators
import xml.etree.ElementTree as ET # For parsing XML like documents
# Numerical libraries
import numpy as np
import pickle   
# Visualisation
import matplotlib.pyplot as plt
import pathlib 
# Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hamming_loss
from sklearn.neighbors import KNeighborsClassifier

# Image processing
import skimage.io as io # image I/O routines
from skimage.transform import rescale, resize
from skimage.filters import sobel, sobel_h, sobel_v

PATH_TO_DB=pathlib.Path(os.path.join(os.getcwd(),'..','SmallDB'))
IMG_DB = 'Images'
ANNOT_DB = 'Annotation'

IMG_FULL = PATH_TO_DB / IMG_DB  # Full path to image folder

TARGET_SIZE = (64, 64)

FIGS = 'figures'
def save_dict(fname, dict):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)


def load_dict(fname):
    with open(fname, 'rb') as f:
        out = pickle.load(f)
    return out



def read_img(n_value, dog_breed, fname, TO_DB=IMG_FULL, color=True):
    """Reads a single image using skimage.io."""
    folder_name = f"{n_value}-{dog_breed}"
    img_path = TO_DB / folder_name / fname
    return io.imread(img_path, as_gray=not color)


def read_db(subset_dogs=None, TO_DB=IMG_FULL, color=True):
    """Loads images, numeric labels, and human-readable label mapping."""
    label_names = dict()
    next_label = 0
    labels = []
    images = []

    if not os.path.exists(TO_DB):
        return None, None, None

    for subdir in sorted(os.listdir(TO_DB)):
        if '-' not in subdir:
            continue
        n_value, dog_breed = subdir.split('-', 1)

        if subset_dogs and dog_breed not in subset_dogs:
            continue

        label_names[next_label] = dog_breed
        nb_img = 0

        for img in os.listdir(os.path.join(TO_DB, subdir)):
            images.append(read_img(n_value, dog_breed, img, TO_DB=TO_DB, color=color))
            nb_img += 1

        labels.extend([next_label] * nb_img)
        next_label += 1

    return images, labels, label_names



def entropy(p):
    """
    Computes Shannon Entropy of a discrete probability distribution.
    Formula: H(P) = -sum(p_i * log_e(p_i))
    """
    
    p = np.array(p, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    entropy_value = -np.sum(p * np.log(p))
    
    return entropy_value

def entropy_breeds(labels):
    """Computes distribution entropy for evaluation of class balance."""
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts)

def plot_barplot(labels, label_names, fig_path = None):
    """Generates class balance bar plots."""
    l, v = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(l, v, color='#2e75b6', edgecolor='black', alpha=0.8)
    ax.set_xticks(l)
    ax.set_xticklabels([label_names[_l] for _l in l], rotation=15, ha='right')
    ax.set_ylabel("Sample Count")
    ax.set_title("SmallDB Class Balance Evaluation", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    fig.tight_layout()
    if fig_path:
        pass 
    plt.close()
    return fig
    



def plot_images(imgs, labels, label_names, nb_rows=3, nb_cols = 4, fig_path=None, rdm_indices=None):
    if not rdm_indices:
        # Random selection of indices from the database
        rdm_indices = random.sample(range(0, len(imgs)),nb_rows*nb_cols)
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=(10, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.4,hspace=0.4)
    for i in range(nb_rows):
        for j in range(nb_cols):
            ax[i][j].imshow(imgs[rdm_indices[i*nb_cols+j]], cmap='gray')
            ax[i][j].set_title(label_names[labels[rdm_indices[i*nb_cols+j]]])

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    
    plt.close()

    return rdm_indices, fig

def extract_bounding_box(fname): 
    """Parses Pascal VOC XML files to retrieve object bounding coordinates."""
    tree = ET.parse(fname)
    root = tree.getroot()
    
    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return xmin, xmax, ymin, ymax

def read_and_crop(n_value, dog_breed, fname, TO_DB=PATH_TO_DB, color=True): 
    """Reads source image alongside its XML bounding box data to crop target."""
    img = io.imread(os.path.join(TO_DB, IMG_DB, f"{n_value}-{dog_breed}", fname), as_gray=not color)
    base_name = os.path.splitext(fname)[0]
    bndbox_name = os.path.join(TO_DB, ANNOT_DB, f"{n_value}-{dog_breed}", base_name)
    if not os.path.exists(bndbox_name):
        bndbox_name += '.xml'
    if not os.path.exists(bndbox_name):
        raise FileNotFoundError(f"Annotation file not found: {bndbox_name}")
    
    xmin, xmax, ymin, ymax = extract_bounding_box(bndbox_name)
    dog = img[ymin:ymax+1, xmin:xmax+1]
    
    
    return img, dog

def read_and_crop_db(subset_dogs=None, TO_DB=PATH_TO_DB, color=True): 
    """Processes entire dataset, returning original and isolated target sub-arrays."""
    label_names = dict()
    next_label = 0
    labels = []
    images = []
    dogs = []

    img_root = os.path.join(TO_DB, IMG_DB)
    if not os.path.exists(img_root):
        return None, None, None, None

    for subdir in sorted(os.listdir(img_root)):
        if '-' not in subdir:
            continue
        n_value, dog_breed = subdir.split('-', 1)
        
        if subset_dogs and dog_breed not in subset_dogs:
            continue
            
        label_names[next_label] = dog_breed
        nb_img = 0
        
        for img_file in os.listdir(os.path.join(img_root, subdir)):
            try:
                img, dog = read_and_crop(n_value, dog_breed, img_file, TO_DB=TO_DB, color=color)
                images.append(img)
                dogs.append(dog)
                nb_img += 1
            except (FileNotFoundError, Exception) as e:
                print(f"  Skipping {img_file}: {e}")
                continue
            
        labels.extend([next_label] * nb_img)
        next_label += 1

    return images, dogs, labels, label_names




def img_rescale(img, ratio): 
    if len(img.shape) == 3:
        return rescale(img, ratio, anti_aliasing=True, channel_axis=2)
    return rescale(img, ratio, anti_aliasing=True)

def img_resize(img, target_size): 
    return resize(img, target_size, anti_aliasing=True)

def resize_and_pad(img, target_size=TARGET_SIZE, pad_type='white'):
    """
    ========================================================================
    TASK 2: ASPECT-PRESERVING SPATIAL SCALING
    ========================================================================
    Resizes an image preserving its original aspect ratio, centering it 
    on a fixed bounding canvas. At least three padding should be implemented
    * pad_type: Type of padding to use ('white' or 'black' or 'continuous').
        - 'white': Pads with maximum intensity (1.0 for normalized images).
        - 'black': Pads with minimum intensity (0.0 for normalized images).
        - 'continuous': Pads with the closest pixel value in the original image.
    Other potential padding strategies (e.g., reflection, edge replication, mean pixel values...) can be implemented as extensions.
    """
    h, w = img.shape[:2]
    multi_channel = len(img.shape) == 3
    height, width = target_size
    
    ### STUDENT IMPLEMENTATION START ###
    
    
    h_r = height / h
    w_r = width / w

    
    ratio = min(h_r, w_r)

   
    new_h = int(round(h * ratio))
    new_w = int(round(w * ratio))

  
    bottom = (height - new_h) // 2
    top = bottom + new_h
    left = (width - new_w) // 2
    right = left + new_w 

   
    
    ### STUDENT IMPLEMENTATION END ###

    resized_img = img_resize(img, (new_h, new_w))
    output_shape = (height, width) + ((img.shape[2],) if multi_channel else ())
    output_img = np.ones(output_shape) if pad_type.lower() == 'white' else np.zeros(output_shape)
    
    if multi_channel:
        output_img[bottom:top, left:right, :] = resized_img
    else:
        output_img[bottom:top, left:right] = resized_img
        
    return output_img


def get_resized_db(img_in, target_size=TARGET_SIZE, pad_type='white'):
    if len(img_in) == 0: return np.array([])
    has_channels = len(img_in[0].shape) >= 3
    shape_tuple = (len(img_in), target_size[0], target_size[1]) + ((img_in[0].shape[2],) if has_channels else ())
    output = np.zeros(shape_tuple)
    for idx, img in enumerate(img_in):
        output[idx] = resize_and_pad(img, target_size=target_size, pad_type=pad_type)
    return output

def convert_ndarrays2data_matrix(arr):
    nb_individuals = len(arr)
    dim = np.prod(arr.shape[1:])
    data_mtx = np.zeros((nb_individuals, dim))
    for i in range(nb_individuals): 
        data_mtx[i, :] = np.ravel(arr[i])
    return data_mtx



def my_PCA(data, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca

def project_onto_PCA(n_components, pca_model, data, center=False):
    """
    ========================================================================
    TASK 3.1: CLASS-WISE PCA FEATURE EXTRACTION
    ========================================================================
    For a given PCA model, projects input data and extracts the top 'nb_components' principal components as features.
    """
    if center:
        data = data - pca_model.mean_
    projected = pca_model.transform(data)
    ### STUDENT IMPLEMENTATION START ###
    
    return projected[:, :n_components] 

def visualize_var_pcs(pca, fig_path=None): 
    """
    ========================================================================
    TASK 3.2: PCA SCREE PLOT VISUALIZATION
    ========================================================================
    Plots the individual and cumulative explained variance ratios for each principal component, with a reference line at 90% cumulative variance.
    """
    exp_var_pca = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_pca)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(exp_var_pca)+1), exp_var_pca, alpha=0.6, label='Individual variance')
    ax.plot(range(1, len(cum_var)+1), cum_var, 'r-o', label='Cumulative variance')
    ax.axhline(0.9, color='green', linestyle='--', label='90% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Scree Plot')
    ax.legend()
    
    if fig_path: 
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.show()
    return fig

def plot_whole_db_on_2d(pca, data_mtx, fig_path=None):
    """
    ========================================================================
    TASK 3.3: PCA PROJECTION SCATTER PLOT
    ========================================================================
    Projects the entire dataset onto the first two principal components and visualizes it as a scatter plot, colored by class labels.
    """
    projected_data = project_onto_PCA(2, pca, data_mtx, True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Projection - Full Dataset')
    plt.colorbar(scatter, ax=ax, label='Sample index')
    if fig_path:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.show()

    return fig 

def display_pca_approx(img, pca, target_size=TARGET_SIZE, fig_path=None): 
    """
    ========================================================================
    TASK 4: PCA RECONSTRUCTION APPROXIMATION
    ========================================================================
    For a given input image, reconstructs approximations using increasing numbers of principal components (k) and visualizes the original and reconstructed images side by side, along with their respective L2 reconstruction errors.
    """
    original_image = img.reshape(target_size[0], target_size[1])  # Reshape back to targeted size

    fig = plt.figure(figsize=(15,15))
    plt.subplot(4,4,1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original image")
    plt.axis('off')

    step = (target_size[0] + target_size[1]) // 2

    k_values = [10, 110, 210, 310, 410, 510, 610, 710,
                810, 910, 1010, 1110, 1210, 1310, 1410]

    for i, k in enumerate(k_values, start=2):
        img_pca = pca.transform(img.reshape(1, -1))
        img_pca[:, k:] = 0

        approx_img = pca.inverse_transform(img_pca).reshape(target_size)
        error = np.linalg.norm(original_image - approx_img)

        plt.subplot(4, 4, i)
        plt.imshow(approx_img, cmap='gray')
        plt.title(f"k={k}\nL2={error:.2f}")
        plt.axis('off')
    img_pca = pca.transform(img.reshape(1, -1))
    img_pca[:, k:] = 0

    approx_img = pca.inverse_transform(img_pca).reshape(target_size)
    error = np.linalg.norm(original_image - approx_img)

    plt.subplot(4, 4, i)
    plt.imshow(approx_img, cmap='gray')
    plt.title(f"k={k}\nL2={error:.2f}")
    plt.axis('off')
        

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.show()

    return fig 

def compute_pca_features(pcas, input_data, nb_components=5):
    out = np.zeros((len(input_data), 0))
    for pca in pcas:
        proj = pca.transform(input_data)
        features = proj[:, :min(proj.shape[1], nb_components)]
        out = np.hstack((out, features))
    return out

def compute_hog(image, nb_height_cells=4, nb_width_cells=4, nb_bins=8):
    """
    ========================================================================
    TASK 5: CLASSICAL HISTOGRAM OF ORIENTED GRADIENTS (HOG)
    ========================================================================
    Decomposes an image into local grid patches, evaluates edge orientations, 
    and accumulates magnitudes within discrete orientation channels.
    """
    h_edge = sobel_h(image)
    v_edge = sobel_v(image)
    magnitude = np.sqrt(h_edge**2 + v_edge**2)

    orientation = np.arctan2(v_edge, h_edge)
    orientation[orientation < 0] += np.pi # Constrain range within [0, \pi)

    H, W = image.shape[:2]
    cell_h = H // nb_height_cells
    cell_w = W // nb_width_cells
    bin_width = np.pi / nb_bins

    output = np.zeros((nb_height_cells, nb_width_cells, nb_bins))

    for i in range(nb_height_cells):
        for j in range(nb_width_cells):
            mag_cell = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            ori_cell = orientation[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

            for y in range(cell_h):
                for x in range(cell_w):
                    bin_idx = int(ori_cell[y, x] // bin_width)
                    if bin_idx == nb_bins:
                        bin_idx = nb_bins - 1

                    output[i, j, bin_idx] += mag_cell[y, x]

    return output.reshape(-1)

if __name__ == '__main__':
    print("Step 1: Loading Dataset")
    
    bw_imgs, bw_dogs, labels, label_names = read_and_crop_db(color=False)
    
    if bw_imgs is None:
        print(" SmallDB not found.")
        label_names = {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}
        labels = [0]*152 + [1]*250 + [2]*300 + [3]*300
        
        np.random.seed(42)
        sample_img = np.zeros((70, 70))
        sample_img[20:55, 15:55] = 0.6
        bw_imgs = [sample_img] * len(labels)
        bw_dogs = [sample_img[10:60, 10:60]] * len(labels)
        
        data_mtx = np.random.rand(len(labels), TARGET_SIZE[0] * TARGET_SIZE[1]) * 0.4
        for idx, lbl in enumerate(labels):
            data_mtx[idx, 400:1500] += lbl * 0.1
    else:
        print(" Real dataset successfully resolved.")
        resized_dogs = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type='white')
        data_mtx = convert_ndarrays2data_matrix(resized_dogs)

    # Calculate Shannon Entropy
    db_entropy = entropy_breeds(labels)
    print(f"   - Dataset Balance Shannon Entropy: {db_entropy:.4f}")
    plot_barplot(labels, label_names, fig_path=os.path.join(FIGS, 'class_balance.png'))
    

    print("\n Step 2: Saving Localization & Bounding Box Crops")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(bw_imgs[0], cmap='gray')
    ax0.set_title("Original Asset Image Frame", fontsize=9)
    ax0.axis('off')
    ax1.imshow(bw_dogs[0], cmap='gray')
    ax1.set_title("XML Isolated Target Region", fontsize=9)
    ax1.axis('off')
    fig.savefig(os.path.join(FIGS, 'localization_bounding_box.png'), bbox_inches='tight', dpi=150)
    plt.show()
    

    print("\n Step 3: Saving Aspect-Preserving Scaling Validations")
    asymmetric_test = np.zeros((20, 80))
    asymmetric_test[4:16, 10:70] = 0.8
    stretched_test = resize(asymmetric_test, TARGET_SIZE, anti_aliasing=True)
    padded_test = resize_and_pad(asymmetric_test, target_size=TARGET_SIZE, pad_type='white')
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(stretched_test, cmap='gray')
    ax0.set_title("Stretched Distorted Scaling", fontsize=9)
    ax0.axis('off')
    ax1.imshow(padded_test, cmap='gray')
    ax1.set_title("Aspect-Preserved Padding Scaling", fontsize=9)
    ax1.axis('off')
    
    fig.savefig(os.path.join(FIGS, 'aspect_preserving_scaling.png'), bbox_inches='tight', dpi=150)
    plt.show()
  

    print("\n Step 4: Stratifying Splits and Training Class PCA Engines")
    data_train, data_test, y_train, y_test = train_test_split(
        data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
    )
    y_train, y_test = np.array(y_train), np.array(y_test)

    # Global PCA Scree Plot mapping
    pca_global = my_PCA(data_train)
    visualize_var_pcs(pca_global, fig_path=os.path.join(FIGS,'pca_reconstruction.png'))

    # PCA database visualisation scatter plot
    plot_whole_db_on_2d(pca_global, data_mtx, fig_path=os.path.join(FIGS, "pca_scatter_plots.png")) # Note that this matrix includes both test and training sets
    
    # PCA Approximation plots
    display_pca_approx(data_mtx[0], pca_global, fig_path=os.path.join(FIGS, "pca_approximation.png")) 

    # PCA Feature engineering: Per-class PCA modeling loops
    pca_per_class = []
    nb_pcs_per_class = 5
    for l in sorted(list(label_names.keys())):
        class_subset = data_train[y_train == l]
        pca_per_class.append(my_PCA(class_subset, n_components=nb_pcs_per_class))

    print("\n Step 5: Extracting Localized Structural Features")
    target_canvas = data_train[0].reshape(TARGET_SIZE)
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    axes[0, 0].imshow(target_canvas, cmap='gray')
    axes[0, 0].set_title("Target Image", fontsize=8)
    axes[0, 1].imshow(sobel(target_canvas), cmap='gray')
    axes[0, 1].set_title("Sobel Magnitude", fontsize=8)
    axes[1, 0].imshow(sobel_v(target_canvas), cmap='gray')
    axes[1, 0].set_title("Vertical $g_y$", fontsize=8)
    axes[1, 1].imshow(sobel_h(target_canvas), cmap='gray')
    axes[1, 1].set_title("Horizontal $g_x$", fontsize=8)
    for ax in axes.ravel(): ax.axis('off')
    fig.savefig(os.path.join(FIGS, 'sobel_features.png'), bbox_inches='tight', dpi=150)
    plt.show()

    X_train_pca = compute_pca_features(pca_per_class, data_train, nb_components=nb_pcs_per_class)
    X_train_hog = np.array([compute_hog(img.reshape(TARGET_SIZE)) for img in data_train])
    X_train_combined = np.hstack((X_train_pca, X_train_hog))

    X_test_pca = compute_pca_features(pca_per_class, data_test, nb_components=nb_pcs_per_class)
    X_test_hog = np.array([compute_hog(img.reshape(TARGET_SIZE)) for img in data_test])
    X_test_combined = np.hstack((X_test_pca, X_test_hog))
    
  

    print("\n Step 6: Standardizing Features & Optimizing Classifiers")
     # =========================================================================
    # TASK 4: ANALYSIS 
    # =========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    
  
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train_scaled, y_train)
    train_err = hamming_loss(y_train, knn.predict(X_train_scaled))
    test_err = hamming_loss(y_test, knn.predict(X_test_scaled))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['Training Set Error', 'Testing Generalization Loss'], [train_err, test_err], color=['#a6a6a6', "#FF0000"], height=0.4, edgecolor='black')
    ax.set_xlim(0, max(train_err, test_err) + 0.1)
    ax.set_title("Model Generalization Diagnostic Gap", fontweight='bold')
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width()*100:.1f}%', va='center', fontweight='bold')
    fig.savefig(os.path.join(FIGS, 'model_generalization_error.png'), bbox_inches='tight', dpi=150)
    plt.close()



    print("\n" + "="*50)
    print("DIAGNOSTIC RESULTS SUMMARY")
    print("="*50)
    print(f" Training Error Rate (Hamming Loss): {train_err*100:.2f}%")
    print(f" Testing Error Rate (Generalization Loss):  {test_err*100:.2f}%")
    print("="*50)


    

    np.save("cache/data_train.npy", data_train)
    np.save("cache/data_test.npy", data_test)
    np.save("cache/y_train.npy", y_train)
    np.save("cache/y_test.npy", y_test)
    np.save("cache/labels.npy", labels)
    save_dict("cache/lbl_names.npy",label_names)



