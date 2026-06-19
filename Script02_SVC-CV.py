import os
import pathlib
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skimage import io
from skimage.color import rgb2gray
from Script01_PreprocessingExploration import resize_and_pad


# Ensure you have your utils.py file or these functions available in your environment
try:
    from Script01_PreprocessingExploration import compute_hog, my_PCA, TARGET_SIZE, read_and_crop_db, get_resized_db, convert_ndarrays2data_matrix, load_dict
except ImportError:
    # Safe defaults if utils are missing during initial setup
    TARGET_SIZE = (64, 64)
    def read_and_crop_db(subset_dogs=None, TO_DB="SmallDB", color=True): return np.zeros((100, TARGET_SIZE[0]*TARGET_SIZE[1])), np.zeros(100, dtype=int)
    def get_resized_db(dogs, target_size=TARGET_SIZE, pad_type='white'): return np.zeros((len(dogs), target_size[0]*target_size[1]))
    def my_PCA(data, n_components=5): pass
    def convert_ndarrays2data_matrix(arr): return arr
    def compute_hog(image, nb_h_cells, nb_w_cells, nb_bins): return np.zeros(nb_h_cells*nb_w_cells*nb_bins)
    def load_dict(f): return {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}


OUTPUT_DIR = os.path.join('..', 'folder_code2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIGS = "figures"
os.makedirs(FIGS, exist_ok=True)



#Chargement des données
if os.path.exists("cache/data_test.npy"):
    X_train = np.load("cache/data_train.npy")
    X_test = np.load("cache/data_test.npy")
    y_train = np.load("cache/y_train.npy")
    y_test = np.load("cache/y_test.npy")
    label_names = load_dict("cache/lbl_names.npy")
    labels = np.load("cache/labels.npy")
else:
    raise FileNotFoundError(" Cache arrays not found. Run the Lab 1 setup script to generate the design matrices.")


# =========================================================================
# TASK 1: QUANTIFYING TRAIN/TEST DISTRIBUTION RESEMBLANCE 
# =========================================================================
# Prevent the "Time Machine Effect". We must ensure our random splits 
# have similar class distributions before applying any transformations.

### STUDENT IMPLEMENTATION START ###
#vérifier le split train/test
# What other similarity metrics could you have used here? (e.g., KL Divergence, Wasserstein Distance, etc.) How would you implement them mathematically?
classes_train, counts_train = np.unique(y_train, return_counts=True)
classes_test, counts_test = np.unique(y_test, return_counts=True)

train_dist = counts_train / len(y_train)
test_dist = counts_test / len(y_test)

similarity_score = np.dot(train_dist, test_dist) / (
    np.linalg.norm(train_dist) * np.linalg.norm(test_dist)
)


### STUDENT IMPLEMENTATION END ###
print(f"   - Train/Test Resemblance (Cosine Similarity): {similarity_score:.6f}")
# ===== Figure TASK 1: Distribution train/test/original =====

classes = np.unique(labels)

train_counts = np.array([np.sum(y_train == c) for c in classes])
test_counts = np.array([np.sum(y_test == c) for c in classes])
original_counts = np.array([np.sum(labels == c) for c in classes])

train_probs = train_counts / len(y_train)
test_probs = test_counts / len(y_test)
original_probs = original_counts / len(labels)

class_names = [label_names[int(c)] for c in classes]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(8, 4))

plt.bar(x - width, train_probs, width, label="Training Split")
plt.bar(x, test_probs, width, label="Testing Split")
plt.bar(x + width, original_probs, width, label="Original Population")

plt.title("Dataset Distribution Mapping Analysis")
plt.ylabel("Empirical Class Probabilities")
plt.xticks(x, class_names, rotation=20)
plt.legend()


plt.tight_layout()
plt.savefig(os.path.join(FIGS, "dataset_distribution_mapping.png"), dpi=150, bbox_inches="tight")
plt.show()

# =========================================================================
# TASK 2: CUSTOM PIPELINE TRANSFORMER WRAPPERS 
# =========================================================================
# The "Lego Block" Philosophy: We wrap our custom functions into classes
# that inherit from BaseEstimator and TransformerMixin.
#création des transformers
class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, nb_h_cells=4, nb_w_cells=4, nb_bins=8):
        self.nb_h_cells = nb_h_cells
        self.nb_w_cells = nb_w_cells
        self.nb_bins = nb_bins
        
    def fit(self, X, y=None):
        return self # HOG requires no training, so fit does nothing
        
    def transform(self, X):
        return np.array([compute_hog(img.reshape(TARGET_SIZE), self.nb_h_cells, self.nb_w_cells, self.nb_bins) for img in X])


class PCAInfoPreprocessing(BaseEstimator, TransformerMixin):
    """
    In Module 1, class-specific PCA was computed manually via loops.
    To prevent Data Leakage, we automate this inside a scikit-learn transformer.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca_per_class = []
        
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Supervised target array 'y' is required to fit class-specific subspaces.")
        
        
        self.pca_per_class = []

        for cls in np.unique(y):
            X_cls = X[y == cls]
            pca = my_PCA(X_cls, n_components=self.n_components)
            self.pca_per_class.append(pca)        
        
        
        ### STUDENT IMPLEMENTATION END ###
        return self
        
    def transform(self, X):
        out = np.zeros((len(X), 0))

        for pca in self.pca_per_class:
            transformed = pca.transform(X)
            for pca in self.pca_per_class:
                transformed = pca.transform(X)[:, :self.n_components]
                out = np.hstack((out, transformed))


        return out


# =========================================================================
# TASK 3: END-TO-END AUTOMATED PIPELINE ARCHITECTURES (See Slide 5)
# =========================================================================
print("\n Step 2: Assembling automated Pipeline and FeatureUnion components...")
#pipeline PCA + HOG + SVC
all_features = FeatureUnion([
    ("pca_info", PCAInfoPreprocessing()),
    ("edge_info", EdgeInfoPreprocessing())
])

pipeline_svc = Pipeline([
    ("minmax_scaler", MinMaxScaler()),
    ("all_features", all_features),
    ("standard_scaler", StandardScaler()),
    ("svc", SVC(kernel="linear"))
])





# =========================================================================
# TASK 4: HYPERPARAMETER OPTIMIZATION SELECTION SPACE (See Slide 6)
# =========================================================================
print("\n Step 3: Tuning via GridSearchCV...")

if pipeline_svc is not None:
    # Switch to RBF kernel for complex boundary mapping
    ppipeline_svc = Pipeline([
    ("minmax_scaler", MinMaxScaler()),
    ("all_features", all_features),
    ("standard_scaler", StandardScaler()),
    ("svc", SVC())])

    param_grid = [
    {
        "svc__kernel": ["linear"],
        "svc__C": [ 1, 10, 50],
        "all_features__pca_info__n_components": [ 5, 10],
        #"all_features__edge_info__nb_h_cells": [4],
        #"all_features__edge_info__nb_w_cells": [4],
        #"all_features__edge_info__nb_bins": [8],
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": [ 20,50,75,100],
        "svc__gamma": [ 0.005, 0.01, 0.02],
        #"all_features__pca_info__n_components": [8,10, 12],
        #"all_features__edge_info__nb_h_cells": [4],
        #"all_features__edge_info__nb_w_cells": [4],
        #"all_features__edge_info__nb_bins": [8],
    }
]
 
    
    
    
    ### STUDENT IMPLEMENTATION END ###
    #GridSearchCV
    # We use 3-Fold Cross Validation to test the parameters safely
    grid_search = GridSearchCV(pipeline_svc, param_grid=param_grid, cv=3, n_jobs=-1,)
    
    print("   Training the Grid Search ")

    grid_search.fit(X_train, y_train)
    #Résultats GridSearch
    print(f"    Optimal Parameters Identified: {grid_search.best_params_}") #meilleur mean test score 
    print("Best CV mean score:", grid_search.best_score_)
    print(f"    Generalization Score on Testing Partition: {grid_search.score(X_test, y_test)*100:.2f}%")
    

#Confusion matrix + precision/recall/F1
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("True labels:     ", y_test[:10])
print("Predicted labels:", y_pred[:10])

test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[label_names[i] for i in sorted(label_names)]
)

disp.plot()
plt.savefig(os.path.join(FIGS, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    target_names=[label_names[i] for i in sorted(label_names)]
))



# =========================================================================
# TASK 5: MULTICLASS STRATEGY COMPARISON (See Slides 7 & 8)
# =========================================================================
# Support Vector Machines are binary. Let's compare the "Round Robin" (OvO)
# strategy against the "Me Against the World" (OvR) strategy.
#comparaison OvO vs OvR
print("\n Step 4: Comparing  OvO vs. OvR ")


pipeline_ovo = Pipeline([
    ("minmax_scaler", MinMaxScaler()),
    ("all_features", all_features),
    ("standard_scaler", StandardScaler()),
    ("ovo", OneVsOneClassifier(SVC(kernel="rbf")))
])
start_ovo=time.time()
pipeline_ovo.fit(X_train, y_train)
ovo_time = time.time() - start_ovo
ovo_score = pipeline_ovo.score(X_test, y_test)


# --- OvR Implementation ---
pipeline_ovr = Pipeline([
    ("minmax_scaler", MinMaxScaler()),
    ("all_features", all_features),
    ("standard_scaler", StandardScaler()),
    ("ovr", OneVsRestClassifier(SVC(kernel="rbf")))
])

start_ovr = time.time()
pipeline_ovr.fit(X_train, y_train)
ovr_time = time.time() - start_ovr
ovr_score = pipeline_ovr.score(X_test, y_test)



print(f"   One-vs-One (OvO) Strategy Score: {ovo_score*100:.2f}%")
print(f"   One-vs-Rest (OvR) Strategy Score: {ovr_score*100:.2f}%")
print(f"   One-vs-One (OvO) Training Time: {ovo_time:.2f} seconds")
print(f"   One-vs-Rest (OvR) Training Time: {ovr_time:.2f} seconds")
import matplotlib.pyplot as plt

strategies = ["One-vs-One (OvO)", "One-vs-Rest (OvR)"]

accuracies = [ovo_score, ovr_score]
C = len(np.unique(y_train))

#C = 1000
model_counts = [
    C * (C - 1) // 2,  # OvO
    C                  # OvR
]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(strategies, accuracies)
plt.color_sequences=["blue ", "orange"]
plt.title("Strategic Performance Comparison")
plt.ylabel("Generalization Score (Accuracy)")
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.bar(strategies, model_counts)
plt.title("Computational Complexity Profile")
plt.ylabel("Total Base Model Estimators Count")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "ovo_ovr_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()





#grid_search.fit(X_train, y_train)


#print("Best parameters:", grid_search.best_params_)
#print("Best CV score:", grid_search.best_score_)
#print(f"Generalization Score on Testing Partition: {grid_search.score(X_test, y_test)*100:.2f}%")

#Test avec une vraie image
#img_path = "test4.jpg"

#img = io.imread(img_path)

#if len(img.shape) == 3:
#    img = rgb2gray(img)

#img_resized = resize_and_pad(img, target_size=TARGET_SIZE, pad_type="white")
#X_new = convert_ndarrays2data_matrix(np.array([img_resized]))

#best_model = grid_search.best_estimator_
#pred = best_model.predict(X_new)[0]

#print("Predicted breed:", label_names[pred])


#For the rapport 
#"import pandas as pd

#results = pd.DataFrame(grid_search.cv_results_)
#results = results.sort_values("rank_test_score")



#simple_results = results[[
 #   "param_svc__kernel",
  #  "param_svc__C",
   # "param_svc__gamma",
    #"param_all_features__pca_info__n_components",
    #"mean_test_score",
    #"rank_test_score"
#]].copy()

#simple_results = simple_results.sort_values("rank_test_score")

#print(simple_results.head(5))
