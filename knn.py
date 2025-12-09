from sklearn import KNeighborsClassifier
from k_fold import k_fold_cv

if __name__ == "__main__":
    knn = KNeighborsClassifier(n_neighbors=5)
    
