from sklearn.datasets import load_iris #load_digits 
from sklearn.feature_selection import SelectKBest, chi2 
X, y = load_iris(return_X_y=True) #load_digits 
iris=load_iris()
class_names=iris.target_names
attribute_names=iris.feature_names
print("class names", class_names)
print("class codes", y)
print("feature names", attribute_names)
print("X initial shape", X.shape)
selector=SelectKBest(chi2, k=3)
selector.fit(X,y)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y) #20
print("X final shape", X_new.shape)

cols_idxs = selector.get_support(indices=True)
print("Relevant features:")
for idx in cols_idxs:
    print(attribute_names[idx])
