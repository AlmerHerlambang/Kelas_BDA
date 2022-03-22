# memanggil dataset iris 
from sklearn.datasets import load_iris 
iris_ku = load_iris() 
  
# simpan data fitur/kolom (X) dan target (y) 
X = iris_ku.data 
y = iris_ku.target 
  
# simpan nama fitur/kolom (X) dan target (y)
feature_names = iris_ku.feature_names 
target_names = iris_ku.target_names 
  
# tampil nama fitur dan target dataset 
print("Feature names:", feature_names) 
print("Target names:", target_names) 
  
# X dan y adalah numpy arrays 
print("\nType of X is:", type(X)) 
  
# tampilkan 5 baris pertama  
print("\nFirst 5 rows of X:\n", X[:5])


# fitur(X) and target(y) 
X = iris_ku.data 
y = iris_ku.target 

# splitting X dan y untuk data latih  dan uji  
from sklearn.model_selection import train_test_split 
X_latih, X_tes, y_latih, y_tes = train_test_split(X, y, test_size=0.4, random_state=1) 
  
# tampilkan data fitur latih dan uji 
print(X_latih.shape)
print(X_tes.shape) 
  
# tampilkan data target latih dan uji 
print(y_latih.shape) 
print(y_tes.shape)

# pelatihan pada data latih menggunakan KNN (k=3)
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_latih, y_latih) 
  
# melakukan prediksi pada data uji 
y_prediksi = knn.predict(X_tes) 
  
# perbandingan nilai aktual (y_tes) dengan nilai prediksi (y_prediksi) 
from sklearn import metrics 
print("Akurasi model kNN :", metrics.accuracy_score(y_tes, y_prediksi)) 
  
# prediksi menggunakan data sampel dibuat sendiri 
contoh = [[3, 5, 4, 2], [2, 3, 5, 4]] 
preds = knn.predict(contoh) 
pred_species = [iris_ku.target_names[p] for p in preds] 
print("Prediksi :", pred_species) 
  
# saving the model 
#from sklearn.externals import joblib 
#joblib.dump(knn, 'iris_knn.pkl')

#import sklearn.external.joblib as extjoblib
import joblib
joblib.dump(knn, 'iris_knn.pkl')