import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from data_generator import generate_student_data

# Generisanje podataka i čuvanje u CSV
generate_student_data(num_students=100, filename='student_data.csv')

# Učitavanje podataka iz CSV fajla
data = pd.read_csv('student_data.csv')

# Standardizacija podataka
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Attendance', 'Task_Scores', 'Exam_Score', 'Total_Score']])

# Primena metode Elbow za određivanje optimalnog broja klastera
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.show()

# Primena K-Means klasteringa sa optimalnim brojem klastera
optimal_k = 3  # Izaberite broj klastera na osnovu Elbow grafa
kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=10)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Izračunavanje silhouette score
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Vizualizacija rezultata
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Attendance', y='Total_Score', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering of Student Behavior in Online Classroom')
plt.xlabel('Attendance Points (0-20)')
plt.ylabel('Total Score (0-100)')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Ispis klastera
for i in range(optimal_k):
    print(f"\nStudents in Cluster {i}:")
    print(data[data['Cluster'] == i])
