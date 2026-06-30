#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import visualization library for statistical plots
import seaborn as sns
# Import function to load the iris dataset
from sklearn.datasets import load_iris
# Import K-means clustering algorithm
from sklearn.cluster import KMeans
# Import pandas library for data manipulation and analysis
import pandas as pd


# In[2]:


# Load the iris dataset from scikit-learn
iris = load_iris()


# In[3]:


# Extract feature matrix (sepal/petal measurements) from iris dataset
X = pd.DataFrame(iris.data)
# Set column names to feature names for better readability
X.columns = iris.feature_names
# Extract target labels (species classifications) from iris dataset
y = iris.target


# In[4]:


X.head()


# In[5]:


y


# In[17]:


# Create a scatter plot using seaborn with X coordinates from first column and Y coordinates from second column
sns.scatterplot(
    x=X['sepal length (cm)'],
    y=X['sepal width (cm)'],
    c=y
)


# In[21]:


# Import StandardScaler for feature normalization
from sklearn.preprocessing import StandardScaler
# Create a StandardScaler instance
scaler = StandardScaler()
# Fit the scaler to the data and transform X to have zero mean and unit variance
X_scaled = scaler.fit_transform(X)


# In[35]:


# Import PCA (Principal Component Analysis) from scikit-learn
from sklearn.decomposition import PCA

# Initialize PCA with 2 components to reduce dimensionality to 2D
pca = PCA(n_components=2)

# Fit PCA to the scaled data and transform it to 2 principal components
pca_data = pca.fit_transform(X_scaled)


# In[43]:


# Initialize empty list to store Within-Cluster Sum of Squares values
wcss = []

# Loop through different numbers of clusters (1 to 10) to find optimal k
for k in range(1,11):
    # Create KMeans clustering model with k clusters
    kmeans = KMeans(n_clusters = k)
    
    # Fit the model to the scaled data
    kmeans.fit(pca_data)
    
    # Store the inertia (WCSS) value for this k
    wcss.append(kmeans.inertia_)


# In[44]:


wcss


# In[47]:


sns.lineplot(x=range(1,11),y=wcss,marker ="o")


# In[83]:


kmeans = KMeans(n_clusters = 3, random_state = 10)
labels = kmeans.fit_predict(pca_data)
sns.scatterplot(
    x=pca_data[: ,0],
    y=pca_data[ : ,1],
    c=labels
)
sns.scatterplot(x=kmeans.cluster_centers_[:, 0],y=kmeans.cluster_centers_[:, 1], marker = "x"  , c="red" , s=200)


# In[85]:


kmeans = KMeans(n_clusters = 3)
labels = kmeans.fit_predict(pca_data)
sns.scatterplot(
    x=X['sepal length (cm)'],
    y=X['petal length (cm)'],
    c=y
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




