import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("India_Menu.csv")

x = df.drop(['Menu Category', 'Menu Items', 'Per Serve Size', 'Energy (kCal)', 'Sat Fat (g)', 'Trans fat (g)', 'Total carbohydrate (g)', 'Added Sugars (g)', 'Sodium (mg)'], axis=1)

st.header("MCD Indian Nutrition")
st.write('Data yang berisikan mengenai nutrition di mcd india')
st.subheader('Data Asli')
st.write(df)

#menampilkan elbow
clusters = []
for i in range(1, 10):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Number of Clusters')

ax.annotate('Possible elbow point', xy=(2, 270000), xytext=(2, 100000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow point', xy=(4, 100000), xytext=(4, 200000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :",1,10)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    plt.figure(figsize=(10, 8))

    sns.scatterplot(x='Total fat (g)', y='Total Sugars (g)', hue='Labels', size='Labels', markers=True, palette=sns.color_palette('hls', n_colors=n_clust), data=x)

    for label in x['Labels']:
        plt.annotate(label,
                 (x[x['Labels'] == label]['Total fat (g)'].mean(),
                  x[x['Labels'] == label]['Total Sugars (g)'].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='blue')

    st.header('Cluster Plot')
    st.pyplot()
    st.write(x)

k_means(clust)