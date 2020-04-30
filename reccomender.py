import numpy as np
import pandas as pd
from ast import literal_eval
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec
import gensim
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import stellargraph as sg
import keras

##################################################################################
#Data cleaning and preprocessing

#load in data
data = pd.read_csv('/Users/VivekTrivedy/Documents/College Work/Data Mining and Knowledge Discovery/Final Project/movies_metadata.csv')

#Remove duplicate rows
data = data.drop_duplicates(subset = ['id'],keep='first')
data = data.drop_duplicates(subset = ['title'],keep='first')
len(data)

#Only look at released movies
data = data[data['status']=='Released']

#Drop features
colsToDrop = ['runtime','release_date','budget','id','production_companies','belongs_to_collection','homepage','imdb_id','poster_path','original_title','production_countries','spoken_languages','status','tagline','video']
data = data.drop(colsToDrop,axis=1)

#Preprocess genres from string to list of dictionaries
data['genres'] = data['genres'].apply(literal_eval)

#define function to preprocess genres
def preprocessGenres(l):
    holder = []
    for d in l:
        genre = d['name'].lower()
        holder.append(genre)
    return '|'.join(holder)
preprocessGenres(data['genres'].iloc[0])

#Preprocess genres dictionaries into strings
data['genres'] = data['genres'].apply(preprocessGenres)

#Only keep english movies
data = data[data['original_language']=='en']
data = data.drop(['original_language'],axis=1)

#Remove rows with missing values for production companies or genres
data = data.drop(data[data['genres']==''].index)

#Set new index based on title
data = data.set_index('title')

#Drop Nan
data = data.dropna(axis=0)


#Use NLP for our movie descriptions
def textPreprocessing(s):
    s = s.strip() #remove whitespace
    s = s.lower() #lowercase
    s = re.sub(r'\d+','',s) #remove digits
    s= s.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    #use nltk to tokenize and remove stop words
    stopWords = set(stopwords.words('english'))
    tokens = word_tokenize(s)
    tokens = [i for i in tokens if i not in stopWords]
    return tokens
#convert movie descriptions to list of tokenized words
data['overview'] = data['overview'].map(textPreprocessing)
overviews = data['overview']
#Need to specifically name the fields "words" and "tags" to work with gensim
docToAnalyze = namedtuple('docToAnalyze','words tags')
docs = []
#Attach the tag for each movie description as the movie name
for tup in zip(overviews,data.index):
    words = tup[0]
    tag = tup[1]
    docs.append(docToAnalyze(words,[tag]))
#Train model to get a vector representation for each movie
model = Doc2Vec(docs,alpha=0.05,min_alpha=0.025,window = 7,epochs = 10,vector_size=25)
#Replace overviews in data with our new features
replaceOverviews = []
for ind in data.index:
    replaceOverviews.append(model.docvecs[ind])
replaceOverviews = np.array(replaceOverviews)
plotDf = pd.DataFrame(replaceOverviews,columns=['g{}'.format(i) for i in range(1,26)])
plotDf['title'] = data.index
plotDf = plotDf.set_index('title') #make sure we have the same index
data = pd.concat([data,plotDf],axis=1) #concatenate our 25 word emdedding features to data
data = data.drop(['overview'],axis=1) #don't need this after embedding it


#Drop rows where we don't know the genre
data = data.drop(data[data['genres'].map(len)==0].index,axis=0)
one_hot_encoding = data["genres"].str.get_dummies(sep='|')
data = pd.concat([data, one_hot_encoding], axis=1)
data = data.drop(['genres'],axis=1) #we don't need movies column anymore


#Encode varianbles - Adult is binary encoding
data = data.replace('True',1)
data = data.replace('False',0)
data[data['adult']>1]

#Some values in popularity column are string - fix this
data['popularity'] = pd.to_numeric(data['popularity'])

#Drop rows where there are no votes for popularity
data = data.drop(data[data['vote_count']<3].index,axis=0)
data = data.drop(['vote_count'],axis=1)

#Imputation for revenue
revMean = data['revenue'].mean()
data.loc[data.revenue <= 10000, "revenue"] = revMean

#Normalize continuous features - MinMax preserves shape of distribution
colsToNormalize = data.loc[:,['popularity','revenue','vote_average']]
scaler = MinMaxScaler()
scaledCols = pd.DataFrame(scaler.fit_transform(colsToNormalize),columns = ['popularity','revenue','vote_average'],index = data.index)
data.loc[:,['popularity','revenue','vote_average']] = scaledCols
##################################################################################



##################################################################################
#K-Means Clustering
#Optimize our hyperparameter - number of clusters of k using inertia
wcss = []
for i in range(1, 15,2):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15,2), wcss)
plt.title('Inertia Values for Different Values of k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show() #optimal k = 3

clustering_kmeans = KMeans(n_clusters=5, init='k-means++',precompute_distances="auto", n_jobs=-1)
data['clusters'] = clustering_kmeans.fit_predict(data)
data.clusters.value_counts(normalize=True)

reduced_data = PCA(n_components=2).fit_transform(data)
reduced_data
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
results

sns.scatterplot(x="pca1", y="pca2", data=results,hue = list(data['clusters']))
plt.title('K-means Clustering with 2 Dimensions')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()

data = data.drop(['clusters'],axis=1)
##################################################################################


##################################################################################
#Create a GCN to compare with a KNN approach
#Convert data into graph based on KNN - return adjacency list
kNNGraph = kneighbors_graph(data,n_neighbors=7)

G = nx.from_scipy_sparse_matrix(kNNGraph, create_using=nx.DiGraph())
#relabel nodes of G
mapping = dict(zip(G, data.index))
G = nx.relabel_nodes(G,mapping)
print(nx.number_connected_components(G.to_undirected())) #1 connected component
nx.set_node_attributes(G,data.to_dict(orient="index")) #include features in graph

#port networkx to stellargraph
Gs = sg.StellarDiGraph.from_networkx(G,node_features=data)
Gs.number_of_edges()

#Use Unsupervised GCN to get Node Embeddings
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras

# parameter specification
number_of_walks = 1
length = 5
batch_size = 500
epochs = 5
num_samples = [10, 10]
layer_sizes = [40, 30]
learning_rate = 5e-2

unsupervisedSamples = UnsupervisedSampler(Gs, nodes=G.nodes(), length=length, number_of_walks=number_of_walks)

generator = GraphSAGELinkGenerator(Gs, batch_size, num_samples)
train_gen = generator.flow(unsupervisedSamples)

assert len(layer_sizes) == len(num_samples)

graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2")


x_inp, x_out = graphsage.build()

prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),loss=keras.losses.binary_crossentropy,metrics=[keras.metrics.binary_accuracy])

history = model.fit(train_gen,epochs=epochs,verbose=2,use_multiprocessing=False,workers=4,shuffle=True)


node_ids = list(Gs.nodes())
node_gen = GraphSAGENodeGenerator(Gs, batch_size, num_samples).flow(node_ids)

embedding_model = keras.Model(inputs=x_inp[::2], outputs=x_out[0])

node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)


#Visualize with T-Sne
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = data.index
node_gen = GraphSAGENodeGenerator(Gs, batch_size, num_samples).flow(node_ids)

node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

X = node_embeddings
if X.shape[1] > 2:
    transform = TSNE  # PCA

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_ids)
else:
    emb_transformed = pd.DataFrame(X, index=node_ids)
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})




sns.scatterplot(data=emb_transformed)
plt.title('T-SNE Visualization of Embedded Features')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()
emb_transformed.shape

alpha = 0.7
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    cmap="jet",
    alpha=alpha,
    edgecolor='r',
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} Visualization of GCN Embedded Features".format(transform.__name__)
)
plt.show()


##################################################################################


##################################################################################
#Create a new graph with embedded features
embeddedGraph = kneighbors_graph(X,n_neighbors=7)

embeddedGnx = nx.from_scipy_sparse_matrix(embeddedGraph, create_using=nx.DiGraph())

#relabel nodes
mapping = dict(zip(embeddedGnx, data.index))
embeddedGnx = nx.relabel_nodes(embeddedGnx,mapping)
print(nx.number_connected_components(embeddedGnx.to_undirected()))
list(embeddedGnx.neighbors("Interstellar"))

#Use percolation method to find cliques
c = list(nx.community.k_clique_communities(embeddedGnx.to_undirected(),6))
for i in c:
    if 'Interstellar' in i:
        print(i)

list(embeddedGnx.neighbors("The Imitation Game"))






##################################################################################
