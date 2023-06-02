import os
import openai
import json
import re
import numpy as np
import hashlib
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
openai.organization = "OPEN_AI_ORG_ID"
openai.api_key = "OPEN_AI_API_KEY"
openai.Model.list()



def get_interests(text):
    result = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a market research assistant."},
            {"role": "user", "content": "The visitor visited website page with the following content" + text + "based on this input list top 5 possible user interests from high probability to low, include also the price range in which the user is interested, describe the interests in detail and provide supportive fact based arguments and examples"}
    ])

    return result

def summarize_list_of_interests(text):
    result = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a market research assistant."},
            {"role": "user", "content": "Below is list of text fragments which users were interested to read, carefully analise and understand what this customers can have in common. It can be any common activity, trend, buying power etc. Formulate answers in an ordered list. Also suggest what features an ideal smart watch must have to fit needs of this type of customers\n" + text}
    ])

    return result['choices'][0]['message']['content']


def get_embedding(text):
	response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002")
	embeddings = response['data'][0]['embedding']
	return embeddings



def read_file(file):
    with open(file, 'r') as openfile:
        # Reading from json file
        texts = json.load(openfile)
        return texts

def write_file(results, file):
    # Serializing json
    json_object = json.dumps(results, indent=4)
    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)

def construct_interests_from_articles(article_file, interests_file):
    print("[*] Constructing interests from articles")
    texts =  read_file(article_file)
    interests = read_file(interests_file)

    if interests is None or len(interests) == 0:
        results = {}
        print("[*] Constructing interests from articles, requesting OpenAi")
        for key in texts:
            results[key] = [i for i in get_interests(texts[key])['choices'][0]['message']['content'].split('\n') if len(i) > 3]
        write_file(results, interests_file)
    else:
        print("[*] Constructing interests from articles, fetching from cache file")
        results = interests

    return interests




def get_embedding_vectors(interests, container, visits,  cache_file):
    print("[*] Getting embedding vectors from interests")
    embedding_vectors = []
    X = container['X']
    embedding_lookups = container['lookups']


    vectors_and_lookups = read_file
    content = read_file(cache_file)

    if content is not None and len(content) != 0:
     print("[*] Getting embedding vectors from interests, fetching from cache")
     return content
    else:
        print("[*] Getting embedding vectors from interests, requesting OpenAi")
        for user in visits:
            for article in visits[user]:
                for interest_item in interests[article]:
                    embed_vec = get_embedding(interest_item)
                    embed_vec_hash = hashlib.md5(json.dumps(embed_vec, sort_keys=True).encode('utf-8')).hexdigest()
                    embedding_lookups[embed_vec_hash] = interest_item
                    X.append(embed_vec)


        answer = {"vectors": X, "lookups": embedding_lookups}
        write_file(answer, cache_file)
        return answer


def perform_clustering(vectors, lookups, cluster_count):
    print("[*] Clustering the vectors")
    vectors = np.array(vectors)
    n_clusters = cluster_count
    cluster_buckets = {}
    vector_buckets = {}
    cluster_size = [0] * n_clusters
    clust = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(vectors)
    for index, label in enumerate(clust.labels_):
        if label not in cluster_buckets:
            cluster_buckets[label] = []
            vector_buckets[label] = []
        vector = vectors[index].tolist()
        lookup_key = hashlib.md5(json.dumps(vector, sort_keys=True).encode('utf-8')).hexdigest()
        interest = lookups[lookup_key]
        cluster_buckets[label].append(lookup_key)
        vector_buckets[label].append(vector)
        cluster_size[label]+=1
    print("[*] Clustering done, with the following cluster sizes", cluster_size)
    return [cluster_buckets, cluster_size, clust.labels_, vector_buckets]

def combine_cluster_interests(cluster,embedding_vectors):
    cluster_interests = ""
    for i in cluster:
        cluster_interests+=embedding_vectors["lookups"][i] + "\n"
    return cluster_interests


def plot_embeddings(vectors, cluster_labels, colors):
    vectors = vectors.tolist()


    """
        Plot in a scatterplot the embeddings of the words specified in the list "words".
        Include a label next to each point.
    """
    index = 0
    for x, y in vectors:
        plt.scatter(x, y, marker='.', color=colors[cluster_labels[index]])
        index+=1
    plt.show()






user_visits = {
    "user_1": ["ball", "headphones"], # loves baseball and night time activities
    "user_2": ["case", "headphones", "laundry"], # preffers eco sustainable products
    "user_3": ["dog_jacket", "travel_mug","inflatable_dog_kennel","long_range_ebike"], # has dogs and loves to travel
    "user_4": ["long_range_ebike", "escooter","ebike"], # loves electronic scooters and bikes
    "user_5": ["inflatable_dog_kennel", "long_range_ebike" ,"ball", "long_range_ebike"], # loves to play with dog at night
    "user_6": ["bed"], # loves healthy lifestyle,
    "user_7": [ "ball", "bed"] # good sleep and baseball
}

web_page_file = "pages.json"
interests_file = "interests.json"
vectors_file = "vectors_and_lookups.json"
colors = ['red', 'green', 'blue', 'black', "purple", "yellow"]
if not os.path.isfile(web_page_file): write_file({},web_page_file)
if not os.path.isfile(interests_file): write_file({}, interests_file)
if not os.path.isfile(vectors_file): write_file({}, vectors_file)

interests = construct_interests_from_articles(web_page_file, interests_file)
embedding_vectors = get_embedding_vectors(interests, { 'X' : [], 'lookups': {} }, user_visits, vectors_file)
[cluster_buckets, cluster_size, cluster_labels, vector_buckets] = perform_clustering(embedding_vectors['vectors'], embedding_vectors['lookups'], 3)

biggest_cluster_index = cluster_size.index(max(cluster_size))
biggest_cluster = cluster_buckets[biggest_cluster_index]


for index in range(len(cluster_buckets)):
    cluster = cluster_buckets[index]
    vectors = vector_buckets[index]
#     # include all cluster members
#     print("\n\n\n\n [*] Overall customer profile for the cluster ",index, " of size", cluster_size[index], "color", colors[index], "\n\n\n")
#     cluster_interests = combine_cluster_interests(cluster, embedding_vectors)
#     interest_summary = summarize_list_of_interests(cluster_interests)
#     print(interest_summary)
#     print("----------------------------------------------------------------------")

    # include the most far away members from given cluster, more cost efficient
    print("\n\n\n\n [*] Overall customer profile for the cluster ",index, " of size", cluster_size[index], "color", colors[index], "\n\n\n")
    [distant_cluster_buckets, distant_cluster_size, distant_cluster_labels, distant_cluster_vector] = perform_clustering(vectors, embedding_vectors['lookups'],  8)
    diverse_cluster = []
    for i in range(len(distant_cluster_buckets)):
        cls = distant_cluster_buckets[i]
        diverse_cluster =diverse_cluster + [cls[0]]
    cluster_interests = combine_cluster_interests(diverse_cluster, embedding_vectors)
    print("input >>>>\n", cluster_interests)
    interest_summary = summarize_list_of_interests(cluster_interests)
    print("output<<<<\n", interest_summary)
    print("----------------------------------------------------------------------")





X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(np.array(embedding_vectors['vectors']))

plot_embeddings(X_embedded, cluster_labels, colors)
