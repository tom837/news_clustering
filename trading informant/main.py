import numpy as np
from matplotlib import pyplot as plt
from data_extraction import extract_csv
from data_cleaning import preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from clustering import *
from sklearn.metrics import pairwise_distances
from data import *
import spacy
import pandas as pd
import pickle
from summary import summarize_text


def process(YEAR):
    print("IMPORTING DATA")
    # titles,dataset,dates = extract_csv("data/guardian_articles.csv",start='2019-09-01',end="2020-09-04")
    # titles,dataset,dates = extract_csv("data/guardian_articles.csv",start='2018-09-01',end="2019-09-04")
    titles, dataset, dates = extract_csv(
        "data/guardian_articles.csv", start=YEAR + "-01-01", end=YEAR + "-12-31"
    )
    N = len(dataset)
    print(N, "articles imported")


    print("CLEANING DATA")
    cleaned_dataset = []
    for article_id in range(N):
        if article_id % 100 == 0:
            print("Article ", article_id, "/", N)
        cleaned_article = preprocess(dataset[article_id])
        if np.size(cleaned_article) != 0:
            cleaned_dataset.append(cleaned_article)


    print("VECTORIZING")
    vectorizer = TfidfVectorizer(min_df=0.05)  # , max_df=0.33)
    X = vectorizer.fit_transform(cleaned_dataset)
    # feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()


    print("CALCULATING SIMILARITY")
    cosine_sim = cosine_similarity(X, X)
    dist = 1.0 - cosine_sim
    dist[dist < 0] = 0


    print("CLUSTERING")
    model = optics_clustering(dist, 5)
    # cluster = dbscan_clustering(dist,0.4,10)
    # plt.hist(model.labels_,bins=100, range=(0,100))
    labels = model.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    n_classified = len(labels) - n_noise_
    print(n_clusters_, "clusters")
    print(n_classified, "classified  | ", n_noise_, "noise")


    print("POSTPROCESSING")
    # Grouping articles by clusters in a dict of shape {cluster_label : [article1_label,article2_label,...]}
    clusters = {}
    for label_id in range(len(labels)):
        if labels[label_id] != -1:  # we don't take the articles not categorized
            if labels[label_id] in clusters.keys():
                clusters[labels[label_id]].append(label_id)
            else:
                clusters[labels[label_id]] = [label_id]

    # Extracting information from clusters
    NER = spacy.load("en_core_web_sm")
    df = []
    # Going through all clusters
    k = 0
    n = len(clusters.keys())
    for label in clusters.keys():
        k += 1
        indices = clusters[label]
        cluster_distances = pairwise_distances(tfidf_scores[indices], metric="cosine")
        variance = np.var(cluster_distances)
        print("Cluster",str(k),"/",str(n)," | ",len(indices),"articles"," | ","Variance =",variance)


        # Creating a dict, for the cluster "label", counting the entities, of shape {entity type : {entity1 : occurences, entity2 : occurences, ...}}
        entities = {} 
        title, text=summarize_text(clusters[label], dataset, titles,3)#get the best suiting title and the summary of all articles
        description = title +"<br />" # Text associated with the label
        for i in range(1,len(text)//100+2):#format the text so it's readable when shown
            description+= text[(i-1)*100:100*i]+"<br />"
        date = dates.iloc[indices[0]] # 
        # Going through all articles
        for i in indices:
            # We get the earliest publication date of the cluster as the reference
            if dates.iloc[i] < date:
                date = dates.iloc[i]

            # Named Entities Recognition, 
            text = NER(cleaned_dataset[i])
            # Going through all entities
            for e in text.ents:
                entity = e.text
                entity_type = e.label_
                if entity_type in entities.keys():
                    if entity in entities[entity_type].keys():
                        entities[entity_type][entity] += 1
                    else:
                        entities[entity_type][entity] = 1
                else:
                    entities[entity_type] = {entity: 1}
        
        # Creation of the line in the dataframe
        entity_type = "GPE"
        if entity_type in entities.keys():
            max_value = max(entities[entity_type].values())
            location = max(entities[entity_type], key=entities[entity_type].get)
            year = date.year
            if location in country_map.keys():
                location = country_map[location]
                line = [location, year, 1, description]
                df.append(line)
            elif location in city_map.keys():
                location = city_map[location]
                line = [location, year, 1, description]
                df.append(line)
            else : print(location, "not recognized")

    print(str(len(df)), "clusters associated to a location")
    # Grouping lines with the same location
    i = 0
    imax = len(df) - 1
    while i < imax - 1:
        change = False
        j = i + 1
        location = df[i][0]
        year = df[i][1]
        while j < len(df) - 1:
            if df[j][0] == location:
                location2, year2, size, description = df.pop(i)
                df[j - 1][1] = min(year, year2)
                df[j - 1][2] += size
                df[j - 1][3] += "<br /><br />" + description
                change = True
                break
            j += 1
        if change == True:
            imax -= 1
        # if we poped an item, no need to increment i
        else:
            i += 1
    print(str(len(df)), "different locations found")
    df = pd.DataFrame(df, columns=["location", "year", "size", "description"])

    #saving the dataframe
    filename = YEAR + ".p"
    pickle.dump(df, open(filename, "wb"))
    print("Dataframe dumped in ", filename)
     
for y in range(2016, 2022):
    y = str(y)
    process(y)
