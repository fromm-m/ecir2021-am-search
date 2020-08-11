import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np
import math
from arclus.settings import PREP_ASSIGNMENTS_TEST
df = pd.read_csv(PREP_ASSIGNMENTS_TEST, sep=";")

df['relevance'].loc[(df['relevance'] == "notRelevant")] = 0
df['relevance'].loc[(df['relevance'] == "yesRelevant")] = 1
df['relevance'].loc[(df['relevance'] == "yesVeryRelevant")] = 2


all_gain = []
query_list = df["queryClaimID"].unique()
#Iterate over each claim
for query in query_list:
    #select all premises for a given claim
    df_temp = df.loc[df["queryClaimID"] == query]

    #find out all clusters for a given claim
    predicted_cluster_values_unique = df_temp["premiseClusterID_first512Tokens"].unique()

    #sort clusters according to their id
    predicted_cluster_values_unique = np.sort(predicted_cluster_values_unique)
    predicted_gain = []
    # iterate over all predicted clusters starting with lowest id
    for item in predicted_cluster_values_unique:
        #only select the premises which are in cluster i
        cluster_list = df_temp.loc[df_temp["premiseClusterID_first512Tokens"] == item]

        #calculate max_length of all premises in the cluster i
        max_length = cluster_list.resultClaimsPremiseText.str.len().max()

        #search for the longest premise in cluster i
        premise_represent = cluster_list.loc[cluster_list["resultClaimsPremiseText"].str.len() == max_length]

        #return the relevance value of the longest premise in cluster i
        if math.isnan(max_length):
            predicted_gain.append(0)
        else:
            predicted_gain.append(premise_represent["relevance"].values[0])

    gt_cluster_values_unique = df_temp["premiseClusterID_groundTruth"].sort_values().dropna().unique()
    gt_gain = []
    # iterate over all groundtruth clusters starting with the lowest id
    for item in gt_cluster_values_unique:
        #only select the premises which are in ground_truth_cluster i
        cluster_list = df_temp.loc[df_temp["premiseClusterID_groundTruth"] == item]
        #calculate max_relevance of all premises in the cluster i
        max_relevance = cluster_list.relevance.max()
        #search for the maximal relevant premise in cluster i
        premise_represent = cluster_list.loc[cluster_list["relevance"] == max_relevance]

        #return the relevance value of the most relevant premise in cluster i
        gt_gain.append(premise_represent["relevance"].values[0])
    print("gt_gain", gt_gain)

    #pad groundtruth gain values to the length of predicted_cluster gain values
    if len(gt_gain) < len(predicted_gain):
        gt_gain = np.pad(gt_gain, (0,len(predicted_gain)-len(gt_gain)), 'constant')
        print("padded gt_gain to:", gt_gain)
    #pad predicted_cluster gain values to the length of groundtruth cluster gain values
    elif len(gt_gain) > len(predicted_gain):
        predicted_gain = np.pad(predicted_gain, (0,len(gt_gain)-len(predicted_gain)), 'constant')
        print("padded predicted_gain to:", predicted_gain)
    all_gain.append(ndcg_score(y_true=[gt_gain], y_score=np.array([predicted_gain])))

#calculate the mean for all 30 claim queries
print(np.array(all_gain).mean())
