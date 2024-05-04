# Method Description:
# In Homework-3, I built a hybrid recommender system of item based CF and model based CF.
# The final prediction was a weighted sum of the predictions from both approaches, where the weights are normalized based on the number of neighbors used in the collaborative filtering step. 

# In the final submission, I have built the recommendation system using a model-based collaborative filtering (CF) approach, specifically employing an XGBoost regressor. 
# I made enhancements to the feature set to optimize model accuracy. Added, business state and category features to capture contextual information. 
# Applied One-hot encoding to the state feature, and employed PCA (Principal Component Analysis) for dimensionality reduction after one-hot encoding for the category feature, which contained a large number of categories. 
# During model training, chose a high value of n_estimators for the XGBoost regressor, and implemented early stopping rounds to halt training at the appropriate iteration, to prevent overfitting. 

# Error Distribution:
# >=0 and <1: 102341
# >=1 and <2: 32766
# >=2 and <3: 6146
# >=3 and <4: 791
# >=4: 0

# RMSE: 0.9520956925441274
# Execution Time: 1051.8167939186096 seconds

import numpy as np
import pandas as pd
import time
import math
import csv
import json
import xgboost as xgb
import datetime
import sys
from pyspark import SparkContext, SparkConf
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings(action='ignore')

TRAIN_FILE_NAME = 'yelp_train.csv'
NUM_PARTITIONS = 30
BUSINESS_FILE_NAME = 'business.json'
USER_FILE_NAME = 'user.json'
CHECKIN_FILE_NAME = 'checkin.json'
TIP_FILE_NAME = 'tip.json'

def get_dict_length(dictionary):
    if dictionary == "None" or dictionary is None:
        return 0
    else:
        return len(dictionary)

def get_list_length(list_str):
    if list_str == "None" or list_str is None:
        return 0
    else:
        return len(list_str.split(","))

def get_max_value(list_str):
    if list_str == "None" or list_str is None:
        return 0
    else:
        return int(sorted(list_str.split(","), reverse=True)[0])

# Function to calculate the average score from a list of scores.
def calculate_average(score_list):
    total_score = sum(score_list)
    return total_score / len(score_list)

# Function to extract business_features
def extract_business_features(business_rdd, business_dict):
    # Extract features from the json and create dictionary
    business_json = business_rdd.map(lambda line: json.loads(line))

    # Extract relevant fields
    business_fields = business_json.map(lambda x: (
        x["business_id"], 
        x["latitude"], 
        x["longitude"], 
        x["stars"], 
        x["review_count"], 
        x["is_open"], 
        get_dict_length(x["attributes"]), 
        get_list_length(x["categories"]), 
        get_dict_length(x["hours"]), 
        x["state"]
    ))

    # Collect into a list
    business_data = business_fields.collect()

    for data in business_data:
        try:
            business_id, latitude, longitude, stars, review_count, is_open, attribute_count, category_count, hours_count, state = data
            business_dict[business_id]["state"] = state
            business_dict[business_id]["latitude"] = latitude
            business_dict[business_id]["longitude"] = longitude
            business_dict[business_id]["stars"] = stars
            business_dict[business_id]["review_count"] = review_count
            business_dict[business_id]["is_open"] = is_open
            business_dict[business_id]["len_attributes"] = attribute_count
            business_dict[business_id]["len_categories"] = category_count
            business_dict[business_id]["len_hours"] = hours_count
        except:
            continue

# Function to create categories
def create_business_categories(business_rdd):
    # Process business categories
    business_categories = business_rdd.map(lambda line: json.loads(line)).map(lambda x: (x["business_id"], x["categories"])).collect()
    categories_df = pd.DataFrame(business_categories, columns=["business_id", "category"])
    categories_df["category"] = categories_df["category"].fillna("")
    categories_df["category"] = categories_df["category"].apply(lambda x: x.split(", "))

    all_categories = set([category for categories in categories_df["category"] for category in categories])

    def has_category(category_list, target_category):
        return 1 if target_category in category_list else 0

    for category in all_categories:
        categories_df[category] = categories_df["category"].apply(lambda x: has_category(x, category))
    
    return categories_df

# Function to extract key value pairs from business-attributes dictionary
def extract_key_value_pairs(dictionary):
    key_value_pairs = {}
    if dictionary is None:
        return {}
    for key, value in dictionary.items():
        if "{" in value:
            temp = [pair.split(": ") for pair in value.replace("'", "").replace("{", "").replace("}", "").split(", ")]
            for pair in temp:
                if pair[1].isdigit():
                    key_value_pairs[pair[0]] = int(pair[1])
                else:
                    key_value_pairs[pair[0]] = pair[1]
        else:
            if value.isdigit():
                key_value_pairs[key] = int(value)
            else:
                key_value_pairs[key] = value
    return key_value_pairs

# Function to create business attributes
def create_business_attributes(business_rdd):
    # Process business attributes
    business_attributes = business_rdd.map(lambda line: json.loads(line)).map(lambda x: (x["business_id"], extract_key_value_pairs(x["attributes"]))).collect()
    attribute_dict = dict(business_attributes)
    attributes_df = pd.DataFrame(attribute_dict).T
    attributes_df = pd.get_dummies(attributes_df, drop_first=True)

    return attributes_df

# Function to perform PCA
def perform_PCA(data, n_components = 5, col_name = ''):
    pca = PCA(n_components=n_components)
    transform_pca = pca.fit_transform(data)
    data_df = pd.DataFrame(transform_pca, index=data.index, columns=[col_name + str(i + 1) for i in range(n_components)])

    return data_df

def extract_users_features(user_rdd, user_dict):
    user_data = user_rdd.map(lambda x: (
        x["user_id"],
        x["review_count"],
        (datetime.date(2021, 3, 10) - datetime.date(
            int(x["yelping_since"].split("-")[0]),
            int(x["yelping_since"].split("-")[1]),
            int(x["yelping_since"].split("-")[2])
        )).days,
        get_list_length(x["friends"]),
        x["useful"],
        x["funny"],
        x["fans"],
        get_list_length(x["elite"]),
        get_max_value(x["elite"]),
        x["average_stars"],
        x["compliment_hot"],
        x["compliment_more"],
        x["compliment_cute"],
        x["compliment_list"],
        x["compliment_note"],
        x["compliment_plain"],
        x["compliment_cool"],
        x["compliment_funny"],
        x["compliment_writer"],
        x["compliment_photos"]
    )).collect()

    for data in user_data:
        try:
            user_id, review_count, days_since_yelping, friend_count, useful, funny, fans, elite_count, max_elite_year, avg_stars, \
            compliment_hot, compliment_more, compliment_cute, compliment_list, compliment_note, compliment_plain, \
            compliment_cool, compliment_funny, compliment_writer, compliment_photos = data

            user_dict[user_id]["review_count"] = review_count
            user_dict[user_id]["date_since"] = days_since_yelping
            user_dict[user_id]["n_friends"] = friend_count
            user_dict[user_id]["useful"] = useful
            user_dict[user_id]["funny"] = funny
            user_dict[user_id]["fans"] = fans
            user_dict[user_id]["n_elite"] = elite_count
            user_dict[user_id]["max_elite"] = max_elite_year
            user_dict[user_id]["avg_stars"] = avg_stars
            user_dict[user_id]["compliment_hot"] = compliment_hot
            user_dict[user_id]["compliment_more"] = compliment_more
            user_dict[user_id]["compliment_cute"] = compliment_cute
            user_dict[user_id]["compliment_list"] = compliment_list
            user_dict[user_id]["compliment_note"] = compliment_note
            user_dict[user_id]["compliment_plain"] = compliment_plain
            user_dict[user_id]["compliment_cool"] = compliment_cool
            user_dict[user_id]["compliment_funny"] = compliment_funny
            user_dict[user_id]["compliment_writer"] = compliment_writer
            user_dict[user_id]["compliment_photos"] = compliment_photos
        except:
            continue

# Function to process checkin data and add it in business features
def process_checkin_data(checkin_rdd, business_dict):
    checkin_data = checkin_rdd.map(lambda line: json.loads(line)) \
        .map(lambda x: (x["business_id"], x["time"])) \
        .map(lambda x: (x[0], list(x[1].values()))) \
        .map(lambda x: (x[0], sum(x[1]), calculate_average(x[1]))).collect()

    for data in checkin_data:
        try:
            business_id, checkin_sum, checkin_avg = data
            business_dict[business_id]["checkin_sum"] = checkin_sum
            business_dict[business_id]["checkin_avg"] = checkin_avg
        except:
            continue

# Function to process tip data and add it in business features
def process_tip_data(tip_rdd, business_dict):
    tip_data_business = tip_rdd.map(lambda line: json.loads(line)) \
                           .map(lambda x: (x["business_id"], (1, x["likes"], len(x["text"])))) \
                           .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
                           .map(lambda x: (x[0], x[1][0], x[1][1] / x[1][0], x[1][2] / x[1][0])) \
                           .collect()

    for data in tip_data_business:
        try:
            business_id, tip_count, avg_likes, avg_tip_length = data
            business_dict[business_id]["n_tip_business"] = tip_count
            business_dict[business_id]["avg_like_business"] = avg_likes
            business_dict[business_id]["avg_tip_len_business"] = avg_tip_length
        except:
            pass

    tip_data_user = tip_rdd.map(lambda line: json.loads(line)) \
                       .map(lambda x: (x["user_id"], (1, x["likes"], len(x["text"])))) \
                       .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
                       .map(lambda x: (x[0], x[1][0], x[1][1] / x[1][0], x[1][2] / x[1][0])) \
                       .collect()

    for data in tip_data_user:
        try:
            user_id, tip_count, avg_likes, avg_tip_length = data
            user_dict[user_id]["n_tip_user"] = tip_count
            user_dict[user_id]["avg_like_user"] = avg_likes
            user_dict[user_id]["avg_tip_len_user"] = avg_tip_length
        except:
            continue

def process_test_train_data(data, isTrain = False):
    data_rows = []
    for val in data:
        row = {}
        for key, value in val[2].items():
            row[key] = value
        for key, value in val[3].items():
            row[key] = value
        for idx, value in enumerate(val[4]):
            row["bus_attr_" + str(idx)] = value
        
        if isTrain:
            row["target"] = val[5]
        data_rows.append(row)
    return data_rows

# Function to create test and train data
def create_test_train_data(train_data, test_data, user_dict, business_dict, attributes_df):
    train_data_rows = process_test_train_data(train_data, isTrain = True)
    train_df = pd.DataFrame.from_dict(train_data_rows)

    test_data_rows = process_test_train_data(test_data)
    test_df = pd.DataFrame.from_dict(test_data_rows)

    X_train = train_df.loc[:, train_df.columns != 'target']
    y_train = train_df["target"]
    X_test = test_df.iloc[:, :]

    train_len = len(X_train)

    dataset = pd.concat(objs=[X_train, X_test], axis=0)
    dataset_one_hot = pd.get_dummies(dataset, drop_first=True)

    X_train = dataset_one_hot[:train_len]
    X_test = dataset_one_hot[train_len:]

    # X_train_final = X_train[:]
    # X_test_final = X_test[:]

    return X_train, y_train, X_test

# Function to preprocess the training data
def preprocess_data(file_path):
    # Read input file as an RDD with 30 partitions
    data_rdd = sc.textFile(file_path, NUM_PARTITIONS)
    
    # Extract the header line
    header = data_rdd.first()

    # Filter out the header from the data
    processed_data_rdd = data_rdd.filter(lambda line: line != header).map(lambda x: x.split(','))
    return processed_data_rdd

if __name__ == '__main__':
    # Start time for the process
    start_time = time.time()

    # Parsing command line arguments
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    conf = SparkConf().setAppName("RecommendationSystem").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # Set the log level to ERROR
    sc.setLogLevel("ERROR")

    # Create RDDs
    train_rdd = preprocess_data(folder_path + TRAIN_FILE_NAME)
    test_rdd = preprocess_data(test_file_name)

    # Create lists of unique users
    user_to_train = set(train_rdd.map(lambda x: x[0]).distinct().collect())
    user_to_test = set(test_rdd.map(lambda x: x[0]).distinct().collect())
    user_list = list(set(user_to_train | user_to_test))

    # Create lists of unique businesses
    business_to_train = set(train_rdd.map(lambda x: x[1]).distinct().collect())
    business_to_test = set(test_rdd.map(lambda x: x[1]).distinct().collect())
    business_list = list(set(business_to_train | business_to_test))

    del business_to_train
    del business_to_test
    del user_to_train
    del user_to_test

    # Create dictionaries for users and businesses
    business_dict = {business: {} for business in business_list}
    user_dict = {user: {} for user in user_list}

    # Extract features from JSON files and store them in dictionaries
    business_rdd = sc.textFile(folder_path + BUSINESS_FILE_NAME, NUM_PARTITIONS)
    extract_business_features(business_rdd, business_dict)

    # Create business categories
    categories_df = create_business_categories(business_rdd)

    # Create business attributes
    attributes_df = create_business_attributes(business_rdd)

    # Perform PCA on attributes
    attributes_df = perform_PCA(attributes_df, n_components = 5, col_name = 'attr_pca_')

    # Perform PCA on categories
    categories_pca_df = perform_PCA(categories_df.iloc[:, 3:], n_components = 10, col_name = 'pca_')
    categories_pca_df["business_id"] = categories_df["business_id"]

    for index, row in categories_pca_df.iterrows():
        business_id = row["business_id"]
        try:
            for i in range(n_components):
                business_dict[business_id]["category_pca_" + str(i + 1)] = row["pca_" + str(i + 1)]
        except:
            continue
    
    # Extract features from user JSON files and store them in dictionaries
    user_rdd = sc.textFile(folder_path + USER_FILE_NAME, NUM_PARTITIONS).map(lambda line: json.loads(line)).filter(lambda x: x["user_id"] in user_list)
    extract_users_features(user_rdd, user_dict)

    # Add checkin feature in business dictionary
    checkin_rdd = sc.textFile(folder_path + CHECKIN_FILE_NAME, NUM_PARTITIONS)
    process_checkin_data(checkin_rdd, business_dict)    

    # Process tip data and add it in business dictionary
    tip_rdd = sc.textFile(folder_path + TIP_FILE_NAME, NUM_PARTITIONS)
    process_tip_data(tip_rdd, business_dict)

    print(f"Duration for data process and feature creation: {time.time() - start_time}")

    # Create test and train data
    train_data = train_rdd.map(lambda row: (row[0], row[1], row[2])).map(lambda x: (x[0], x[1], user_dict[x[0]], business_dict[x[1]], attributes_df.loc[x[1]], float(x[2]))).collect()
    test_data = test_rdd.map(lambda row: (row[0], row[1])).map(lambda x: (x[0], x[1], user_dict[x[0]], business_dict[x[1]], attributes_df.loc[x[1]])).collect() 
    X_train, y_train, X_test = create_test_train_data(train_data, test_data, user_dict, business_dict, attributes_df)

    # Train XGBoost Regressor
    xgb_regressor_1 = xgb.XGBRegressor(
        objective='reg:linear',
        colsample_bytree=0.7,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        random_state=0,
        min_child_weight=4,
        reg_alpha=1.0,
        reg_lambda=1.0
    )
    
    # Train XGBoost Regressor
    xgb_regressor_2 = xgb.XGBRegressor(
        objective='reg:linear',
        colsample_bytree=0.7,
        learning_rate=0.05,
        max_depth=8,
        n_estimators=300,
        subsample=0.8,
        random_state=0,
        min_child_weight=4,
        reg_alpha=1.0,
        reg_lambda=1.0
    )

    xgb_regressor_1.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_train, y_train)], early_stopping_rounds=5)
    xgb_regressor_2.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_train, y_train)], early_stopping_rounds=5)
    print("XGBoost Training Completed")

    # Predict on test data
    predictions_1 = xgb_regressor_1.predict(X_test)
    predictions_2 = xgb_regressor_2.predict(X_test)
    
    final_predictions = (predictions_1 + predictions_2) / 2

    answer = []
    for test_data, predicted_score in zip(test_data, final_predictions):
        user_id, business_id = test_data[:2]
        if predicted_score > 5:
            answer.append((user_id, business_id, 5.0))
        elif predicted_score < 1:
            answer.append((user_id, business_id, 1.0))
        else:
            answer.append((user_id, business_id, predicted_score))

    with open(output_file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["user_id", "business_id", "prediction"])
        for row in answer:
            writer.writerow(row)

    print("File Write Completed")
    
    # Stop the Spark context
    sc.stop()

    # Calculate end time
    end_time = time.time()

    # Print the duration
    print("Duration for the script:", end_time-start_time)
