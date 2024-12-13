# %%

import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
from datasets import load_dataset

ds = load_dataset("chcaa/memo-canonical-novels")
# make df
df = pd.DataFrame(ds['train'])
df.head()
df.columns


# %%
use_cats = ['O', 'HISTORICAL', 'CANON']

if len(use_cats) == 3:
    nice_labels = {'O': 'Other', 'HISTORICAL': 'Historical', 'CANON': 'Canon'}
    # Combine categories in the 'CATEGORY' column
    df['CATEGORY'] = df['CATEGORY'].replace({
        'LEX_CANON': 'CANON',
        'CANON_HISTORICAL': 'CANON',
        'CE_CANON': 'CANON'
    })

    if len(df['CATEGORY'].unique()) == 3:
        print('--- using only 3 categories ---')
        print('Unique categories:', df['CATEGORY'].unique())
        print('\n')

if len(use_cats) == 5:
    nice_labels = {
        'O': 'Other',
        'HISTORICAL': 'Historical',
        'LEX_CANON': 'Lex Canon',
        'CE_CANON': 'CE Canon',
        'CANON_HISTORICAL': 'Canon/historical'
    }
    group_labels = ['O', 'HISTORICAL', 'LEX_CANON', 'CE_CANON', 'CANON_HISTORICAL']



# %% 
# ML experiments


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder




# %%
import json
# Load the embeddings data
with open('data/meanpool__intfloat__multilingual-e5-large-instruct_identify_author.json', 'r') as f:
    embeddings_data = [json.loads(line) for line in f]

embeddings_df = pd.DataFrame(embeddings_data)
embeddings_df.head()

# %%

embeddings_df['embedding'] = embeddings_df['embedding'].apply(np.array)

# Merge datasets
merged_df = pd.merge(df, embeddings_df, left_on='FILENAME', right_on='filename')

# %%
class_column = 'CATEGORY'
print(merged_df[class_column].value_counts())

# %%
# %%
from nltk.tokenize import sent_tokenize

merged_df['avg_sentence_length'] = merged_df['TEXT'].apply(lambda x: np.mean([len(sent_tokenize(s)) for s in sent_tokenize(x)]))

# %%


from sklearn.preprocessing import OneHotEncoder
# Number of iterations
num_iterations = 20

# Define feature combinations
feature_combinations = {
    'sentence length': lambda df: df['avg_sentence_length'].values.reshape(-1, 1),
    'embeddings': lambda df: np.stack(df['embedding'].values),
    'price': lambda df: df['PRICE'].values.reshape(-1, 1),
    'publisher': lambda df: publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1)),
    'embeddings_price': lambda df: np.hstack([np.stack(df['embedding'].values), df['PRICE'].values.reshape(-1, 1)]),
    'embeddings_publisher': lambda df: np.hstack([np.stack(df['embedding'].values), 
                                                  publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1))]),
    'publisher_price': lambda df: np.hstack([publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1)),
                                             df['PRICE'].values.reshape(-1, 1)]),
    'embeddings_publisher_price': lambda df: np.hstack([np.stack(df['embedding'].values), 
                                                        publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1)),
                                                        df['PRICE'].values.reshape(-1, 1)])
}

# Dictionary to store class-wise metrics for all feature combinations
results = {feature_set: {} for feature_set in feature_combinations}


# OneHotEncoder for the 'publisher' feature
publisher_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

for feature_set_name, feature_set_func in feature_combinations.items():
    print(f"Evaluating feature set: {feature_set_name}")
    
    # Initialize storage for class-wise metrics
    class_performance = {}
    
    for i in range(num_iterations):
        # Step 1: Find the minimum class size
        min_class_size = merged_df[class_column].value_counts().min()

        # Step 2: Down-sample each class
        balanced_df = (
            merged_df.groupby(class_column)
            .apply(lambda x: x.sample(n=min_class_size, random_state=i))  # Vary random_state
            .reset_index(drop=True)
        )

        # Step 3: Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=i).reset_index(drop=True)

        # Step 4: Create feature matrix and target array
        X = feature_set_func(balanced_df)
        y = balanced_df[class_column].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=i)
        clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)  # Get report as a dictionary

        # Store class-wise scores
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue  # Skip non-class entries

            if class_name not in class_performance:
                class_performance[class_name] = {'precision': [], 'recall': [], 'f1-score': []}

            class_performance[class_name]['precision'].append(metrics['precision'])
            class_performance[class_name]['recall'].append(metrics['recall'])
            class_performance[class_name]['f1-score'].append(metrics['f1-score'])
    
    # Calculate mean performance for each class and store results
    results[feature_set_name] = {
        class_name: {
            'mean_precision': np.mean(scores['precision']),
            'mean_recall': np.mean(scores['recall']),
            'mean_f1': np.mean(scores['f1-score'])
        }
        for class_name, scores in class_performance.items()
    }

# Display results
for feature_set_name, class_metrics in results.items():
    print(f"Feature Set: {feature_set_name}")
    for class_name, metrics in class_metrics.items():
        print(f"  Class {class_name}:")
        print(f"    Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    Mean Recall: {metrics['mean_recall']:.3f}")
        print(f"    Mean F1-Score: {metrics['mean_f1']:.3f}")
    print()
# %%
# %%
# again, in which we try to distinguish only two classes

df_two_classes = merged_df[merged_df['CATEGORY'].isin(['CANON', 'O'])]
print(df_two_classes[class_column].value_counts())

# %%
# Number of iterations
num_iterations = 20

# Define feature combinations
feature_combinations = {
    'sentence length': lambda df: df['avg_sentence_length'].values.reshape(-1, 1),
    'embeddings': lambda df: np.stack(df['embedding'].values),
    'price': lambda df: df['PRICE'].values.reshape(-1, 1),
    'publisher': lambda df: publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1)),
    'embeddings_price': lambda df: np.hstack([np.stack(df['embedding'].values), df['PRICE'].values.reshape(-1, 1)]),
    'embeddings_publisher': lambda df: np.hstack([np.stack(df['embedding'].values), 
                                                  publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1))]),
    'publisher_price': lambda df: np.hstack([publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1)),
                                             df['PRICE'].values.reshape(-1, 1)]),
    'embeddings_publisher_price': lambda df: np.hstack([np.stack(df['embedding'].values), 
                                                        publisher_encoder.fit_transform(df['PUBLISHER'].values.reshape(-1, 1)),
                                                        df['PRICE'].values.reshape(-1, 1)])
}

# Dictionary to store class-wise metrics for all feature combinations
results = {feature_set: {} for feature_set in feature_combinations}


# OneHotEncoder for the 'publisher' feature
publisher_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

for feature_set_name, feature_set_func in feature_combinations.items():
    print(f"Evaluating feature set: {feature_set_name}")
    
    # Initialize storage for class-wise metrics
    class_performance = {}
    
    for i in range(num_iterations):
        # Step 1: Find the minimum class size
        min_class_size = df_two_classes[class_column].value_counts().min()
        print('sample size:', min_class_size)

        # Step 2: Down-sample each class
        balanced_df = (
            df_two_classes.groupby(class_column)
            .apply(lambda x: x.sample(n=min_class_size, random_state=i))  # Vary random_state
            .reset_index(drop=True)
        )

        # Step 3: Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=i).reset_index(drop=True)

        # Step 4: Create feature matrix and target array
        X = feature_set_func(balanced_df)
        y = balanced_df[class_column].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=i)
        clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)  # Get report as a dictionary

        # Store class-wise scores
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue  # Skip non-class entries

            if class_name not in class_performance:
                class_performance[class_name] = {'precision': [], 'recall': [], 'f1-score': []}

            class_performance[class_name]['precision'].append(metrics['precision'])
            class_performance[class_name]['recall'].append(metrics['recall'])
            class_performance[class_name]['f1-score'].append(metrics['f1-score'])
    
    # Calculate mean performance for each class and store results
    results[feature_set_name] = {
        class_name: {
            'mean_precision': np.mean(scores['precision']),
            'mean_recall': np.mean(scores['recall']),
            'mean_f1': np.mean(scores['f1-score'])
        }
        for class_name, scores in class_performance.items()
    }

# Display results
for feature_set_name, class_metrics in results.items():
    print(f"Feature Set: {feature_set_name}")
    for class_name, metrics in class_metrics.items():
        print(f"  Class {class_name}:")
        print(f"    Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    Mean Recall: {metrics['mean_recall']:.3f}")
        print(f"    Mean F1-Score: {metrics['mean_f1']:.3f}")
    print()
# %%
