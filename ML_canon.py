# %%
# importing libraries
import pandas as pd
import numpy as np
import json

import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from nltk.tokenize import sent_tokenize

# %%
ds = load_dataset("chcaa/memo-canonical-novels")
# make df
meta = pd.DataFrame(ds['train'])
meta.head()
meta.columns

# %%
# we want to use only 3 categories (O, HISTORICAL, CANON)

# define the nice labels for the categories
nice_labels = {'O': 'Other', 'HISTORICAL': 'Historical', 'CANON': 'Canon'}

    # Combine categories in the 'CATEGORY' column
meta['CATEGORY'] = meta['CATEGORY'].replace({
    'LEX_CANON': 'CANON',
    'CE_CANON': 'CANON',
    'CANON_HISTORICAL': 'CANON', # canon books that are also historical will be considered canon
})

# make sure we only have 3 categories
if len(meta['CATEGORY'].unique()) == 3:
    print('--- using only 3 categories ---')
    print('Unique categories:', meta['CATEGORY'].unique())


# %%
# Load the embeddings data
with open('data/meanpool__intfloat__multilingual-e5-large-instruct_identify_author.json', 'r') as f:
    embeddings_data = [json.loads(line) for line in f]

embeddings_df = pd.DataFrame(embeddings_data)
embeddings_df.head()

# %%
# make sure that the embeddings are in the right format
embeddings_df['embedding'] = embeddings_df['embedding'].apply(np.array)

# Merge embeddings with the main dataframe
merged_df = pd.merge(meta, embeddings_df, left_on='FILENAME', right_on='filename')

# add sentence length as a baseline feature for the model
merged_df['avg_sentence_length'] = merged_df['TEXT'].apply(lambda x: np.mean([len(sent_tokenize(s)) for s in sent_tokenize(x)]))

# %%
class_column = 'CATEGORY'
print(merged_df[class_column].value_counts())

# define the testset size and the number of iterations
test_size = 0.1
num_iterations = 50
print('test size:', test_size, 'num iterations:', num_iterations)

# OneHotEncoder for the 'publisher' feature
publisher_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Define feature combinations
feature_combinations = {
    'avg_sentence_length': lambda df: df['avg_sentence_length'].values.reshape(-1, 1),
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

print('number of feature combinations:', len(feature_combinations))
# %%
# First ML run w 3 classes

# Dictionary to store class-wise metrics for all feature combinations
results = {feature_set: {} for feature_set in feature_combinations}

# dictionary to store results for the confusion matrix
confusion_matrix_results = {feature_set: None for feature_set in feature_combinations}

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=i)
        clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)  # Get report as a dictionary

        # store results for the confusion matrix
        # Compute the confusion matrix for this iteration
        cm = confusion_matrix(y_test, y_pred)
        # Accumulate confusion matrices
        if confusion_matrix_results[feature_set_name] is None:
            confusion_matrix_results[feature_set_name] = cm  # Initialize with the first matrix
        else:
            confusion_matrix_results[feature_set_name] += cm  # Add subsequent matrices

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
            'mean_f1': np.mean(scores['f1-score']),
            'std_f1': np.std(scores['f1-score']),
        }
        for class_name, scores in class_performance.items()
    }

    # Average the confusion matrix across all iterations
    confusion_matrix_results[feature_set_name] = (
        confusion_matrix_results[feature_set_name] / num_iterations
    )
    

# %%
# Display results
for feature_set_name, class_metrics in results.items():
    print(f"Feature Set: {feature_set_name}")
    for class_name, metrics in class_metrics.items():
        print(f"  Class {class_name}:")
        print(f"    Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    Mean Recall: {metrics['mean_recall']:.3f}")
        print(f"    Mean F1-Score: {metrics['mean_f1']:.3f}")
        # and get the SD of the F1 score
        print('    ..')
        print(f"    STD F1-Score: {metrics['std_f1']:.3f}")
    print()

# save them to a txt in results folder
with open('results/ML_3classes_results.txt', 'w') as f:
    for feature_set_name, class_metrics in results.items():
        f.write(f"Feature Set: {feature_set_name}\n")
        for class_name, metrics in class_metrics.items():
            f.write(f"  Class {class_name}:\n")
            f.write(f"    Mean Precision: {metrics['mean_precision']:.3f}\n")
            f.write(f"    Mean Recall: {metrics['mean_recall']:.3f}\n")
            f.write(f"    Mean F1-Score: {metrics['mean_f1']:.3f}\n")
            f.write('    ..\n')
            f.write(f"    STD F1-Score: {metrics['std_f1']:.3f}\n")
        f.write('\n')

# print the confusion matrix for the full feature set
print('Confusion Matrix for the full feature set:')
class_labels = sorted(merged_df['CATEGORY'].unique())  # Ensure labels match matrix order

plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_matrix_results['embeddings_publisher_price'],
    annot=True,
    cmap='Blues',
    xticklabels=class_labels, 
    yticklabels=class_labels
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('figs/ML_3classes_confusion_matrix.png')
plt.show()

# %%

# OK

# %%
# again, ML run in which we try to distinguish only two classes

# keep the histoprical in, but relabel them as 'O'
df_two_classes = merged_df.copy()
df_two_classes['CATEGORY'] = df_two_classes['CATEGORY'].replace({'HISTORICAL': 'O'})

print(df_two_classes[class_column].value_counts())

# %%

# and second ML run over 2 classes

# Dictionary to store class-wise metrics for all feature combinations
results = {feature_set: {} for feature_set in feature_combinations}


# OneHotEncoder for the 'publisher' feature
publisher_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

print('full class sizes', df_two_classes[class_column].value_counts())
print('sample size:', df_two_classes[class_column].value_counts().min())

for feature_set_name, feature_set_func in feature_combinations.items():
    print(f"Evaluating feature set: {feature_set_name}")
    
    # Initialize storage for class-wise metrics
    class_performance = {}
    
    for i in range(num_iterations):
        # Step 1: Find the minimum class size
        min_class_size = df_two_classes[class_column].value_counts().min()

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

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
            'mean_f1': np.mean(scores['f1-score']),
            'std_f1': np.std(scores['f1-score']),
        }
        for class_name, scores in class_performance.items()
    }

# %%
# Display results
for feature_set_name, class_metrics in results.items():
    print(f"Feature Set: {feature_set_name}")
    for class_name, metrics in class_metrics.items():
        print(f"  Class {class_name}:")
        print(f"    Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    Mean Recall: {metrics['mean_recall']:.3f}")
        print(f"    Mean F1-Score: {metrics['mean_f1']:.3f}")
        # and get the SD of the F1 score
        print('    ..')
        print(f"    STD F1-Score: {metrics['std_f1']:.3f}")
    print()

# save them to a txt in results folder
with open('results/ML_2classes_results.txt', 'w') as f:
    for feature_set_name, class_metrics in results.items():
        f.write(f"Feature Set: {feature_set_name}\n")
        for class_name, metrics in class_metrics.items():
            f.write(f"  Class {class_name}:\n")
            f.write(f"    Mean Precision: {metrics['mean_precision']:.3f}\n")
            f.write(f"    Mean Recall: {metrics['mean_recall']:.3f}\n")
            f.write(f"    Mean F1-Score: {metrics['mean_f1']:.3f}\n")
            f.write('    ..\n')
            f.write(f"    STD F1-Score: {metrics['std_f1']:.3f}\n")
        f.write('\n')
        
# %%
