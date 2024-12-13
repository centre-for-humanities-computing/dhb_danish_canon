# %%

import datasets
import pandas as pd
import numpy as np

# %%
from datasets import load_dataset

ds = load_dataset("chcaa/memo-canonical-novels")
# make df
df = pd.DataFrame(ds['train'])
df.head()
df.columns

# %%
# let's try a distribution plot of the groups (CATEGORY) and prices
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

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

    group_labels = ['Other', 'Historical', 'Canon']

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

# Testing
measure = 'PRICE'

# Ensure the unique categories align with group_labels
unique_cats = df['CATEGORY'].unique()
group_labels = [nice_labels[cat] for cat in unique_cats]

# Perform comparisons
for i, group in enumerate(unique_cats):
    for j, other_group in enumerate(unique_cats):
        if i < j:
            group_data = df[df['CATEGORY'] == group][measure].dropna()
            other_group_data = df[df['CATEGORY'] == other_group][measure].dropna()

            # Mann-Whitney U test
            t, p = mannwhitneyu(group_data, other_group_data)
            print(f"Comparing {group_labels[i].upper()} and {group_labels[j].upper()}:")
            print(f"Mann-Whitney U test: U = {t}, p = {p}")
            print('..')

            # t-test
            t, p = ttest_ind(group_data, other_group_data)
            print(f"t-test: t = {t}, p = {p}")
            print('..')

            print('means, stds')
            print(f"Mean {group_labels[i]}: {group_data.mean()}, std: {group_data.std()}")
            print(f"Mean {group_labels[j]}: {other_group_data.mean()}, std: {other_group_data.std()}")
            print('//')




# %% 
# make some nice boxplots showing the scatterpoints as well
def different_proxy_types_boxplots(df, col_name, measure, w, h):
    # Generate a color palette based on the unique values in the column
    unique_categories = df[col_name].unique()
    palette = sns.color_palette("tab20", len(unique_categories))  # Use a palette with distinct colors
    category_colors = dict(zip(unique_categories, palette))  # Map categories to colors
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    sns.set(style="whitegrid", font_scale=1, font='serif')

    # add a dotted grey line that shows the data mean
    ax.axhline(df[measure].mean(), color='lightgrey', linestyle='dashed', linewidth=1.8)
    
    # Create the boxplot
    sns.boxplot(data=df, x=col_name, y=measure, showfliers=False, ax=ax, palette=category_colors,  boxprops=dict(alpha=0.35, linewidth=1))
    
    # Plot all individual points, color-coded by category
    for category, color in category_colors.items():
        category_data = df[df[col_name] == category]
        x_positions = np.random.normal(unique_categories.tolist().index(category), 0.13, size=len(category_data))
        ax.scatter(x_positions, category_data[measure], alpha=0.25, color=color, label=category, s=45, edgecolor=color)
    
    ax.set_ylabel(measure.lower())
    ax.set_xlabel('')
    
    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Show the plot
    plt.tight_layout()
    # rotate the x-axis labels
    plt.xticks(rotation=60)
    plt.xticks(ticks=np.arange(len(unique_categories)), labels=[nice_labels[cat] for cat in unique_categories], fontsize=14)
    plt.show()

    return fig

# Example usage
measure = 'PRICE'
x = different_proxy_types_boxplots(df, 'CATEGORY', measure, 9, 5)


# %%
# make a distribution plot of the prices
# make some nice colors
unique_categories = df['CATEGORY'].unique()

colors = sns.color_palette('tab20', n_colors=len(unique_categories))
plt.figure(figsize=(10, 3))
for i, group in enumerate([df[df['CATEGORY'] == cat][measure] for cat in unique_cats]):
    sns.histplot(group, label=nice_labels[unique_cats[i]], kde=True, stat='density', color=colors[i], alpha=0.2, line_kws={'linewidth': 2})
plt.legend(nice_labels)
plt.xlabel('price')
plt.xlim(0, 9)


# %%

def plot_histograms_two_groups(df, scores_list, group_column='CATEGORY', cutoff=None, logscale=None):
    plots_per_row = 3

    if len(scores_list) <= plots_per_row:
        fig, axes_list = plt.subplots(1, len(scores_list), figsize=(20, 4), dpi=300, sharey=True)
    else:
        rows = len(scores_list) // plots_per_row
        if len(scores_list) % plots_per_row != 0:
            rows += 1
        fig, axes_list = plt.subplots(rows, plots_per_row, figsize=(20, 4 * rows), dpi=300, sharey=True)
        
    fig.tight_layout(pad=3)

    canon = df.loc[df[group_column] == 1]
    noncanon = df.loc[df[group_column]== 0]
    print('len per group', len(canon), len(noncanon))

    labels = [x.replace('_', ' ').lower() for x in scores_list]

    for i, score in enumerate(scores_list):
        plt.tight_layout()

        sns.set(style="whitegrid", font_scale=2, font='serif')

        ax = axes_list.flat[i]

        sns.histplot(data=noncanon[score], ax=ax, color='#38a3a5')
        sns.histplot(data=canon[score], ax=ax, color='lightcoral')

        # Set labels
        ax.set_xlabel(labels[i])
        
        if i >= 1:
            ax.set_ylabel('')  # Set the y-axis label to an empty string
        
        if cutoff is not None:
            ax.set_ylim(0, cutoff)
        if logscale is not None:
            ax.set_xscale('log')

    plt.show()
    return fig

# %%
# make a hetamap of the groups and the publishing houses
# get the unique values for the publishers
publishers = df['PUBLISHER'].unique()
# and drop empty values
publishers = publishers[pd.notnull(publishers)]
# get the unique values for the groups
categories = df['CATEGORY'].unique()

# create a matrix with the counts
matrix = np.zeros((len(publishers), len(categories)))

for i, publisher in enumerate(publishers):
    for j, category in enumerate(categories):
        matrix[i, j] = len(df[(df['PUBLISHER'] == publisher) & (df['CATEGORY'] == category)])

# make it percentages instead
# Calculate the percentage of each group (category) that comes from a given publisher
matrix_perc = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-10)

# Plot the normalized matrix
plt.figure(figsize=(4, 22))
sns.set_style('white')
sns.heatmap(matrix_perc, annot=True, fmt=".1%", xticklabels=categories, yticklabels=publishers, cbar=False, cmap='Reds', mask=matrix == 0)
plt.title("Percentage of books by publisher per category")

plt.xticks(ticks=np.arange(len(categories)) + 0.5, labels=[nice_labels[cat] for cat in categories], rotation=45)
plt.show()

# %%
# we want to get the entropy of each publisher distribution
from scipy.stats import entropy

# Calculate the entropy of each category's distribution of publishers
# make each a list
for cat in unique_categories:
    dist = df[df['CATEGORY'] == cat]['PUBLISHER'].value_counts(normalize=True)
    print(f"Entropy of {nice_labels[cat]} distribution: {round(entropy(dist),3)}")

print('// and with sampling //')

# if we take a random sample of the smaller group (so all should have the size of the smallest group)
# we can calculate the entropy of the distribution of publishers
for cat in unique_categories:
    data = df[df['CATEGORY'] == cat].sample(len(df[df['CATEGORY'] == 'HISTORICAL']))
    dist = data['PUBLISHER'].value_counts(normalize=True)
    print(f"Entropy of {nice_labels[cat]} distribution: {round(entropy(dist),3)}")

# %%
# we want to plot the page count per group
# make a boxplot
sns.set(style="whitegrid", font_scale=1.5, font='serif')
different_proxy_types_boxplots(df, 'CATEGORY', 'PAGES', 9, 5)

# and as a distribution plot
colors = sns.color_palette('tab20', n_colors=len(unique_categories))
plt.figure(figsize=(10, 3))
for i, group in enumerate([df[df['CATEGORY'] == cat]['PAGES'] for cat in unique_cats]):
    sns.histplot(group, label=nice_labels[unique_cats[i]], kde=True, stat='density', color=colors[i], alpha=0.2, line_kws={'linewidth': 2})
plt.legend(nice_labels)
plt.xlabel('pages')
plt.xlim(0, 1000)

# %%




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

# merge with df
metadata = df.copy()

# Merge datasets
merged_df = pd.merge(metadata, embeddings_df, left_on='FILENAME', right_on='filename')

# Create a new column 'classes' as a copy of 'CATEGORY'
merged_df['classes'] = merged_df['CATEGORY']


# %%
class_column = 'classes'

# Step 1: Find the minimum class size
min_class_size = merged_df[class_column].value_counts().min()

# Step 2: Down-sample each class
balanced_df = (
    merged_df.groupby(class_column)
    .apply(lambda x: x.sample(n=min_class_size, random_state=42))
    .reset_index(drop=True)
)

# Step 3: Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the class distribution
print(balanced_df[class_column].value_counts())

# %%
# Extract features and target
X = np.stack(balanced_df['embedding'].values)
y = balanced_df['classes'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# get a confusion matrix
from sklearn.metrics import confusion_matrix

conf_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=clf.classes_, columns=clf.classes_)
conf_df

# make a heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(len(clf.classes_)) + 0.5, labels=['CANON', 'HISTORICAL', 'OTHER'], rotation=0)
plt.yticks(ticks=np.arange(len(clf.classes_)) + 0.5, labels=['CANON', 'HISTORICAL', 'OTHER'], rotation=0)
plt.show()


# %%
# Extract features and target
#balanced_df['PRICE'] = balanced_df['PRICE'].fillna(0)
print(len(balanced_df))
balanced_df['PRICE'] = balanced_df['PRICE'].dropna()
print(len(balanced_df))

X = balanced_df['PRICE'].values.reshape(-1, 1)
y = balanced_df['classes'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# %%
# Extract embeddings and reshape PRICE
embeddings = np.stack(balanced_df['embedding'].values)  # Convert embeddings to a 2D array
price = balanced_df['PRICE'].values.reshape(-1, 1)  # Reshape PRICE into a 2D array

# Combine embeddings and PRICE into a single feature array
X = np.hstack((embeddings, price))  # Horizontally stack embeddings and PRICE
y = balanced_df['classes'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# %%
from nltk.tokenize import sent_tokenize

merged_df['AVG_SENTENCE_LENGTH'] = merged_df['TEXT'].apply(lambda x: np.mean([len(sent_tokenize(s)) for s in sent_tokenize(x)]))

# %%


from sklearn.preprocessing import OneHotEncoder
# Number of iterations
num_iterations = 20

# Define feature combinations
feature_combinations = {
    'sentence length': lambda df: df['AVG_SENTENCE_LENGTH'].values.reshape(-1, 1),
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

# Column names for classes and features
class_column = 'classes'

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
        y = balanced_df['classes'].values

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
