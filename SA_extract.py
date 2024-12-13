
# %%
categories
# %%
# now want to do SA on the text
print('Sentiment analysis')
from transformers import pipeline

xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# %%
# SA scoring
use_col = 'TEXT'
# Ensure text is strings
df['TEXT'] = df['TEXT'].astype(str)

# Define function to convert labels to continuous
# to convert transformer scores to the same scale as the dictionary-based scores
def conv_scores(lab, sco, spec_lab): #insert exact labelnames in order positive, negative og as positive, neutral, negative
    
    converted_scores = []
    
    if len(spec_lab) == 2:
        spec_lab[0] = "positive"
        spec_lab[1] = "negative"

        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            else:
                converted_scores.append(-sco[i])
            
    if len(spec_lab) == 3:
        spec_lab[0] = "positive"
        spec_lab[1] = "neutral"
        spec_lab[2] = "negative"
        
        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            elif lab[i] == "neutral":
                converted_scores.append(0)
            else:
                converted_scores.append(-sco[i])
    
    return converted_scores


xlm_labels = []
xlm_scores = []

for s in df[use_col][:3]:
    # Join to string if list
    if isinstance(s, list):
        s = " ".join(s)
    print('LEN', len(s))

    # get sent-label & confidence to transform to continuous
    sent = xlm_model(s)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))

# function defined in functions to transform score to continuous
xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])
#df["tr_xlm_roberta"] = xlm_converted_scores
xlm_converted_scores

# %%
# Check df for nan values
nan_counts = df.isna().sum()
print("NaN counts per column:")
print(nan_counts)

nan_rows_annotators = df[df[['HUMAN', 'tr_xlm_roberta']].isna().any(axis=1)]
print("Rows with NaN values in SA columns:")
print(nan_rows_annotators)

df.head()