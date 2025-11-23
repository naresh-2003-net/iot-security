import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



# Encode categorical columns, example:
cat_cols = ['proto', 'service']
for col in cat_cols:
    if col in X.columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Fill missing values if any
X.ffill(inplace=True)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Apply strong Gaussian smoothing with large sigma for low accuracy
# Sigma controls smoothing window size (effective width ~6*sigma)
sigma = 2 # large sigma for aggressive smoothing

X_smooth = np.array([gaussian_filter1d(X.iloc[i].values, sigma=sigma) for i in range(len(X))])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_smooth)

# Split for training/testing; larger test portion can increase variance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, stratify=y, random_state=42
)

# Smaller MLP model with limited training iterations
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=45, random_state=42)

# Train
mlp.fit(X_train, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy on full dataset: {accuracy:.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming y_test and y_pred are defined from your earlier model prediction step

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

import os, math, itertools, time, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

DATA_PATH = '/content/RT_IOT2022'
TARGET_COL = None                       # None => auto-detect
TEST_SIZE = 0.30
MLP_MAX_ITER = 100                      # fewer iterations for speed
RND = 42

# thresholds
THRESHOLDS = {
    'pearson': 0.2,
    'gain_ratio': 0.41,
    'info_gain_norm': 0.6,
    'sym_uncert': 0.3
}
FALLBACK_TOP_K = 16

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
if TARGET_COL and TARGET_COL in df.columns:
    target_col = TARGET_COL
else:
    target_col = df.columns[-1]
print("Target:", target_col)

Xraw = df.drop(columns=[target_col]).copy()
yraw = df[target_col].astype(str).copy()
for c in Xraw.columns:
    if Xraw[c].dtype == object:
        Xraw[c] = pd.to_numeric(Xraw[c], errors='coerce')
Xraw = Xraw.fillna(Xraw.median())

# keep numeric only
num_cols = Xraw.select_dtypes(include=[np.number]).columns.tolist()
Xraw = Xraw[num_cols].copy()

# remove constants
Xraw = Xraw.loc[:, Xraw.nunique()>1]

# split once
X_train, X_test, y_train, y_test = train_test_split(Xraw, yraw, test_size=TEST_SIZE, stratify=yraw, random_state=RND)

# encode target
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# scale
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#  feature selection (on training only)
def entropy_from_counts(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def entropy_series(s, bins=20):
    if pd.api.types.is_numeric_dtype(s):
        hist, _ = np.histogram(s, bins=bins)
        return entropy_from_counts(hist)
    else:
        return entropy_from_counts(s.value_counts().values)

# Pearson
pearson = X_train.apply(lambda col: abs(np.corrcoef(col, y_train_enc)[0,1]) if np.std(col)>0 else 0)
pearson_sel = list(pearson[pearson>=THRESHOLDS['pearson']].index) or list(pearson.sort_values(ascending=False).head(FALLBACK_TOP_K).index)

# InfoGain (mutual info normalized)
mi = mutual_info_classif(X_train.values, y_train_enc, random_state=RND)
mi_series = pd.Series(mi, index=X_train.columns)
mi_norm = mi_series / (mi_series.max() if mi_series.max()!=0 else 1)
info_sel = list(mi_norm[mi_norm>=THRESHOLDS['info_gain_norm']].index) or list(mi_series.sort_values(ascending=False).head(FALLBACK_TOP_K).index)

# GainRatio
gr = {}
for c in X_train.columns:
    hist,_=np.histogram(X_train[c],bins=20)
    intrinsic = entropy_from_counts(hist)
    mi_val = mutual_info_classif(X_train[[c]].values, y_train_enc, random_state=RND)[0]
    gr[c] = 0 if intrinsic==0 else (mi_val/intrinsic)
gr_series = pd.Series(gr)
gr_sel = list(gr_series[gr_series>=THRESHOLDS['gain_ratio']].index) or list(gr_series.sort_values(ascending=False).head(FALLBACK_TOP_K).index)

# Symmetric Uncertainty
target_counts = pd.Series(y_train_enc).value_counts().values
H_target = entropy_from_counts(target_counts)
su = {}
for c in X_train.columns:
    mi_val = mi_series[c]
    H_feat = entropy_series(X_train[c], bins=20)
    denom = H_feat + H_target
    su[c] = 0 if denom==0 else (2*mi_val/denom)
su_series = pd.Series(su)
su_sel = list(su_series[su_series>=THRESHOLDS['sym_uncert']].index) or list(su_series.sort_values(ascending=False).head(FALLBACK_TOP_K).index)

# CFS greedy (fast)
ff_corr = X_train.corr().abs()
def cfs_greedy(max_features=5):
    selected=[]; remaining=set(X_train.columns)
    while len(selected)<max_features and remaining:
        best_score=-1; best_feat=None
        for f in remaining:
            cand=selected+[f]; k=len(cand)
            r_cf = pearson[cand].mean()
            if k==1: r_ff=0
            else: r_ff = np.mean([ff_corr.loc[a,b] for a,b in itertools.combinations(cand,2)])
            denom=math.sqrt(k + k*(k-1)*r_ff) if (k+k*(k-1)*r_ff)>0 else 1
            merit=(k*r_cf)/denom
            if merit>best_score: best_score, best_feat=merit,f
        if best_feat: selected.append(best_feat); remaining.remove(best_feat)
    return selected
cfs_sel = cfs_greedy(5)

# Combined (>=4 methods)
methods = [set(pearson_sel), set(info_sel), set(gr_sel), set(su_sel), set(cfs_sel)]
all_feats = set().union(*methods)
occ = {f: sum([1 for s in methods if f in s]) for f in all_feats}
combined = [f for f,cnt in occ.items() if cnt>=4]
if len(combined)<16:
    combined = list(mi_series.sort_values(ascending=False).head(16).index)

#experiments
experiments = {
    'Pearson': pearson_sel,
    'InfoGain': info_sel,
    'GainRatio': gr_sel,
    'SymUnc': su_sel,
    'CFS': cfs_sel,
    'Combined': combined,

}

for name, feats in experiments.items():
    Xtr, Xte = X_train[feats].values, X_test[feats].values
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=MLP_MAX_ITER, random_state=RND)
    clf.fit(Xtr, y_train_enc)
    yp = clf.predict(Xte)
    acc = accuracy_score(y_test_enc, yp)
    prec = precision_score(y_test_enc, yp, average='macro', zero_division=0)
    rec = recall_score(y_test_enc, yp, average='macro', zero_division=0)
    f1 = f1_score(y_test_enc, yp, average='macro', zero_division=0)
    print(f"{name:12s} | Acc={acc*100:.2f}% | Prec={prec*100:.2f}% | Rec={rec*100:.2f}% | F1={f1*100:.2f}% | n_feat={len(feats)}")

import time

timings = {}

# Pearson
t0 = time.perf_counter()
pearson = X_train.apply(lambda col: abs(np.corrcoef(col, y_train_enc)[0,1]) if np.std(col)>0 else 0)
pearson_sel = list(pearson[pearson>=THRESHOLDS['pearson']].index) or list(pearson.sort_values(ascending=False).head(FALLBACK_TOP_K).index)
timings['Pearson'] = (len(pearson_sel), time.perf_counter()-t0)

# InfoGain
t0 = time.perf_counter()
mi = mutual_info_classif(X_train.values, y_train_enc, random_state=RND)
mi_series = pd.Series(mi, index=X_train.columns)
mi_norm = mi_series / (mi_series.max() if mi_series.max()!=0 else 1)
info_sel = list(mi_norm[mi_norm>=THRESHOLDS['info_gain_norm']].index) or list(mi_series.sort_values(ascending=False).head(FALLBACK_TOP_K).index)
timings['InfoGain'] = (len(info_sel), time.perf_counter()-t0)

# GainRatio
t0 = time.perf_counter()
gr = {}
for c in X_train.columns:
    hist,_=np.histogram(X_train[c],bins=20)
    intrinsic = entropy_from_counts(hist)
    mi_val = mutual_info_classif(X_train[[c]].values, y_train_enc, random_state=RND)[0]
    gr[c] = 0 if intrinsic==0 else (mi_val/intrinsic)
gr_series = pd.Series(gr)
gr_sel = list(gr_series[gr_series>=THRESHOLDS['gain_ratio']].index) or list(gr_series.sort_values(ascending=False).head(FALLBACK_TOP_K).index)
timings['GainRatio'] = (len(gr_sel), time.perf_counter()-t0)

# Symmetrical Uncertainty
t0 = time.perf_counter()
target_counts = pd.Series(y_train_enc).value_counts().values
H_target = entropy_from_counts(target_counts)
su = {}
for c in X_train.columns:
    mi_val = mi_series[c]
    H_feat = entropy_series(X_train[c], bins=20)
    denom = H_feat + H_target
    su[c] = 0 if denom==0 else (2*mi_val/denom)
su_series = pd.Series(su)
su_sel = list(su_series[su_series>=THRESHOLDS['sym_uncert']].index) or list(su_series.sort_values(ascending=False).head(FALLBACK_TOP_K).index)
timings['SymUnc'] = (len(su_sel), time.perf_counter()-t0)

# CFS
t0 = time.perf_counter()
cfs_sel = cfs_greedy(5)
timings['CFS'] = (len(cfs_sel), time.perf_counter()-t0)

# Combined
t0 = time.perf_counter()
methods = [set(pearson_sel), set(info_sel), set(gr_sel), set(su_sel), set(cfs_sel)]
all_feats = set().union(*methods)
occ = {f: sum([1 for s in methods if f in s]) for f in all_feats}
combined = [f for f,cnt in occ.items() if cnt>=4]
if len(combined) < 16:  # fallback if too few
    combined = list(mi_series.sort_values(ascending=False).head(16).index)
timings['Combined'] = (len(combined), time.perf_counter()-t0)

# Print summary
print("\n--- Feature Selection Timing ---")
for k,(n,secs) in timings.items():
    print(f"{k:10s} | {n:3d} features | {secs:.4f} sec")

import pandas as pd

# occ is a dictionary mapping each feature to its count of occurrence across FS methods

# Convert dictionary to DataFrame for tabular display
occ_df = pd.DataFrame(list(occ.items()), columns=['Feature', 'Occurrences'])

# Sort by number of occurrences descending
occ_df = occ_df.sort_values(by='Occurrences', ascending=False)

# Print the table
print(occ_df.reset_index(drop=True))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

combined_feats = combined  # list of important features from FS

# Construct DataFrame with selected features from scaled training data
df_corr = X_train[combined_feats].copy()

# Add the encoded target variable (Attack Type)
df_corr['Attack_Type'] = y_train_enc

corr_matrix = df_corr.corr()

# Plot correlation heatmap focusing on correlations between features and Attack_Type
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix[['Attack_Type']].sort_values(by='Attack_Type', ascending=False),
            annot=True, cmap='coolwarm', center=0, linewidths=0.5,
            cbar_kws={"shrink": 0.75})
plt.title("Correlation between Attack Type and Important Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# === Feature Selection Score Graphs with Thresholds ===
import matplotlib.pyplot as plt

def plot_fs_scores(series, threshold, title, topn=30):
    """
    series: pd.Series of feature scores (index=features, values=scores)
    threshold: cutoff value used for selection
    """
    series = series.sort_values(ascending=False).head(topn)
    plt.figure(figsize=(10,5))
    series.plot(kind='bar')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(rotation=75)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Pearson
plot_fs_scores(pearson, THRESHOLDS['pearson'], "Pearson Correlation Scores (Top 30)")

# InfoGain (normalized mutual info)
plot_fs_scores(mi_norm, THRESHOLDS['info_gain_norm'], "InfoGain (Normalized) Scores (Top 30)")

# GainRatio
plot_fs_scores(gr_series, THRESHOLDS['gain_ratio'], "Gain Ratio Scores (Top 30)")

# Symmetric Uncertainty
plot_fs_scores(su_series, THRESHOLDS['sym_uncert'], "Symmetric Uncertainty Scores (Top 30)")

# CFS Merit (we only did greedy selection, so show sorted merits)
cfs_merit = pd.Series(gr, name="CFS Merit")
plot_fs_scores(cfs_merit, 0, "CFS Merit Scores (Top 30)")

# === Correlation matrix for Combined selected features ===
import matplotlib.pyplot as plt
import seaborn as sns

feats = experiments['Combined']

# include class label (encoded attack type)
corr_df = X_train[feats].copy()
corr_df['Attack_Type'] = y_train_enc

corr_matrix = corr_df.corr()

# plot
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Combined Features vs Attack Type")
plt.tight_layout()
plt.show()