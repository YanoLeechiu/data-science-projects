import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# -- Load data -----------------------------------------------------------------
heart_df = pd.read_csv('Myocardial_infarction_complications_Database.csv')

# -- Rename columns to readable names ------------------------------------------
rename_map = {
    'AGE':          'Age',
    'SEX':          'Sex',
    'INF_ANAM':     'Prior MI History',
    'STENOK_AN':    'Angina History',
    'FK_STENOK':    'Angina Severity',
    'IBS_POST':     'Coronary Artery Disease',
    'GB':           'Hypertension',
    'SIM_GIPERT':   'Symptomatic Hypertension',
    'DLIT_AG':      'Hypertension Duration (yrs)',
    'ZSN_A':        'Prior Heart Failure',
    'S_AD_KBRIG':   'Systolic BP (Ambulance)',
    'D_AD_KBRIG':   'Diastolic BP (Ambulance)',
    'S_AD_ORIT':    'Systolic BP (ICU)',
    'D_AD_ORIT':    'Diastolic BP (ICU)',
    'O_L_POST':     'Pulm Edema on Admission',
    'K_SH_POST':    'Cardiogenic Shock on Admission',
    'ant_im':       'Anterior MI',
    'lat_im':       'Lateral MI',
    'inf_im':       'Inferior MI',
    'post_im':      'Posterior MI',
    'GIPO_K':       'Low Potassium',
    'K_BLOOD':      'Blood Potassium',
    'GIPER_NA':     'High Sodium',
    'NA_BLOOD':     'Blood Sodium',
    'ALT_BLOOD':    'ALT Enzyme',
    'AST_BLOOD':    'AST Enzyme',
    'KFK_BLOOD':    'CK Enzyme',
    'L_BLOOD':      'White Blood Cell Count',
    'ROE':          'ESR (Inflammation)',
    'TIME_B_S':     'Time to Treatment (min)',
    'IBS_NASL':     'Hereditary Heart Disease',
    'NITR_S':       'Nitrates Given',
    'B_BLOK_S_n':   'Beta Blockers Given',
    'ANT_CA_S_n':   'Calcium Blockers Given',
    'GEPAR_S_n':    'Heparin Given',
    'ASP_S_n':      'Aspirin Given',
    'FIBR_PREDS':   'Atrial Fibrillation',
    'PREDS_TAH':    'Supraventricular Tachycardia',
    'JELUD_TAH':    'Ventricular Tachycardia',
    'FIBR_JELUD':   'Ventricular Fibrillation',
    'A_V_BLOK':     'AV Block',
    'OTEK_LANC':    'Pulmonary Edema',
    'RAZRIV':       'Heart Rupture',
    'DRESSLER':     'Dressler Syndrome',
    'ZSN':          'Heart Failure',
    'REC_IM':       'Recurrent MI',
    'P_IM_STEN':    'Post-MI Angina',
    'LET_IS':       'In-Hospital Mortality',
}
heart_df.rename(columns=rename_map, inplace=True)

# -- Drop high-missing and other target columns --------------------------------
drop_cols = ['ID', 'CK Enzyme', 'Hereditary Heart Disease',
             'Systolic BP (Ambulance)', 'Diastolic BP (Ambulance)',
             'Atrial Fibrillation', 'Supraventricular Tachycardia',
             'Ventricular Tachycardia', 'Ventricular Fibrillation',
             'AV Block', 'Pulmonary Edema', 'Heart Rupture',
             'Dressler Syndrome', 'Recurrent MI', 'Post-MI Angina',
             'In-Hospital Mortality']

# -- Save complication columns for chart 9 before they are dropped -------------
comp_cols = ['Atrial Fibrillation', 'Ventricular Tachycardia', 'Ventricular Fibrillation',
             'AV Block', 'Pulmonary Edema', 'Heart Rupture', 'Dressler Syndrome', 'Recurrent MI']
comp_df = heart_df[comp_cols + ['Heart Failure']].copy()

X = heart_df.drop(columns=drop_cols + ['Heart Failure', 'Age Group',
    'ID', 'KFK_BLOOD', 'IBS_NASL', 'S_AD_KBRIG', 'D_AD_KBRIG',
    'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK',
    'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'REC_IM', 'P_IM_STEN', 'LET_IS', 'ZSN'],
    errors='ignore').select_dtypes(include='number')
y = heart_df['Heart Failure']

# -- Data description ----------------------------------------------------------
key_cols = ['Age', 'Systolic BP (ICU)', 'Diastolic BP (ICU)',
            'Blood Potassium', 'Blood Sodium', 'White Blood Cell Count',
            'ESR (Inflammation)', 'Time to Treatment (min)',
            'ALT Enzyme', 'AST Enzyme']

print("--- Key Variable Summary ---")
print(heart_df[key_cols].describe().round(2).to_string())

# -- Impute missing values and split -------------------------------------------
numerical_cols   = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

imputer_numerical   = SimpleImputer(strategy='median')
imputer_categorical = SimpleImputer(strategy='most_frequent')

X_imputed_numerical = pd.DataFrame(imputer_numerical.fit_transform(X[numerical_cols]),
                                    columns=numerical_cols, index=X.index)
if not categorical_cols.empty:
    X_imputed_categorical = pd.DataFrame(imputer_categorical.fit_transform(X[categorical_cols]),
                                          columns=categorical_cols, index=X.index)
    X_imp = pd.concat([X_imputed_numerical, X_imputed_categorical], axis=1)
    X_imp = X_imp[X.columns]
else:
    X_imp = X_imputed_numerical

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y)

# -- Age group column (numeric bins to avoid string conversion errors) ---------
heart_df['Age Group'] = pd.cut(heart_df['Age'], bins=[20, 40, 50, 60, 70, 80, 100],
                                labels=[1, 2, 3, 4, 5, 6])

# -- Define models -------------------------------------------------------------
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

# -- Train and evaluate all models ---------------------------------------------
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'roc_auc':  round(roc_auc_score(y_test, y_proba), 3),
        'cm':       confusion_matrix(y_test, y_pred)
    }
    print(f"\n{name}")
    print(f"  Accuracy : {results[name]['accuracy']}")
    print(f"  ROC-AUC  : {results[name]['roc_auc']}")
    print(f"  Confusion matrix:\n{results[name]['cm']}")

# -- Feature importances from Random Forest ------------------------------------
feat_imp = pd.Series(
    models['Random Forest'].feature_importances_, index=X.columns
).sort_values(ascending=True).tail(15)

# == GRAPHS ====================================================================

names    = list(results.keys())
accuracy = [results[n]['accuracy'] for n in names]
roc_auc  = [results[n]['roc_auc']  for n in names]
colors   = ['steelblue', 'tomato', 'seagreen']

# 1. Accuracy comparison
plt.figure()
plt.bar(names, accuracy, color=colors)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.01, str(v), ha='center')
plt.tight_layout()
plt.savefig('1_accuracy_comparison.png')
plt.show()

# 2. ROC-AUC comparison
plt.figure()
plt.bar(names, roc_auc, color=colors)
plt.title('Model ROC-AUC Comparison')
plt.ylabel('ROC-AUC')
plt.ylim(0, 1)
for i, v in enumerate(roc_auc):
    plt.text(i, v + 0.01, str(v), ha='center')
plt.tight_layout()
plt.savefig('2_roc_auc_comparison.png')
plt.show()

# 3. Feature importances (Random Forest)
plt.figure(figsize=(8, 6))
plt.barh(feat_imp.index, feat_imp.values, color='steelblue')
plt.title('Top 15 Most Important Variables (Random Forest)')
plt.xlabel('Importance score')
plt.tight_layout()
plt.savefig('3_feature_importance.png')
plt.show()

# 4. Overall Heart Failure distribution
counts = y.value_counts().sort_index()
plt.figure()
plt.bar(['No Heart Failure', 'Heart Failure'], counts, color=['steelblue', 'tomato'])
plt.title('Heart Failure Distribution')
plt.ylabel('Number of patients')
for i, v in enumerate(counts):
    plt.text(i, v + 10, str(v), ha='center')
plt.tight_layout()
plt.savefig('4_hf_distribution.png')
plt.show()

# 5. Heart Failure rate by age group
age_rate = heart_df.groupby('Age Group', observed=True)['Heart Failure'].mean() * 100
age_labels = ['20-40', '41-50', '51-60', '61-70', '71-80', '80+']
plt.figure()
plt.bar(age_labels, age_rate.values, color='steelblue')
plt.title('Heart Failure Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Heart Failure Rate (%)')
for i, v in enumerate(age_rate):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('5_hf_by_age.png')
plt.show()

# 6. Heart Failure rate by sex
sex_rate = heart_df.groupby('Sex')['Heart Failure'].mean() * 100
plt.figure()
plt.bar(['Female', 'Male'], sex_rate.values, color=['tomato', 'steelblue'])
plt.title('Heart Failure Rate by Sex')
plt.ylabel('Heart Failure Rate (%)')
for i, v in enumerate(sex_rate):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('6_hf_by_sex.png')
plt.show()

# 7. Heart Failure rate by prior HF history
prior_rate = heart_df.groupby(heart_df['Prior Heart Failure'] > 0)['Heart Failure'].mean() * 100
plt.figure()
plt.bar(['No Prior Heart Failure', 'Prior Heart Failure'], prior_rate.values, color=['steelblue', 'tomato'])
plt.title('Heart Failure Rate by Prior Heart Failure History')
plt.ylabel('Heart Failure Rate (%)')
for i, v in enumerate(prior_rate):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('7_hf_by_prior_history.png')
plt.show()

# 8. Systolic BP at ICU by Heart Failure status
bp_no  = heart_df[heart_df['Heart Failure'] == 0]['Systolic BP (ICU)'].dropna()
bp_yes = heart_df[heart_df['Heart Failure'] == 1]['Systolic BP (ICU)'].dropna()
plt.figure()
plt.hist(bp_no,  bins=20, alpha=0.6, color='steelblue', label='No Heart Failure')
plt.hist(bp_yes, bins=20, alpha=0.6, color='tomato',    label='Heart Failure')
plt.title('Systolic BP (ICU) by Heart Failure Status')
plt.xlabel('Systolic BP (mmHg)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('8_bp_distribution.png')
plt.show()

# 9. Complication rates: Heart Failure vs no Heart Failure
no_hf_rates  = comp_df[comp_df['Heart Failure'] == 0][comp_cols].mean() * 100
yes_hf_rates = comp_df[comp_df['Heart Failure'] == 1][comp_cols].mean() * 100
x = np.arange(len(comp_cols))
w = 0.35
plt.figure(figsize=(11, 5))
plt.bar(x - w/2, no_hf_rates,  w, label='No Heart Failure', color='steelblue')
plt.bar(x + w/2, yes_hf_rates, w, label='Heart Failure',    color='tomato')
plt.xticks(x, comp_cols, rotation=30, ha='right', fontsize=9)
plt.ylabel('Rate (%)')
plt.title('Complication Rates: Heart Failure vs No Heart Failure')
plt.legend()
plt.tight_layout()
plt.savefig('9_complications.png')
plt.show()

# 10. Confusion matrices for all 3 models
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, name in zip(axes, names):
    cm = results[name]['cm']
    ax.imshow(cm, cmap='Blues')
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No HF', 'HF'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No HF', 'HF'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.suptitle('Confusion Matrices')
plt.tight_layout()
plt.savefig('10_confusion_matrices.png')
plt.show()

# 11. Correlation heatmap
heatmap_cols = [
    'Age', 'Prior Heart Failure', 'Systolic BP (ICU)', 'Diastolic BP (ICU)',
    'White Blood Cell Count', 'ESR (Inflammation)', 'Blood Potassium',
    'Blood Sodium', 'AST Enzyme', 'ALT Enzyme', 'Time to Treatment (min)',
    'Angina History', 'Hypertension', 'Anterior MI', 'Inferior MI',
    'Heart Failure'
]
plt.figure(figsize=(12, 10))
corr = heart_df[heatmap_cols].corr()
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(heatmap_cols)), heatmap_cols, rotation=45, ha='right', fontsize=9)
plt.yticks(range(len(heatmap_cols)), heatmap_cols, fontsize=9)
plt.title('Correlation Heatmap of Key Variables')
plt.tight_layout()
plt.savefig('11_heatmap.png')
plt.show()
