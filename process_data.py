import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from util import dose_to_label

def remove_colinearity(arr, threshold=1):
    df = pd.DataFrame(arr)
    cor_matrix = df.corr().abs()
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    lst = np.argwhere(upper.values>=1)
    to_drop = set(l[1] for l in lst)
    out = df.drop(columns=to_drop).values
    return out

def process_data_clinical_dose(path):
    df = pd.read_csv(path)
    df = df.iloc[:,:63].dropna(subset=['Therapeutic Dose of Warfarin']).reset_index(drop=True)
    df['Age in decades'] = df['Age'].fillna('7').apply(lambda x: int(str(x)[0])) # 70-79 biggest patient age bucket
    df['Height in cm'] = df['Height (cm)'].fillna(df['Height (cm)'].median())
    df['Weight in kg'] = df['Weight (kg)'].fillna(df['Weight (kg)'].median())
    df['Asian Race'] = (df['Race']=='Asian').astype(float)
    df['Black or African American'] = (df['Race']=='Black or African American').astype(float)
    df['Missing or Mixed Race'] = (df['Race']=='Unknown').astype(float)
    df['Enzyme Inducer Status'] = ((df['Carbamazepine (Tegretol)']==1)|(df['Phenytoin (Dilantin)']==1)|(df['Rifampin or Rifampicin']==1)).astype(float)
    df['Amiodarone Status'] = (df['Amiodarone (Cordarone)']==1).astype(float)
    features, cols = df.iloc[:, -8:].values, df.iloc[:, -8:].columns.tolist()
    ids = df['PharmGKB Subject ID']
    dosage = df['Therapeutic Dose of Warfarin']
    labels = dosage.apply(dose_to_label).values
    return features, cols, labels

def process_data_pharmacogenetic_dose(path):
    df = pd.read_csv(path)
    df = df.iloc[:,:63].dropna(subset=['Therapeutic Dose of Warfarin']).reset_index(drop=True)
    df['Age in decades'] = df['Age'].fillna('7').apply(lambda x: int(str(x)[0])) # 70-79 biggest patient age bucket
    df['Height in cm'] = df['Height (cm)'].fillna(df['Height (cm)'].median())
    df['Weight in kg'] = df['Weight (kg)'].fillna(df['Weight (kg)'].median())
    df['Asian Race'] = (df['Race']=='Asian').astype(float)
    df['Black or African American'] = (df['Race']=='Black or African American').astype(float)
    df['Missing or Mixed Race'] = (df['Race']=='Unknown').astype(float)
    df['Enzyme Inducer Status'] = ((df['Carbamazepine (Tegretol)']==1)|(df['Phenytoin (Dilantin)']==1)|(df['Rifampin or Rifampicin']==1)).astype(float)
    df['Amiodarone Status'] = (df['Amiodarone (Cordarone)']==1).astype(float)
    
    df['VKORC1 A/G'] = (df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']=='A/G').astype(float)
    df['VKORC1 A/A'] = (df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']=='A/A').astype(float)
    df['VKORC1 Unknown'] = (df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'].isna()).astype(float)
    
    df['CYP2C9 *1/*2'] = (df['Cyp2C9 genotypes']=='*1/*2').astype(float)
    df['CYP2C9 *1/*3'] = (df['Cyp2C9 genotypes']=='*1/*3').astype(float)
    df['CYP2C9 *2/*2'] = (df['Cyp2C9 genotypes']=='*2/*2').astype(float)
    df['CYP2C9 *2/*3'] = (df['Cyp2C9 genotypes']=='*2/*3').astype(float)
    df['CYP2C9 *3/*3'] = (df['Cyp2C9 genotypes']=='*3/*3').astype(float)
    df['CYP2C9 Unknown'] = (df['Cyp2C9 genotypes'].isna()).astype(float)
    
    features, cols = df.iloc[:, -17:].values, df.iloc[:, -17:].columns.tolist()
    ids = df['PharmGKB Subject ID']
    dosage = df['Therapeutic Dose of Warfarin']
    labels = dosage.apply(dose_to_label).values
    return features, cols, labels


def process_data(path, threshold=1):
    df = pd.read_csv(path)
    df = df.iloc[:,:63].dropna(subset=['Therapeutic Dose of Warfarin']).reset_index(drop=True)
    df['Age in decades'] = df['Age'].fillna('7').apply(lambda x: int(str(x)[0])) # 70-79 biggest patient age bucket
    ids = df['PharmGKB Subject ID']
    dosage = df['Therapeutic Dose of Warfarin']
    labels = dosage.apply(dose_to_label).values
    features_df = df.drop(columns=['PharmGKB Subject ID', 'Therapeutic Dose of Warfarin', 'Age'])
    
    numeric_features = ['Age in decades', 'Height (cm)', 'Weight (kg)', 'INR on Reported Therapeutic Dose of Warfarin']    
    for col in features_df.columns:
        if col not in numeric_features:
            features_df[col]= features_df[col].astype('str')
    
    features_df = process_disease(features_df)
    process_trtmnt(features_df)
    categorical_features = [col for col in features_df.columns if features_df[col].dtype=='object']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    features = preprocessor.fit_transform(features_df)
    
    features = remove_colinearity(features, threshold)
    
    return features, labels

def process_disease(features_df):
    df = features_df
    d = df["Comorbidities"]
    d_list = d.tolist()
    list_of_lists = []
    for element in d_list:
        if isinstance(element, str):
            names = element.strip("'").split("; ")
            list_of_lists.append(names)
        else:
            list_of_lists.append([element])

    my_list_of_lists = [["none"] if all(isinstance(elem, float) and math.isnan(elem) for elem in sublist) else sublist for sublist in list_of_lists]
    d_list_of_lists = [["none"] if not sublist else [elem for elem in sublist if not isinstance(elem, str) or not elem.startswith("No ")] or ["none"] for sublist in my_list_of_lists]
    formatted_list = [[name.strip().lower().replace('-', '') for name in sublist] for sublist in d_list_of_lists]

    # manually merge similar diseases
    for i in range(len(formatted_list)):
        for j in range(len(formatted_list[i])):
            if "cancer" in formatted_list[i][j]:
                formatted_list[i][j] = "cancer"
            if "diabetes" in formatted_list[i][j]:
                formatted_list[i][j] = "diabetes"
            if "fibri" in formatted_list[i][j]:
                formatted_list[i][j] = "a fib / flutter"
            if "flutter" in formatted_list[i][j]:
                formatted_list[i][j] = "a fib / flutter"
            if "heart failure" in formatted_list[i][j]:
                formatted_list[i][j] = "heart failure"
            if "dyslipidemia" in formatted_list[i][j]:
                formatted_list[i][j] = "hyperlipidemia"
            if "high cholesterol" in formatted_list[i][j]:
                formatted_list[i][j] = "hyperlipidemia"
            if "arrythmia" in formatted_list[i][j]:
                formatted_list[i][j] = "cardiac arrhythmia"
            if "valve" in formatted_list[i][j]:
                formatted_list[i][j] = "valve repair"
            if "hypertension" in formatted_list[i][j]:
                formatted_list[i][j] = "hypertension"
            if "malignancy" in formatted_list[i][j]:
                formatted_list[i][j] = "cancer"
            if "coronary artery" in formatted_list[i][j]:
                formatted_list[i][j] = "coronary artery disease / bypass"
            if "angioplasty" in formatted_list[i][j]:
                formatted_list[i][j] = "coronary artery disease / bypass"

    disease_codes = {"a fib / flutter": 1, "hypertension": 2, "valve repair": 3, "diabetes": 4,
    "coronary artery disease / bypass": 5, "heart failure": 6, "hyperlipidemia":7, "cardiac arrhythmia":8,
    "cancer":9}

    for i in range(len(formatted_list)):
        for j in range(len(formatted_list[i])):
            if formatted_list[i][j] in disease_codes:
                formatted_list[i][j] = disease_codes[formatted_list[i][j]]
            else:
                formatted_list[i][j] = None

        formatted_list[i] = [0 if disease is None else disease for disease in formatted_list[i]]
    for i in range(len(formatted_list)):
        num_zeros = formatted_list[i].count(0)
        if len(formatted_list[i]) == 1 and formatted_list[i][0] == 0:
            continue
        else:
            formatted_list[i] = [x for x in formatted_list[i] if x != 0]

    for i in range(len(formatted_list)):
        if len(formatted_list[i]) == 0:
            formatted_list[i] = [0]
        disease_set = set(formatted_list[i])
        sorted_diseases = sorted(list(disease_set))
        formatted_list[i] = sorted_diseases

    col_names = [str(i) for i in range(1, 10)]
    data_dict = {col_name: [] for col_name in col_names}
    for sublist in formatted_list:
        for i in range(1, 10):
            if i in sublist:
                data_dict[str(i)].append(1)
            else:
                data_dict[str(i)].append(0)

    new_df = pd.DataFrame(data_dict)
    new_df.columns = list((disease_codes.keys()))
    out_df = pd.concat([df, new_df], axis=1)
    out_df = out_df.drop(columns=['Comorbidities', 'Medications'])
    return out_df

def process_trtmnt(features_df):    
    trtmnt_col = ['DVT','PE','Afib/flutter','Heart Valve','Cardiomyopathy/LV Dilation','Stroke','Post-Orthopedic','Other','NA']
    trtmnt_col_map = dict(zip(trtmnt_col,range(1, 10))) 
    trtmnt = features_df['Indication for Warfarin Treatment']
    features_df['Indication for Warfarin Treatment'] = features_df['Indication for Warfarin Treatment'].fillna('9')
    features_df['Indication for Warfarin Treatment'] = features_df['Indication for Warfarin Treatment'].apply(lambda x: x.replace('or', ';').replace(' ',''))
    features_df['Indication for Warfarin Treatment'] = features_df['Indication for Warfarin Treatment'].apply(lambda x: x.split(';'))
    def assign(col):
        features_df[col] = features_df['Indication for Warfarin Treatment'].apply(lambda x: 1 if str(trtmnt_col_map[col]) in x else 0)
    for col in trtmnt_col:
        assign(col)
    features_df.drop(columns='Indication for Warfarin Treatment', inplace=True)

# this was used to test the data processing functions
# if __name__ == '__main__':
#     path = 'data/warfarin.csv'
#     process_data(path)
