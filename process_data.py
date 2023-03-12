import math
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from util import dose_to_label

def process_data_clinical_dose(path):
    df = pd.read_csv(path)
    df = df.iloc[:,:63].dropna(subset=['Therapeutic Dose of Warfarin'])
    df['Age in decades'] = df['Age'].fillna('0').apply(lambda x: int(str(x)[0]))
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

def process_data(path):
    df = pd.read_csv(path)
    df = df.iloc[:,:63].dropna(subset=['Therapeutic Dose of Warfarin'])
    ids = df['PharmGKB Subject ID']
    dosage = df['Therapeutic Dose of Warfarin']
    labels = dosage.apply(dose_to_label).values
    features_df = df.drop(columns=['PharmGKB Subject ID', 'Therapeutic Dose of Warfarin'])
    # features_df['Comorbidities'] = LabelEncoder().fit_transform(features_df['Comorbidities'])
    # features_df['Medications'] = LabelEncoder().fit_transform(features_df['Medications'])
    features_df = process_disease(features_df)
    # features_df.to_csv("test.csv", index=False)
    categorical_features = [col for col in features_df.columns if features_df[col].dtype=='object']
    numeric_features = [col for col in features_df.columns if features_df[col].dtype!='object']

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
    feature_names = preprocessor.get_feature_names_out()   
    
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

# this was used to test the data processing functions
# if __name__ == '__main__':
#     path = 'data/warfarin.csv'
#     process_data(path)
