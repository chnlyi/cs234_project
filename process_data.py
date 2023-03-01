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
    return df.iloc[:, -8:]

def process_data(path):
    df = pd.read_csv(path)
    df = df.iloc[:,:63].dropna(subset=['Therapeutic Dose of Warfarin'])
    ids = df['PharmGKB Subject ID']
    dosage = df['Therapeutic Dose of Warfarin']
    labels = dosage.apply(dose_to_label)
    features_df = df.drop(columns=['PharmGKB Subject ID', 'Therapeutic Dose of Warfarin'])
    features_df['Comorbidities'] = LabelEncoder().fit_transform(features_df['Comorbidities'])
    features_df['Medications'] = LabelEncoder().fit_transform(features_df['Medications'])
    categorical_features = [col for col in features_df.columns if features_df[col].dtype=='object']
    numeric_features = [col for col in features_df.columns if features_df[col].dtype!='object']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    features = preprocessor.fit_transform(features_df).toarray()
    feature_names = preprocessor.get_feature_names_out()   
    
    return ids, features, labels, dosage, feature_names
