import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIG — sesuaikan path jika perlu
# ============================================================
INPUT_PATH  = 'titanic_raw/train.csv'
OUTPUT_DIR  = 'titanic_preprocessing'
OUTPUT_FILE = f'{OUTPUT_DIR}/train_preprocessed.csv'


# ============================================================
# FUNGSI PREPROCESSING
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f'Data dimuat — Shape: {df.shape}')
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    print(f'Kolom tidak relevan dihapus — Shape: {df.shape}')
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    print(f'Missing values ditangani — Sisa null: {df.isnull().sum().sum()}')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    df['AgeGroup']   = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    )
    print('Feature engineering selesai: FamilySize, IsAlone, AgeGroup')
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df['Sex']      = le.fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df = pd.get_dummies(df, columns=['AgeGroup'],  drop_first=True)
    print(f'Encoding selesai — Shape: {df.shape}')
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f'  Outlier {col}: hapus {before - len(df)} baris')
    print(f'Shape setelah hapus outlier: {df.shape}')
    return df


def normalize_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print('Normalisasi fitur numerik selesai')
    return df


def save_result(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f'Hasil disimpan ke {path} — Shape: {df.shape}')


# ============================================================
# PIPELINE UTAMA
# ============================================================

def run_preprocessing(input_path: str = INPUT_PATH,
                      output_path: str = OUTPUT_FILE) -> pd.DataFrame:
    print('=' * 50)
    print('  MULAI PREPROCESSING TITANIC')
    print('=' * 50)

    df = load_data(input_path)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical(df)
    df = remove_outliers_iqr(df, columns=['Fare', 'FamilySize'])
    df = normalize_features(df, columns=['Age', 'Fare', 'FamilySize'])
    save_result(df, output_path)

    print('=' * 50)
    print('  PREPROCESSING SELESAI!')
    print('=' * 50)
    return df


if __name__ == '__main__':
    run_preprocessing()
