# Eksperimen_SML_Fauzan-Aidil-Luthfi

**Nama:** Fauzan Aidil Luthfi  
**NIM:** apc352d6y0439  
**Dataset:** Titanic - ML from Disaster (Kaggle)

## Cara Download Dataset

1. Buka https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data
2. Download `train.csv`
3. Taruh di folder `titanic_raw/train.csv`


## Tahapan Preprocessing

1. Drop kolom tidak relevan (PassengerId, Name, Ticket, Cabin)
2. Handle missing values (Age → median, Embarked → modus)
3. Feature Engineering (FamilySize, IsAlone, AgeGroup)
4. Encoding kategorikal (Label Encoding + One-Hot)
5. Remove Outlier (IQR Method)
6. Normalisasi (StandardScaler)

## GitHub Actions

Workflow otomatis berjalan ketika:
- Ada push ke `main` branch (khususnya folder `titanic_raw/` atau script preprocessing)
- Dijalankan manual lewat tab **Actions** → **Run workflow**

Hasil preprocessing akan otomatis ter-commit ke repo.
