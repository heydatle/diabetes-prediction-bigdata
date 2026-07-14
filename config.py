CSV_PATH = "data/diabetes_prediction_dataset.csv"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "diabetes"
COLL_NAME = "patients"

# Nguong phan nhom, nguon xem kien_truc.txt muc 10:
#   Tuoi     - ADA/CDC Diabetes Risk Test
#   HbA1c    - ADA Standards of Care in Diabetes
#   Glucose  - ADA Standards of Care in Diabetes (fasting plasma glucose)
#   BMI      - WHO obesity classification
NGUONG_TUOI = (40, 50, 60)
NGUONG_HBA1C = (5.7, 6.5)
NGUONG_GLUCOSE = (100, 126)
NGUONG_BMI = (18.5, 25, 30)
