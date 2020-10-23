# Data Handling
import logging
import pickle
import numpy as np
from pydantic import BaseModel

# Server
import uvicorn
from fastapi import FastAPI, Body

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Initialize files
logit = pickle.load(open('data/logit.pickle', 'rb'))
knn = pickle.load(open('data/knn.pickle', 'rb'))
svc_linear = pickle.load(open('data/svc_linear.pickle', 'rb'))
svc_rbf = pickle.load(open('data/svc_rbf.pickle', 'rb'))
gaussian = pickle.load(open('data/gaussian.pickle', 'rb'))
decision_tree = pickle.load(open('data/decision_tree.pickle', 'rb'))
rf = pickle.load(open('data/rf.pickle', 'rb'))

encoderLabel = pickle.load(open('data/encoderLabel.pickle', 'rb'))
features = pickle.load(open('data/features.pickle', 'rb'))


class Data(BaseModel):
    radius: float
    texture: float
    perimeter: float
    area: float
    smoothness: float
    compactness: float
    concavity: float
    concave_points: float
    symmetry: float
    fractal_dimension: float
    radiusSE: float
    textureSE: float
    perimeterSE: float
    areaSE: float
    smoothnessSE: float
    compactnessSE: float
    concavitySE: float
    concave_pointsSE: float
    symmetrySE: float
    fractal_dimensionSE: float
    radiusW: float
    textureW: float
    perimeterW: float
    areaW: float
    smoothnessW: float
    compactnessW: float
    concavityW: float
    concave_pointsW: float
    symmetryW: float
    fractal_dimensionW: float
    
        
@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = np.array([data_dict[feature] for feature in features])
        print(to_predict.reshape(1, -1).shape)

        # Create and return prediction
        prediction_logit = logit.predict(to_predict.reshape(1, -1))
        prediction_knn = knn.predict(to_predict.reshape(1, -1))
        prediction_svcLinear = svc_linear.predict(to_predict.reshape(1, -1))
        prediction_svcRBF = svc_rbf.predict(to_predict.reshape(1, -1))
        prediction_gaussian = gaussian.predict(to_predict.reshape(1, -1))
        prediction_decisionTree = decision_tree.predict(to_predict.reshape(1, -1))
        prediction_rf = rf.predict(to_predict.reshape(1, -1))

        logit_proba = max(logit.predict_proba(to_predict.reshape(1, -1))[0])
        knn_proba = max(knn.predict_proba(to_predict.reshape(1, -1))[0])
        gaussian_proba = max(gaussian.predict_proba(to_predict.reshape(1, -1))[0])
        tree_proba = max(decision_tree.predict_proba(to_predict.reshape(1, -1))[0])
        rf_proba = max(rf.predict_proba(to_predict.reshape(1, -1))[0])
        
        prediction = {
            'Logistic Regression': "{} ({:.2f}%)".format(encoderLabel.inverse_transform(prediction_logit)[0], logit_proba*100),
            'Nearest Neighbor': "{} ({:.2f}%)".format(encoderLabel.inverse_transform(prediction_knn)[0], knn_proba*100),
            'Support Vector Machines': "{}".format(encoderLabel.inverse_transform(prediction_svcLinear)[0]),
            'RBF Kernel SVM': "{}".format(encoderLabel.inverse_transform(prediction_svcRBF)[0]),
            'Naive Bayes': "{} ({:.2f}%)".format(encoderLabel.inverse_transform(prediction_gaussian)[0], gaussian_proba*100),
            'Decision Tree Algorithm': "{} ({:.2f}%)".format(encoderLabel.inverse_transform(prediction_decisionTree)[0], tree_proba*100),
            'Random Forest Classification': "{} ({:.2f}%)".format(encoderLabel.inverse_transform(prediction_rf)[0], rf_proba*100),
        }
        return prediction
    
    except:
        my_logger.error("Something went wrong!")
        prediction_error = {
            'Logistic Regression': 'error',
            'Nearest Neighbor': 'error',
            'Support Vector Machines': 'error',
            'RBF Kernel SVM': 'error',
            'Naïve Bayes': 'error',
            'Decision Tree Algorithm': 'error',
            'Random Forest Classification': 'error',
        }
        return prediction_error

@app.get("/example")
async def input_example():
    example = {
        "radius": 7.760,
        "texture": 24.54,
        "perimeter": 47.92,
        "area": 181.0,
        "smoothness": 0.05263,
        "compactness": 0.04362,
        "concavity": 0.0,
        "concave_points": 0.0,
        "symmetry": 0.1587,
        "fractal_dimension": 0.05884,
        "radiusSE": 0.3857,
        "textureSE": 1.428,
        "perimeterSE": 2.548,
        "areaSE": 19.15,
        "smoothnessSE": 0.007189,
        "compactnessSE": 0.004660,
        "concavitySE": 0.0,
        "concave_pointsSE": 0.0,
        "symmetrySE": 0.02676,
        "fractal_dimensionSE": 0.002783,
        "radiusW": 9.456,
        "textureW": 30.37,
        "perimeterW": 59.16,
        "areaW": 268.6,
        "smoothnessW": 0.08996,
        "compactnessW": 0.06444,
        "concavityW": 0.0,
        "concave_pointsW": 0.0,
        "symmetryW": 0.2871,
        "fractal_dimensionW": 0.07039,
    }
    return example