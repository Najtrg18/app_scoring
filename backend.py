##########################################################
# to run locally: FLASK_APP=backend.py flask run, with app.run(debug=True) at the end
# to run on AWS cloud : python3 backend.py, with app.run(debug=True, host="0.0.0.0", port=5000) at the end
##########################################################

from flask import Flask, jsonify
import json
from json import JSONEncoder
import pandas as pd
import numpy as np
import pickle
import shap

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Config de l'app
app = Flask(__name__)

PATH_D="Dataset/"
PATH_O="Objects/"

#Chargement des données 
df_kernel = pd.read_csv(PATH_D+'df_kernel_reduced.csv')
df_train = pd.read_csv(PATH_D+'df_train_reduced.csv')

#Renommage des features pour meilleure comprehension
info_cols = {
            'SK_ID_CURR': "SK_ID_CURR",
            'TARGET': "TARGET",
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT_FAMILIAL", 
            'CNT_CHILDREN': "NB_ENFANTS",
            'FLAG_OWN_CAR': "PROPRIETAIRE_VEHICULE",
            'FLAG_OWN_REALTY': "PROPRIETAIRE_IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU_EDUCATION",
            'OCCUPATION_TYPE': "OCCUPATION", 
            'DAYS_EMPLOYED': "NB_ANNEES_EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT_CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE_CONTRAT",
            'AMT_ANNUITY': "MONTANT_ANNUITES",
            'NAME_INCOME_TYPE': "TYPE_REVENUS",
            'EXT_SOURCE_1': "SCORE_SOURCE_1",
            'EXT_SOURCE_2': "SCORE_SOURCE_2",
            'EXT_SOURCE_3': "SCORE_SOURCE_3",
            'INSTAL_DPD_MEAN' : "MOY_DELAI_PAIEMENT",
            'PAYMENT_RATE' : "TAUX_PAIEMENT",
            'INSTAL_AMT_INSTALMENT_MEAN' : "DUREE_MOYENNE_CREDIT",
            'OWN_CAR_AGE' : "AGE_VEHICULE",
            'APPROVED_CNT_PAYMENT_MEAN' : "MOYENS_PAIEMENT",
            'ANNUITY_INCOME_PERC' : "PERC_ANNUITE_REVENU"
                }
relevant_train = [col for col in df_train.columns if col in info_cols.keys()]
df_train = df_train[relevant_train]

df_kernel.rename(columns=info_cols, inplace=True)
df_train.rename(columns=info_cols, inplace=True)

#Correction de quelques features de df_train
df_train["AGE"] = df_train["AGE"]/-365
df_train['NB_ANNEES_EMPLOI'] = df_train['NB_ANNEES_EMPLOI'].replace(365243,np.nan)
df_train["NB_ANNEES_EMPLOI"] = df_train["NB_ANNEES_EMPLOI"]/-365
df_train = df_train[df_train["REVENUS"]<1e8]
df_train['GENRE'] = df_train['GENRE'].replace("XNA",np.nan)
df_train['STATUT_FAMILIAL'] = df_train['STATUT_FAMILIAL'].replace("Unknown",np.nan)

#Liste de tous les clients
all_id_client = df_kernel['SK_ID_CURR'].unique()

#Chargement du modèle et seuil
model = pickle.load(open(PATH_O+'best_model.pkl', 'rb'))
seuil = pickle.load(open(PATH_O+'best_threshold.pkl', 'rb'))

#Definition de la classe pour convertir en json des objets plus complexes
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
@app.route('/')
def home():
    sentence = "Bonjour, prêt pour la prédiction de l'acceptation ou non d'un prêt?"
    return sentence

#Fonction qui nous donne la liste de tous les IDs clients
@app.route('/give_ids')
def give_ids():
    arrayIDs = {"array": all_id_client}
    return json.dumps(arrayIDs, cls=NumpyArrayEncoder)
    
#Fonction qui nous donne les infos d'un client donne
@app.route('/get_info/<id_client>')
def get_info(id_client):
    client_info = df_train[df_train['SK_ID_CURR']==int(id_client)]
    return client_info.to_json(orient='records')
    
#Fonction qui nous donne les infos d'un client donne
@app.route('/compare/<feature>')
def table(feature):
    feature_info = df_train[['TARGET', feature]]
    return feature_info.to_json(orient='records')
    
#Fonction qui nous predit pour un client donne, l'accord de pret ou non et son scoring
@app.route('/predict/<id_client>')
def predict(id_client):
    
    ID = int(id_client)
    
    X = df_kernel[df_kernel['SK_ID_CURR'] == int(id_client)]   
    ignore_features = ['TARGET','SK_ID_CURR','PREV_APP_CREDIT_PERC_MAX', 'REFUSED_APP_CREDIT_PERC_MAX', 'INSTAL_PAYMENT_PERC_MAX']
    relevant_features = [col for col in df_kernel.columns if col not in ignore_features]
    X = X[relevant_features]
    
    probability_default_payment = model.predict_proba(X)[:, 1]
    if probability_default_payment >= seuil:
        prediction = "Prêt non accepté"
    else:
        prediction = "Prêt accepté"
    print(prediction)
    
    client_scoring = {
        'prediction' : str(prediction),
        'proba' : float(probability_default_payment),
        'seuil' : float(seuil)
    }
    
    return jsonify(client_scoring)

#Fonction qui nous fournit pour un client donne, l'interpretation de son scoring
@app.route('/interpret/<id_client>')
def interpret(id_client):
    
    
    list_out = ['TARGET','PREV_APP_CREDIT_PERC_MAX', 'REFUSED_APP_CREDIT_PERC_MAX', 'INSTAL_PAYMENT_PERC_MAX']
    X_shap = df_kernel.drop(columns = list_out)
    index_client = X_shap[X_shap['SK_ID_CURR'] == int(id_client)].index
    
    explainer = shap.Explainer(model, X_shap)
    shap_values = explainer(X_shap, check_additivity=False)
    
    shap_front = pd.DataFrame()
    shap_front['index'] = X_shap.columns
    shap_front['shap'] = shap_values[index_client[0]].values
    shap_front['shap_abs'] = shap_front['shap'].abs()
    shap_front = shap_front.sort_values('shap_abs', ascending=False)
    shap_front = shap_front.head(10)
    shap_front.drop('shap_abs', axis= 1 , inplace= True)
      
    return shap_front.to_json(orient='records')

#lancement de l'application
if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8080)