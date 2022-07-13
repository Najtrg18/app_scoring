##########################################################
# to run: streamlit run frontend.py
##########################################################

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from urllib.request import urlopen
import json
import requests
import shap
from lightgbm import LGBMClassifier
import time # pour temps reel
import plotly.express as px # interactivite
import matplotlib.pyplot as plt
import seaborn as sns
import math
from io import BytesIO
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


#Chargement des données 
data = pd.read_csv('df_train_reduced.csv')
df = pd.read_csv('df_kernel_reduced.csv')

#Chargement du logo
LOGO_IMAGE = "logo.png"

#Renommage des variables pour pour des plus parlant
personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL", 
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "PROPRIETAIRE VEHICULE",
            'FLAG_OWN_REALTY': "PROPRIETAIRE BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "OCCUPATION", 
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS"

                }
loan_info_cols = {
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "SCORE SOURCE 1",
            'EXT_SOURCE_2': "SCORE SOURCE 2",
            'EXT_SOURCE_3': "SCORE SOURCE 3"

             }

info_cols_vis = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL", 
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "PROPRIETAIRE VEHICULE",
            'FLAG_OWN_REALTY': "PROPRIETAIRE BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "OCCUPATION", 
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "SCORE SOURCE 1",
            'EXT_SOURCE_2': "SCORE SOURCE 2",
            'EXT_SOURCE_3': "SCORE SOURCE 3"
}

info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL", 
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "PROPRIETAIRE VEHICULE",
            'FLAG_OWN_REALTY': "PROPRIETAIRE IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "OCCUPATION", 
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "SCORE SOURCE 1",
            'EXT_SOURCE_2': "SCORE SOURCE 2",
            'EXT_SOURCE_3': "SCORE SOURCE 3",
            'INSTAL_DPD_MEAN' : "MOY DELAI PAIEMENT",
            'PAYMENT_RATE' : "TAUX PAIEMENT",
            'INSTAL_AMT_INSTALMENT_MEAN' : "DUREE MOYENNE CREDIT",
            'OWN_CAR_AGE' : "AGE VEHICULE",
            'APPROVED_CNT_PAYMENT_MEAN' : "MOYENS PAIEMENT",
            'ANNUITY_INCOME_PERC' : "% ANNUITE REVENU"
                }

df.rename(columns=info_cols, inplace=True)

numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

#@st.cache
def get_result(url):
    json_url = urlopen(url)
    return json.loads(json_url.read())
    
#@st.cache
def compare_numerical(df, feature, client_feature_val):

    fig, ax = plt.subplots(1, 1, figsize = (10, 5), dpi=300)
    #fig = plt.figure(figsize = (10, 5))
    df = df[df.AMT_INCOME_TOTAL<1e8]
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243,np.nan)
    df['DAYS_BIRTH'] = df['DAYS_BIRTH']/-365
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED']/365
    df0 = df.loc[data['TARGET'] == 0]
    df1 = df.loc[data['TARGET'] == 1]
    
    sns.kdeplot(df0[feature].dropna(), label = 'Bon client', color='g')
    sns.kdeplot(df1[feature].dropna(), label = 'Mauvais client', color='r')
    
    if feature == "DAYS_BIRTH":
        client_feature_val = client_feature_val/-365
    elif feature == "DAYS_EMPLOYED":
        client_feature_val = client_feature_val/365
    plt.axvline(float(client_feature_val), color="black", linestyle='--', label = 'Client')
    
    #plt.title(title, fontsize='20', fontweight='bold')
    ax.set_ylabel('')    
    ax.set_xlabel('')
    plt.legend()
    #plt.show()  
    st.pyplot(fig)
    
#@st.cache
def compare_categorical(df,feature,client_feature_val):
                               
    #fig, ax = plt.subplots(1, 1, figsize = (10, 5), dpi=300)
    fig = plt.figure(figsize = (10, 5))
    
    df['CODE_GENDER'] = df['CODE_GENDER'].replace("XNA",np.nan)
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].replace("Unknown",np.nan)
    
    categories = df[feature].unique()
    categories = list(categories)
       
    cat_perc = (df[[feature,'TARGET']].groupby([feature],as_index=False).mean())
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    pos_client = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
    
    sns.countplot(x = feature, 
                            data=df,
                            hue ="TARGET",
                            order=cat_perc[feature],
                            palette=['g','r'])
    plt.axvline(int(pos_client), color="black", linestyle='--', label = 'Client')
    #ax.set_ylabel('')    
    #ax.set_xlabel('')
    plt.legend(['Client','Bon client','Mauvais client' ])
    #plt.set(yticklabels=[])
    #plt.set(xticklabels=[])
    st.pyplot(fig)


st.set_page_config(
    page_title = 'Interactive Scoring Dashboard',
    page_icon = LOGO_IMAGE,
    layout = 'wide'
)

client_liste = pd.unique(df['SK_ID_CURR'])

# Panel de gauche
##########################################################
with st.sidebar:
    st.image(LOGO_IMAGE, width=300)
    st.markdown("<h1 style='text-align: center; color: black;'>Interactive Scoring Dashboard</h1>", unsafe_allow_html=True)
    st.text("")
    st.markdown("<h2 style='text-align: center; color: grey;'>Ce dashboard interactif est mis a disposition pour permettre de connaitre et de comprendre pour un client donne, la decision d'accord de pret ou non.</h2>", unsafe_allow_html=True)

    #API_url_cl = "http://127.0.0.1:5000/give_ids"
    #json_url_cl = urlopen(API_url_cl)
    #liste = json.loads(json_url_cl.read())
    #client_id = st.selectbox("Choisir le client ID", liste)
    st.markdown("***")
    client_id = st.selectbox("Choisir le client ID", pd.unique(df['SK_ID_CURR']))
    st.markdown("***")
    st.markdown("<h3 style='text-align: center; color: black;'>Ce dashboard est mis a disposition par l'entreprise Prêt à dépenser</h3>", unsafe_allow_html=True)
    st.text("")
    st.markdown("<h3 style='text-align: center; color: black;'>Ce dashboard a pour derniere version celle en date du 11/07/2022</h3>", unsafe_allow_html=True)

# Recuperation des donnees concernant le client selectionne
client_info = data[data['SK_ID_CURR']==int(client_id)]        
    
# Container de haut avec les information relatives au client
##########################################################
placeholder = st.empty()
    
with placeholder.container():
    st.markdown("## Information relative aux caracteristiques du client")
    
    #Separation du container en 3 colonnes
    fig_col1, fig_col2, fig_col3 = st.columns(3)
    
    #Premiere colonne avec les infos liees au clients
    with fig_col1:
        st.markdown("### Profil personnel du client")
        #Initialisation des infos
        default_list_perso=["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","OCCUPATION","REVENUS" ]   
        personal_info_df = client_info[list(personal_info_cols.keys())]
        personal_info_df.rename(columns=personal_info_cols, inplace=True)
        personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
        personal_info_df["NB ANNEES EMPLOI"] = int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))
        
        #Affichage du filtres ett valeurs du filtre
        filtered_perso = st.multiselect("Selectionner les informations à afficher", options=list(default_list_perso), default=list(default_list_perso))                             
 
        #Affichage du tableau filtre
        df_info_perso = personal_info_df[filtered_perso] 
        df_info_perso['SK_ID_CURR'] = client_info['SK_ID_CURR']
        df_info_perso = df_info_perso.set_index('SK_ID_CURR')
        st.table(df_info_perso.astype(str).T)
            
    #Deuxieme colonne avec les infos liees au pret du client
    with fig_col2:
        st.markdown("### Profil pret du client")

        #Initialisation des infos
        default_list_loan = ["MONTANT CREDIT","TYPE CONTRAT","MONTANT ANNUITES","SCORE SOURCE 1","SCORE SOURCE 2","SCORE SOURCE 3"]
        loan_info_df = client_info[list(loan_info_cols.keys())]
        loan_info_df.rename(columns=loan_info_cols, inplace=True)
        
        #Affichage du filtres ett valeurs du filtre
        filtered_loan = st.multiselect("Selectionner les informations à afficher", options=list(loan_info_df.columns), default=list(default_list_loan))
        
        #Affichage du tableau filtre
        df_info_loan = loan_info_df[filtered_loan] 
        df_info_loan['SK_ID_CURR'] = client_info['SK_ID_CURR']
        df_info_loan = df_info_loan.set_index('SK_ID_CURR')
        st.table(df_info_loan.astype(str).T)
    
    #Troisieme colonne avec les graphes de comparaison du client aux autres
    with fig_col3:
        st.markdown("### Comparaion du client aux autres")
        var = st.selectbox("Sélectionner une variable", list(info_cols_vis.values()))
        feature = list(info_cols_vis.keys())\
                [list(info_cols_vis.values()).index(var)]    

        if (feature in numerical_features):                
            compare_numerical(data, feature, client_info[feature])
        else:
            compare_categorical(data, feature, client_info[feature])
                    
st.markdown("***")

# Container du bas avec les information relatives au scoring du client
##########################################################
placeholder = st.empty()

with placeholder.container():
        st.markdown("## Information relative a l'accord du pret ou non au client")
        
        #Separation du container en 3 colonnes
        fig_col11, fig_col21, fig_col31 = st.columns(3)
        
        #Premiere colonne avec le scoring du client et l'accord ou pas
        with fig_col11:
            
            st.markdown("### Scoring : Accord pret ou non du client")
            
            #Recuperation des resultats du scoring du backend
            predict_data = get_result("http://127.0.0.1:5000/predict/" + str(client_id))
            classe_predite = predict_data['prediction']
            client_score = predict_data['proba']*100
            seuil_banque = predict_data['seuil']*100

            if classe_predite == "Prêt accepté":
                gauge = go.Figure(go.Indicator(
                    mode = "gauge+delta+number",
                    title = {'text': 'Bonne nouvelle, votre pret est accepte!'},
                    value = client_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {'axis': {'range': [None, 100]},
                             'steps' : [
                                 {'range': [0, seuil_banque], 'color': "lightgreen"},
                                 {'range': [seuil_banque, 100], 'color': "red"},
                                 ],
                             'threshold': {
                            'line': {'color': "black", 'width': 5},
                            'thickness': 0.5,
                            'value': client_score},

                             'bar': {'color': "black", 'thickness' : 0.1},
                            },
                    ))
            else:
                 gauge = go.Figure(go.Indicator(
                    mode = "gauge+delta+number",
                    title = {'text': 'Desole, votre pret n''est pas accepte!'},
                    value = client_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {'axis': {'range': [None, 100]},
                             'steps' : [
                                 {'range': [0, seuil_banque], 'color': "green"},
                                 {'range': [seuil_banque, 100], 'color': "red"},
                                 ],
                             'threshold': {
                            'line': {'color': "black", 'width': 5},
                            'thickness': 0.5,
                            'value': client_score},

                             'bar': {'color': "black", 'thickness' : 0.1},
                            },
                    ))

            gauge.update_layout(width=450, height=250, 
                                    margin=dict(l=50, r=50, b=0, t=0, pad=4))

            st.plotly_chart(gauge)


            
            #fig = px.density_heatmap(data_frame=df, y = 'age_new', x = 'marital')
           #st.write(fig)
        with fig_col21:
            st.markdown("### Interpretation du scoring propre au client")
            shap.initjs()
            ignore_features = ['TARGET','SK_ID_CURR','PREV_APP_CREDIT_PERC_MAX', 'REFUSED_APP_CREDIT_PERC_MAX', 'INSTAL_PAYMENT_PERC_MAX']
            relevant_features = [col for col in df.columns if col not in ignore_features]
            
            X = df[df['SK_ID_CURR']==int(client_id)]
            X = X[relevant_features]
            

            fig, ax = plt.subplots(figsize=(15, 15))
            #explainer = shap.TreeExplainer(model)
            #shap_values = explainer.shap_values(X)
            
            interpret_data = get_result("http://127.0.0.1:5000/interpret/" + str(client_id))
            finalshap = np.asarray(interpret_data['array'])
            shap.summary_plot(finalshap[1], X, plot_type ="bar", \
                               max_display=5, color_bar=False, plot_size=(8, 8))
            
            #st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="jpg")
            st.image(buf, width=250)
            #st.image(buf)

            
        with fig_col31:
            st.markdown("### Interpretation globale du scoring ")
            st.image('global_feature_importance.png')
            