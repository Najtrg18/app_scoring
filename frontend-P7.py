##########################################################
# to run: streamlit run frontend.py
##########################################################

import streamlit as st
import numpy as np
import pickle
import pandas as pd
from urllib.request import urlopen
import json
import requests
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

#Chargement du logo
LOGO_IMAGE = "logo.png"

#PATH_B = "http://54.175.61.168:5000/"
PATH_B = "http://127.0.0.1:5000/"

numerical_features = ['AGE', 'NB_ENFANTS', 'NB_ANNEES_EMPLOI', 'REVENUS','MONTANT_CREDIT','MONTANT_ANNUITES','SCORE_SOURCE_1','SCORE_SOURCE_2','SCORE_SOURCE_3']

#@st.cache
def get_result(url):
    json_url = urlopen(url)
    return json.loads(json_url.read())
    
#@st.cache
def compare_numerical(df, feature, client_feature_val):

    fig, ax = plt.subplots(1, 1, figsize = (10, 5), dpi=300)
    #fig = plt.figure(figsize = (10, 5))
    df0 = df.loc[df['TARGET'] == 0]
    df1 = df.loc[df['TARGET'] == 1]
    
    sns.kdeplot(df0[feature].dropna(), label = 'Bon client', color='g')
    sns.kdeplot(df1[feature].dropna(), label = 'Mauvais client', color='r')
    
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


# Panel de gauche
##########################################################
with st.sidebar:
    st.image(LOGO_IMAGE, width=300)
    st.markdown("<h1 style='text-align: center; color: black;'>Interactive Scoring Dashboard</h1>", unsafe_allow_html=True)
    st.text("")
    st.markdown("<h2 style='text-align: center; color: grey;'>Ce dashboard interactif est mis a disposition pour permettre de connaitre et de comprendre pour un client donne, la decision d'accord de pret ou non.</h2>", unsafe_allow_html=True)

    #liste = requests.get("http://127.0.0.1:5000/give_ids")
    liste = get_result(PATH_B+"give_ids")
    final_liste = np.asarray(liste['array'])
    client_id = st.selectbox("Choisir le client ID", final_liste)
    st.markdown("***")
    st.markdown("<h3 style='text-align: center; color: black;'>Ce dashboard est mis a disposition par l'entreprise Prêt à dépenser</h3>", unsafe_allow_html=True)
    st.text("")
    st.markdown("<h3 style='text-align: center; color: black;'>Ce dashboard a pour derniere version celle en date du 14/07/2022</h3>", unsafe_allow_html=True)

# Recuperation des donnees concernant le client selectionne
client_info = get_result(PATH_B+"get_info/" + str(client_id)) 
#client_info = requests.get("http://127.0.0.1:5000/get_info/" + str(client_id))
client_info_df = pd.DataFrame(client_info)
    
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
        default_list_perso=["GENRE","AGE","STATUT_FAMILIAL","NB_ENFANTS","OCCUPATION","REVENUS" ]
        option_list_perso = ["GENRE","AGE","STATUT_FAMILIAL","NB_ENFANTS","OCCUPATION","REVENUS","PROPRIETAIRE_VEHICULE", "PROPRIETAIRE_IMMOBILIER","NIVEAU_EDUCATION", "NB_ANNEES_EMPLOI"]
        
        #Affichage du filtres ett valeurs du filtre
        filtered_perso = st.multiselect("Selectionner les informations à afficher", options=list(option_list_perso), default=list(default_list_perso))                             
 
        #Affichage du tableau filtre
        df_info_perso = client_info_df[filtered_perso] 
        st.table(df_info_perso.astype(str).T)
            
    #Deuxieme colonne avec les infos liees au pret du client
    with fig_col2:
        st.markdown("### Profil pret du client")

        #Initialisation des infos
        default_list_loan = ["MONTANT_CREDIT","TYPE_CONTRAT","MONTANT_ANNUITES","SCORE_SOURCE_1","SCORE_SOURCE_2","SCORE_SOURCE_3"]
        option_list_loan = ["MONTANT_CREDIT","TYPE_CONTRAT","MONTANT_ANNUITES","SCORE_SOURCE_1","SCORE_SOURCE_2","SCORE_SOURCE_3", "TYPE_REVENUS"]
        
        #Affichage du filtres ett valeurs du filtre
        filtered_loan = st.multiselect("Selectionner les informations à afficher", options=list(option_list_loan), default=list(default_list_loan))                             
 
        #Affichage du tableau filtre
        df_info_loan = client_info_df[filtered_loan] 
        st.table(df_info_loan.astype(str).T)
    
    #Troisieme colonne avec les graphes de comparaison du client aux autres
    with fig_col3:
        st.markdown("### Comparaion du client aux autres")
        list_all = option_list_perso + option_list_loan
        feature = st.selectbox("Sélectionner une variable", list(list_all))
        #feature = list(info_cols_vis.keys())\
                #[list(info_cols_vis.values()).index(var)]    
            
        compare_info = get_result(PATH_B+"compare/" + str(feature))  
        compare_info_df = pd.DataFrame(compare_info)

        if (feature in numerical_features):                
            compare_numerical(compare_info_df, feature, client_info_df[feature])
        else:
            compare_categorical(compare_info_df, feature, client_info_df[feature])
                    
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
            predict_data = get_result(PATH_B+"predict/" + str(client_id))
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
            
            # Recuperation des donnees concernant le client selectionne
            shap_info = get_result(PATH_B+"interpret/" + str(client_id)) 
            #client_info = requests.get("http://127.0.0.1:5000/get_info/" + str(client_id))
            shap_info_df = pd.DataFrame(shap_info)
            colors = ['r' if e >= 0 else 'b' for e in shap_info_df['shap']]
            x= shap_info_df['index']
            y= shap_info_df['shap']
            sns.barplot(y, x, palette=colors, errwidth=0) 

            #buf = BytesIO()
            #fig.savefig(buf, format="jpg")
            #st.image(buf, width=250)
            #st.image(buf)

            
        with fig_col31:
            st.markdown("### Interpretation globale du scoring ")
            st.image('global_feature_importance.png')