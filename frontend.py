##########################################################
# to run: streamlit run frontend.py
##########################################################

import streamlit as st
import numpy as np
import pandas as pd
from urllib.request import urlopen
import json
import requests
import time # pour temps reel
import matplotlib.pyplot as plt
import seaborn as sns
#import math
from io import BytesIO
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

#Lien vers les objets
PATH_O = "Objects/"

#Chargement du logo
LOGO_IMAGE = PATH_O+"logo.png"

#Lien vers le backend
#PATH_B = "http://3.87.56.168:8080/" #if backend on AWS cloud
PATH_B = "http://127.0.0.1:5000/" #if backend is local

numerical_features = ['AGE', 'NB_ENFANTS', 'NB_ANNEES_EMPLOI', 'REVENUS','MONTANT_CREDIT','MONTANT_ANNUITES','SCORE_SOURCE_1','SCORE_SOURCE_2','SCORE_SOURCE_3']

#Fonction pour ouvrir les url et charger les objets au format JSON
#@st.cache
def get_result(url):
    json_url = urlopen(url)
    return json.loads(json_url.read())

#Fonction pour comparer une variable numerique donnee
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
    
#Fonction pour comparer une variable categorique donnee
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

#Onglet de la page web
st.set_page_config(
    page_title = 'Interactive Scoring Dashboard',
    page_icon = LOGO_IMAGE,
    layout = 'wide'
)


#Panel de gauche
##########################################################
with st.sidebar:
    st.image(LOGO_IMAGE, width=300)
    st.markdown("<h1 style='text-align: center; color: black;'>Dashboard Scoring Interactif</h1>", unsafe_allow_html=True)
    st.text("")
    st.markdown("<h2 style='text-align: center; color: grey;'>Ce dashboard interactif est mis à disposition pour permettre de connaitre et de comprendre pour un client donné, la décision d'accord de prêt ou non.</h2>", unsafe_allow_html=True)
    st.markdown("***")
    liste = get_result(PATH_B+"give_ids")
    final_liste = np.asarray(liste['array'])
    client_id = st.selectbox("Choisir l'ID du client", final_liste)
    st.markdown("***")
    st.markdown("<h3 style='text-align: center; color: black;'>Ce dashboard est mis à disposition par l'entreprise 'Prêt à dépenser'</h3>", unsafe_allow_html=True)
    st.markdown("***")
    st.markdown("<h3 style='text-align: center; color: black;'>Ce dashboard a pour dernière version celle en date du 21/07/2022</h3>", unsafe_allow_html=True)

# Recuperation des donnees concernant le client selectionne
client_info = get_result(PATH_B+"get_info/" + str(client_id)) 
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
        st.markdown("### Profil 'prêt' du client")

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
        #Affichage du filtres et valeurs du filtre
        list_all = option_list_perso + option_list_loan
        feature = st.selectbox("Sélectionner la variable pour comparaison", list(list_all)) 
        
        #Récupération des données à comparer 
        compare_info = get_result(PATH_B+"compare/" + str(feature))  
        compare_info_df = pd.DataFrame(compare_info)

        #Affichage du graphique correspondant
        if (feature in numerical_features):                
            compare_numerical(compare_info_df, feature, client_info_df[feature])
        else:
            compare_categorical(compare_info_df, feature, client_info_df[feature])
                    
st.markdown("***")

# Container du bas avec les information relatives au scoring du client
##########################################################
placeholder = st.empty()

with placeholder.container():
        st.markdown("## Information relative à l'accord du prêt ou non au client")
        
        #Separation du container en 3 colonnes
        fig_col11, fig_col21, fig_col31 = st.columns(3)
        
        #Premiere colonne avec le scoring du client et l'accord ou pas
        with fig_col11:
            
            st.markdown("### Scoring : Accord prêt ou non du client")
            
            #Recuperation des resultats du scoring du backend
            predict_data = get_result(PATH_B+"predict/" + str(client_id))
            classe_predite = predict_data['prediction']
            client_score = predict_data['proba']*100
            seuil_banque = predict_data['seuil']*100

            if classe_predite == "Prêt accepté":
                gauge = go.Figure(go.Indicator(
                    mode = "gauge+delta+number",
                    title = {'text': 'Bonne nouvelle, votre prêt est accepté!'},
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
                    title = {'text': 'Desolé, votre prêt n''est pas accepté!'},
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
            st.markdown("### Interprétation 'client' du scoring")    

            fig, ax = plt.subplots(figsize=(8, 6))
            # Recuperation des donnees concernant le client selectionne
            shap_info = get_result(PATH_B+"interpret/" + str(client_id))
            shap_info_df = pd.DataFrame(shap_info)
            colors = ['r' if e >= 0 else 'b' for e in shap_info_df['shap']]
            x= shap_info_df['index']
            y= shap_info_df['shap']
            sns.barplot(y, x, palette=colors, errwidth=0) 
            sns.despine(bottom = True, left = True)
            ax.set(xlabel=None)
            ax.set(ylabel=None)

            st.pyplot(fig, use_column_width='auto', width=600)
           
        with fig_col31:
            st.markdown("### Interprétation globale du scoring ")
            st.image(PATH_O+'global_feature_importance.png', width=500)
            