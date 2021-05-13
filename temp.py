# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import streamlit as st
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import time
import shap
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
st.set_page_config(layout="wide")
with open('shap_explainer_values_X_test.pickle', 'rb') as handle:
    shap_values = pickle.load(handle)
        
    
st.markdown("<h1 style='text-align: right; color: #161616;'><strong><u>Dashboard App</u></strong></h1>", unsafe_allow_html=True)



DATA = (r'C:\Antoine\Data Scientist Openclassrooms\P7\df_train.csv')
APPLICATION_TRAIN = (r'C:\Antoine\Data Scientist Openclassrooms\P7\application_train.csv')
HOMECREDIT = (r'C:\Antoine\Data Scientist Openclassrooms\P7\HomeCredit_columns_description.csv')

def load_data(url, nrows, encoding):
    data = pd.read_csv(url, nrows=nrows, encoding=encoding)
    #lowercase = lambda x: str(x).lower()
    #data.rename(lowercase, axis='columns', inplace=True)
    return data

# Load some rows of data into the dataframe.
data = load_data(DATA, 100000, encoding='utf-8')
data1 = data.loc[data['TARGET'] == 1]
data0 = data.loc[data['TARGET'] == 0]
HomeCredit_columns_description = load_data(HOMECREDIT, 1000, encoding='unicode_escape')

#create df with description of features
T = HomeCredit_columns_description.iloc[:, 2:4].T
T = T.rename(columns=T.iloc[0])
T = T.drop('Row', axis=0)
T = T.loc[:, ~T.columns.duplicated()]
# drop the features we deleted in df
T.drop(T.iloc[:, 96:116], axis=1, inplace=True)
T.drop(T.iloc[:, 32:34], axis=1, inplace=True)
T.drop(T.iloc[:, 42:87], axis=1, inplace=True)
# add new features to explanation dataframe 
T['FLAG_TOTAL_DOC_NUM'] = 'Total number of documents presented (document 1 to 21)'
T['new_app_credit_ratio'] = 'Credit requested/credit given ratio'
T['NEW_ANNUITY_CREDIT_RATIO'] = 'Loan installment/credit amount ratio'
T['NEW_CREDIT_GOODS_RATIO'] = 'Credit amount/goods price ratio'
T['NEW_AMT_INTEREST'] = 'Interest amount'
T['NEW_INTEREST_RATIO'] = 'Interest ratio'
T['NEW_AMT_NEEDED_CREDIT_RATIO'] = 'needed amount/credit amount ratio'
T['DAYS_EMPLOYED_PERC'] = 'Ratio of working time during lifetime'
T['INCOME_CREDIT_PERC'] = 'Income/credit amount ratio'
T['INCOME_PER_PERSON'] = 'Income per person in family'
T['ANNUITY_INCOME_PERC'] = 'Loan installment/income ratio'
T['Credit_Duration'] = 'Duration of the credit (days)'
T['PERC_PAYM_AMT'] = 'Percentage of the amount paid in each installment'
T.columns = T.columns.str.lower()
T = T.sort_index(ascending=True, axis=1)

data.fillna(data.mean(), inplace=True)
y = data['TARGET']
X = data.drop(['TARGET'], axis=1)
application_train = load_data(APPLICATION_TRAIN, 100, encoding='utf-8')

temp_value_cnt = data["TARGET"].value_counts()
df1 = pd.DataFrame({'Class': temp_value_cnt.index,
                   'Values': temp_value_cnt.values
                  })

df_education = data.groupby('TARGET')['NAME_EDUCATION_TYPE_Incomplete higher',
                                          'NAME_EDUCATION_TYPE_Higher education',
                                          'NAME_EDUCATION_TYPE_Academic degree',
                                          'NAME_EDUCATION_TYPE_Lower secondary',
                                          'NAME_EDUCATION_TYPE_Secondary / secondary special'
                                          ].sum().sort_values(by=1,
                                                              axis=1,
                                                              ascending=False)

                                                              
df_organization_type = data.groupby('TARGET')['ORGANIZATION_TYPE_Military',
                                    'ORGANIZATION_TYPE_Legal Services',
                                    'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Other',
                                    'ORGANIZATION_TYPE_Police', 'ORGANIZATION_TYPE_Postal',
                                    'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Restaurant',
                                    'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Security',
                                    'ORGANIZATION_TYPE_Security Ministries',
                                    'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_Services',
                                    'ORGANIZATION_TYPE_Telecom', 'ORGANIZATION_TYPE_Trade',
                                    'ORGANIZATION_TYPE_Transport',
                                    'ORGANIZATION_TYPE_Advertising', 'ORGANIZATION_TYPE_Agriculture',
                                    'ORGANIZATION_TYPE_Bank', 'ORGANIZATION_TYPE_Business Entity',
                                    'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Construction',
                                    'ORGANIZATION_TYPE_Culture', 'ORGANIZATION_TYPE_Electricity',
                                    'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government',
                                    'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing',
                                    'ORGANIZATION_TYPE_Industry', 'ORGANIZATION_TYPE_Medicine',
                                    ].sum().sort_values(by=1,
                                                       axis=1,
                                                       ascending=False)

df_occupation_type = data.groupby('TARGET')['OCCUPATION_TYPE_Low-skill Laborers',
                                          'OCCUPATION_TYPE_Managers',
                                          'OCCUPATION_TYPE_Medicine staff','OCCUPATION_TYPE_Realty agents',
                                          'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries',
                                          'OCCUPATION_TYPE_Security staff',
                                          'OCCUPATION_TYPE_Waiters/barmen staff',
                                          'OCCUPATION_TYPE_Accountants',
                                          'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
                                          'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
                                          'OCCUPATION_TYPE_HR staff',
                                          'OCCUPATION_TYPE_High skill tech staff',
                                          'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
                                          'OCCUPATION_TYPE_Private service staff',
                                         ].sum().sort_values(by=1,
                                                       axis=1,
                                                       ascending=False)                                                        

df_family_status = data.groupby('TARGET')['NAME_FAMILY_STATUS_Married',
                                      'NAME_FAMILY_STATUS_Separated',
                                      'NAME_FAMILY_STATUS_Single / not married',
                                      'NAME_FAMILY_STATUS_Widow'
                                     ].sum().sort_values(by=1,
                                                       axis=1,
                                                       ascending=False)




#st.dataframe(application_train)
st.sidebar.markdown("<h1 style='text-align: center; color: #aaccee;'><strong><u>Bank Dashboard</u></strong></h1>", unsafe_allow_html=True)
st.sidebar.write("\n")


info = st.sidebar.radio("Select tab",
                 ('General', 'Personal', 'loan accredibility'))

st.set_option('deprecation.showPyplotGlobalUse', False)

if info == 'Personal':

    st.write('Your personal informations :')
    application_train.style.background_gradient(cmap='Oranges')
    ID = st.number_input('Input your ID_CURR',
                          min_value=min(application_train['SK_ID_CURR']),
                          max_value=max(application_train['SK_ID_CURR']))
    # define
    #x_sex=application_train.loc[application_train['SK_ID_CURR'] == ID].iloc[:, 3].values[0]
    st.write(f'**Sex** : `{application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 3].values[0]}`')
    st.write(f'**Age** : `{np.floor(application_train.loc[application_train["SK_ID_CURR"] == ID]["DAYS_BIRTH"]/-365).astype("int").values[0]} years old`')
    st.write(f'**Do you own a house or a flat ?** : `{application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 5].values[0]}`')
    st.write(f'**Do you own a car** : `{application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 4].values[0]}`')
    st.write(f'**Family status** : `{application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 14].values[0]}`')
    st.write(f'**Annual income ** : `{application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 7].astype("int").values[0]}$`')
    st.write(f'**Credit amount ** : `{application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 8].astype("int").values[0]}$`')
    st.write(f'**Installment amount each month ** : `{(application_train.loc[application_train["SK_ID_CURR"] == ID].iloc[:, 9].values[0]/12).astype("int")}$`')

elif info == 'General':
    
    st.subheader("General information")
    option = st.selectbox(
    'Select your KPI',
     ('Default payment', 'Education', 'Work field', 'Occupation', 'Family status'))

    #left_general, right_general = st.beta_columns(2)
    
    if option == 'Default payment':
        #plt.style.use('ggplot')
        labels = ['Credit accredibility', 'Clients with payment difficulties']
        values = df1['Values']
        colors=['#022282', '#FF511A']
        pie = go.Figure(data=[go.Pie(labels=labels,
                                     values=values,
                                     pull=[0, 0.2],
                                     rotation=105)])
        pie.update_traces(hoverinfo='none', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))
        pie.update(layout_title_text='Percentage of propable refusal for loan')
        
        st.write(pie)
        
    elif option == 'Education':
        colors=['#022282','#412E68', '#813A4E', '#C04434', '#FF511A']
        columns = pd.Series(100*(df_education.iloc[1,:]/df_education.iloc[0,:]).sort_values()).index.values
        array_values = 100*(df_education.iloc[1,:]/df_education.iloc[0,:]).sort_values().values        
        bar = px.bar(y=columns,
                     x=array_values,
                     orientation='h',
                     opacity=0.9)
        bar.update_layout(title='Percentage of client with default risk based on education type',
                      yaxis_title='Education type',
                      xaxis_title='%',
                     template="plotly_dark",
                     hovermode='x')
        bar.update_yaxes(showticklabels=True,
                     ticktext=['Lower secondary',
                               'Secondary',
                               'Incomplete higher',
                               'Higher education',
                               'Academic degree'][::-1],
                     tickvals=columns)
        bar.update_traces(hovertemplate=None, hoverinfo='x', textfont_size=20,
                  marker=dict(color=colors, line=dict(color='#000000', width=1)))
        
        st.plotly_chart(bar)   
        
    elif option =='Work field':
        colors=['#022282', '#262973','#4A2F64', '#6E3655', '#933D47', '#B74438', '#DB4A29', '#FF511A']
        columns = pd.Series(100*(df_organization_type.iloc[1,:]/df_organization_type.iloc[0,:]).sort_values()).index.values
        array_values = 100*(df_organization_type.iloc[1,:]/df_organization_type.iloc[0,:]).sort_values().values
        bar2 = px.bar(y=columns[:8],
                     x=array_values[:8],
                     orientation='h',
                     opacity=0.9)
        bar2.update_layout(title='Top work fields for clients with default risk',
                      yaxis_title='Work field',
                      xaxis_title='%',
                     template="plotly_dark",
                     hovermode='x')
        bar2.update_yaxes(showticklabels=True,
                     ticktext=['Restaurant', 'Agriculture', 'Construction',
                               'Self_employed', 'Security', 'Trade', 'Postal',
                               'Transport'][::-1], #,'Business entity', 'Telecom','Electricity', 'Advertising', 'Housing','Industry', 'Other'],
                     tickvals=columns[:8])
        bar2.update_traces(hovertemplate=None, hoverinfo='x', textfont_size=40,
                  marker=dict(color=colors, line=dict(color='#000000', width=1)))
        st.plotly_chart(bar2)   
        
    
    elif option =='Occupation':
        colors=['#022282', '#262973','#4A2F64', '#6E3655', '#933D47', '#B74438', '#DB4A29', '#FF511A']
        columns = pd.Series(100*(df_occupation_type.iloc[1,:]/df_occupation_type.iloc[0,:]).sort_values()).index.values
        array_values = 100*(df_occupation_type.iloc[1,:]/df_occupation_type.iloc[0,:]).sort_values().values
        bar3 = px.bar(y=columns[10:],
                     x=array_values[10:],
                     orientation='h',
                     opacity=0.9)
        bar3.update_layout(title='Top jobs with default risk',
                      yaxis_title='Job',
                      xaxis_title='%',
                     template="plotly_dark",
                     hovermode='x')
        bar3.update_yaxes(showticklabels=True,
                     ticktext=['Low-skill laborers', 'Drivers', 'Waiter/Barmen',
                               'Security staff', 'Laborers', 'Cooking staff',
                               'Sales staff', 'Cleaning staff',
                               'Realty agents', 'Secretaries'][::-1],
                     tickvals=columns[8:])
        bar3.update_traces(hovertemplate=None, hoverinfo='x', textfont_size=40,
                  marker=dict(color=colors, line=dict(color='#000000', width=1)))
        st.plotly_chart(bar3)   
    
    elif option =='Family status':
        colors=['#022282','#412E68', '#813A4E', '#C04434', '#FF511A']
        columns = pd.Series(100*(df_family_status.iloc[1,:]/df_family_status.iloc[0,:]).sort_values()).index.values
        array_values = 100*(df_family_status.iloc[1,:]/df_family_status.iloc[0,:]).sort_values().values
        bar4 = px.bar(y=columns,
                     x=array_values,
                     orientation='h',
                     opacity=0.9)
        bar4.update_layout(title='Percentage of persons with payment difficulties based on family status',
                      yaxis_title='Status',
                      xaxis_title='%',
                     template="plotly_dark",
                     hovermode='x')
        bar4.update_yaxes(showticklabels=True,
                     ticktext=['Single/Not married', 'Separated',
                               'Married',
                               'Widowed'][::-1],
                     tickvals=columns)
        bar4.update_traces(hovertemplate=None, hoverinfo='x', textfont_size=40,
                  marker=dict(color=colors, line=dict(color='#000000', width=1)))
        st.plotly_chart(bar4)      
    else: 
        st.write('Warum ?')
    
    left_general, right_general = st.beta_columns(2)

    # left column
    left_general.subheader("Informations for persons with accepted loans")
        
    left_general.write("\n")
    left_general.write("\n")
    general_payment = data.loc[data['TARGET'] == 0]
    general_default = data.loc[data['TARGET'] == 1]
    
    
    left_general.write(f'**Mean credit amount ** : `{np.floor(general_payment["AMT_CREDIT"].median())}$`')
    left_general.write(f'**Mean duration of credits ** : `{np.round((general_payment["AMT_CREDIT"] / general_payment["AMT_ANNUITY"]).mean(), 1)} years`')
    left_general.write(f'**Mean age at the time of application ** : `{np.round(-1/365*general_payment["DAYS_BIRTH"].mean(), 1)} years`')
    left_general.write(f'**Mean annual income ** : `{np.floor(general_payment["AMT_INCOME_TOTAL"].median())}$`')
    left_general.write(f'**Mean employment ** : `{np.round(-1/365*general_payment["DAYS_EMPLOYED"].median(), 1)} years`')

    # right column
    right_general.subheader("Informations for persons with risk")
        
    right_general.write("\n")
    right_general.write("\n")


    right_general.write(f'**Mean credit amount ** : `{np.floor(general_default["AMT_CREDIT"].median())}$`')
    right_general.write(f'**Mean duration of credits ** : `{np.round((general_default["AMT_CREDIT"] / general_default["AMT_ANNUITY"]).mean(), 1)} years`')
    right_general.write(f'**Mean age at the time of application ** : `{np.round(-1/365*general_default["DAYS_BIRTH"].mean(), 1)} years`')
    right_general.write(f'**Mean annual income ** : `{np.floor(general_default["AMT_INCOME_TOTAL"].median())}$`')
    right_general.write(f'**Mean employment ** : `{np.round(-1/365*general_default["DAYS_EMPLOYED"].median(), 1)} years`')    
    
elif info =='loan accredibility':
    st.subheader("Predictions on default payment")
    
    ID = st.number_input('Input your ID_CURR',
                          min_value=min(data['SK_ID_CURR']),
                          max_value=max(data['SK_ID_CURR']))
    left_shap, right_shap = st.beta_columns(2)
    if (ID == data['SK_ID_CURR']).any(): 
        
        slider1 = left_shap.slider('Number of features', min_value=2,
                                   max_value=20, value=5)
        fig = shap.plots.waterfall(shap_values[data.reset_index()[data.reset_index()['SK_ID_CURR'] == ID].index[0]  ], max_display=slider1)
        st.pyplot(fig, bbox_inches='tight', figsize=(15, 15))
        plt.clf()
    
        feature_name = st.selectbox('Choose the feature for definition',
                                 T.columns.values)
    
        st.write(T[feature_name]['Description'])

    else: 
        
        st.markdown(
        f'<div style="color: #FF0000; font-size: large; text-align: center">Error : wrong ID number</div>',
        unsafe_allow_html=True)
        
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")


