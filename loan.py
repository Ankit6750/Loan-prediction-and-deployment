#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install streamlit


# In[4]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle


# In[5]:


from PIL import Image


# In[6]:


pickle_in=open("rf.pkl","rb")
rf=pickle.load(pickle_in)


# In[9]:


def predict_loan(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,
                Loan_Amount_Term,Credit_History,Property_Area):
    prediction=rf.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,
       Loan_Amount_Term,Credit_History,Property_Area]])
    print(prediction)
    return prediction


# In[10]:


def main():
    st.title('Loan prediction')
    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;>Streamlit Loan Prediction ML App"</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Gender = st.text_input("Gender","type here")
    Married = st.text_input("married","type here")
    Dependents = st.text_input("dependents","type here")
    Education = st.text_input("Education","type here")
    Self_Employed = st.text_input("self employee","type here")
    ApplicantIncome = st.text_input("applicant income","type here")
    CoapplicantIncome = st.text_input("Co applicant income","type here")
    LoanAmount = st.text_input("Loan amount","type here")
    Loan_Amount_Term = st.text_input("loan amount term","type here")
    Credit_History = st.text_input("credit history","type here")
    Property_Area = st.text_input("property area","type here")
    result=""
    if st.button("Predict"):
        result=predict_loan(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,
                            Loan_Amount_Term,Credit_History,Property_Area)
    st.success('the output is {}'.format(result))
    
if __name__=='__main__':
    main()
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




