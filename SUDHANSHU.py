#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Student-Performance-csv_T4qDx.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


"""
Here we got mean, std deviation 5 number summary , of every column accept , Extracurricular Activities
answer me why?
"""


# In[6]:


df.isnull().sum() #no null


# In[7]:


df.dtypes


# In[8]:


#Extracurricular Activities    ---->      object
"""
we have to handle it , coz machine will never understand it
"""


# In[9]:


df["Extracurricular Activities"]


# In[10]:


"""
if yes --> 1
no --> 0
"""


# In[11]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[12]:


le=LabelEncoder() #make object
le.fit_transform(df["Extracurricular Activities"])


# In[13]:


df.head()


# In[14]:


#still not chnged


# In[15]:


df["Extracurricular Activities"]=le.fit_transform(df["Extracurricular Activities"])


# In[16]:


df.head()


# In[17]:


x=df[["Hours Studied","Previous Scores","Extracurricular Activities","Sleep Hours","Sample Question Papers Practiced"]]


# In[18]:


x.head()


# In[19]:


y=df["Performance Index"]


# In[20]:


y.head()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)


# In[23]:


x_train #8000 train 


# In[24]:


x_test #2000 test


# In[25]:


"""
see the variations of numerical , some ranges bw 0-1 and 0-10 some are above 99 , this way model 
doesn't able to predict good , so what we ahve to do is feature scaling .
"""


# In[26]:


"""
In Python, feature scaling, or normalization, is a crucial preprocessing step in machine learning
to ensure all features have a similar scale, preventing features with larger magnitudes from
dominating the analysis. Common methods include standardization (Z-score normalization) and
min-max scaling, implemented using libraries like scikit-learn. 
"""


# In[27]:


"""
types:-
    1)Standardization (Z-score Normalization)
    2)Min-Max Scaling (Normalization):
    Scales the data to a specific range, often or [-1, 1].
    3)obust Scaling:
    Handles outliers more effectively than standardization or min-max scaling by using the median 
    and interquartile range (IQR) instead of the mean and standard deviation.
"""


# In[28]:


scaler = StandardScaler()


# In[29]:


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


model=LinearRegression()
model.fit(x_train_scaled,y_train)


# In[32]:


model.predict(x_test_scaled)


# In[33]:


x_test_scaled


# In[34]:


model.predict([[-1.21654316, -0.39749971, -0.998002  , -0.26408162, -0.18967192]])


# In[35]:


y_predict=model.predict(x_test_scaled)


# In[36]:


from sklearn.metrics import r2_score


# In[37]:


r2_score(y_test,y_predict)


# In[38]:


"""
as of now my model is in variable , as soon i turn off my pc it will lost all variables so, 
we have to store it in our file.
"""


# In[39]:


import pickle #pickel will help to store model in physical file so i need not to , run all celss again


# In[40]:


with open ("student_final_model.pkl","wb") as file :
    pickle.dump((model,scaler,le),file)


# In[41]:


"""
here the name of physical file is "student_final_model.pkl"
and now we want to store model , with all the changes we did so the chnges were feature scaling , 
stored in variable scaler , and le stores the chnge of categorical variable to 0,1
"""


# In[42]:


"""
now we will make an app, where we will use user interface , where as soon y click submit button
with all independent feature so the application will able to predict the model
"""


# In[43]:


#making application


# In[44]:


import streamlit as st #download this library in anaconda prompt it is used to make ui interface


# In[45]:


def load_model():
    with open("student_final_model.pkl","rb") as file: #making binary file
        model,scaler,le=pickle.load(file) #these 3 1) model , 2)scler that is feature scaling 3) le stores 4)le.fit_transform(df["Extracurricular Activities"]) 
    return model,scaler,le #it will return this 3
def pre_processing_input_data(data,scaler,le):
    data["Extracurricular Activities"]=le.fit_transform([data["Extracurricular Activities"]])[0]
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed


# In[46]:


def predict_data(data): #will predict the data
    model,scaler,le=load_model()
    processed_data=pre_processing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction


# In[47]:


def main():
    st.title("student performance prediction")
    st.write("enter your data to get your predictions ")


# In[48]:


if __name__=='__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




