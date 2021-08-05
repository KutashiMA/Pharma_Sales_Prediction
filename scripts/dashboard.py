import streamlit as st
import pandas as pd
import numpy as np
import pickle

months = {'Jan':4,'Feb':3,'Mar':8,'Apr':6,'May':7,'Jun':0,'Jul':5,'Aug':2,'Sep':11,'Oct':10,'Nov':9,'Dec':1}
months_list = list(months.keys())

def convert_month(name):
    m_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,\
        'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    
    return m_dict[name]

def main():

    st.title('Pharmaceutics Store Sales Predictor')

    st.write('### **This system estimates the sales a store will make in a day based on certain parameters**')

    right_column, left_column = st.beta_columns(2)

    # field to accept the store's number (max store number is 1115 and minimum is 1)
    right_column.write("### What is the Store's Number")
    store = right_column.number_input('', step=1, min_value=1, max_value=1115)

    # field to accept the expected number of customers (minimum value is 1)
    left_column.write("### Enter the expected number of customers")
    customers = left_column.number_input('', step=1, min_value=0)

    # field to selcect if there's a promo or not
    right_column.write("### Is There A promo?")
    promo = right_column.selectbox('', ['No','Yes'])

    if promo == 'No':
        promo = 0
    else:
        promo = 1

    d, m, y = st.beta_columns(3)

    d.write("### Day")
    day = d.number_input('', step=1, min_value=1, max_value=31)

    m.write("### Month")
    month = m.selectbox('', months_list)
    month = convert_month(month)

    y.write("### Year")
    year = y.number_input('', step=1, min_value=1990)

    date = str(year) + '-' + str(month) + '-' + str(day)

    df = pd.DataFrame({'Date':[date]})

    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    # extracting days
    df['day'] = df['Date'].dt.day_name()

    df['day'].replace({'Friday':5, 'Thursday':4, 'Wednesday':3, 'Tuesday':2,\
        'Monday':1, 'Sunday':7, 'Saturday':6}, inplace=True)

    # extracting year
    df['year'] = [d.year for d in df.Date]

    # extracting month
    df['month'] = [d.strftime('%b') for d in df.Date]

    df['month'].replace(months, inplace=True)

    features = []
    features.append([store])
    features.append([df.iloc[0,1]])
    features.append([customers])
    features.append([promo])
    features.append([df.iloc[0,2]])
    features.append([df.iloc[0,3]])

    scaler_name = "./models/StandardScaler.pkl"
    scaler = pickle.load(open(scaler_name, 'rb'))

    scaled_features = scaler.fit_transform(features).reshape(-1, 6)

    model_name = "./models/DecisionTreeRegressor.pkl"
    model = pickle.load(open(model_name, 'rb'))

    prediction = model.predict(scaled_features)

    predict = st.button('Predict')

    if predict:
        st.info(f'The estimated amount of sales is {prediction[0]}')

if __name__ == "__main__":
    main()
