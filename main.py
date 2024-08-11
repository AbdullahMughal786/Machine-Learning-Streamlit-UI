from joblib import load
import streamlit as st
import numpy as np


model = load('Final Model.joblib')
scaler = load('Scaler.joblib')

Item_Identifier_encoder = load('identifier_encoder.joblib')
Item_Fat_Content_encoder = load('fat_encoder.joblib')
Item_Type_encoder = load('type_encoder.joblib')
Outlet_Identifier_encoder = load('outlet_identifier_encoder.joblib')
Outlet_Size_encoder = load('outlet_size_encoder.joblib')
Outlet_Location_Type_encoder = load('outlet_location_encoder.joblib')
Outlet_Type_encoder = load('outlet_type_encoder.joblib')


st.title('Welcome to the BigMart')

st.markdown("This is the Machine Learning Model")
st.markdown('This model will predict the Item Outlet Sales from the mart')
st.subheader('Okay Lets enter the following features to predict the Item Sales:')


Item_Identifier = st.text_input('Item Identifier:')

Item_Weight	= st.number_input('Item_Weight:')


Item_Fat_Content = st.text_input('Item_Fat_Content:')

Item_Visibility	= st.number_input('Item_Visibility:')

Item_Type = st.text_input('Item_Type:')

Item_MRP = st.number_input('Item_MRP:')

Outlet_Identifier = st.text_input('Outlet_Identifier:')


Outlet_Establishment_Year = st.number_input('Outlet_Establishment_Year:')

Outlet_Size	= st.text_input('Outlet_Size:')


Outlet_Location_Type = st.text_input('Outlet_Location_Type:')


Outlet_Type	= st.text_input('Outlet_Type:')

# def encode_with_default(encoder, value, default=0):   # Using this for unseen catagorical values
#     try:
#         return encoder.transform([value])[0]
#     except ValueError:
#         return default

def encode_with_default(encoder, values, default=0):
    try:
        return encoder.transform(values)
    except ValueError:
        return np.full_like(np.array(values), default)


if st.button('Predict'):
    # Item_Identifier = Item_Identifier_encoder.transform([Item_Identifier])
    # Item_Fat_Content = Item_Fat_Content_encoder.transform([Item_Fat_Content])
    # Item_Type = Item_Type_encoder.transform([Item_Type])
    # Outlet_Identifier = Outlet_Identifier_encoder.transform([Outlet_Identifier])
    # Outlet_Size = Outlet_Size_encoder.transform([Outlet_Size])
    # Outlet_Location_Type = Outlet_Location_Type_encoder.transform([Outlet_Location_Type])
    # Outlet_Type = Outlet_Type_encoder.transform([Outlet_Type])




    encoded_item_identifier = encode_with_default(Item_Identifier_encoder, Item_Identifier)
    encoded_fat_content = encode_with_default(Item_Fat_Content_encoder, Item_Fat_Content)
    encoded_item_type = encode_with_default(Item_Type_encoder, Item_Type)
    encoded_outlet_identifier = encode_with_default(Outlet_Identifier_encoder, Outlet_Identifier)
    encoded_outlet_size = encode_with_default(Outlet_Size_encoder, Outlet_Size)
    encoded_outlet_location = encode_with_default(Outlet_Location_Type_encoder, Outlet_Location_Type)
    encoded_outlet_type = encode_with_default(Outlet_Type_encoder, Outlet_Type)



    arr = np.array([[encoded_item_identifier,Item_Weight,encoded_fat_content,
                         Item_Visibility,encoded_item_type,Item_MRP,encoded_outlet_identifier,
                          Outlet_Establishment_Year,encoded_outlet_size,encoded_outlet_location,encoded_outlet_type]])

    scaled_arr = scaler.transform(arr)

    pred = model.predict(scaled_arr)

    # pred2 = model.predict([[-1.735801,1.867626, -2.123779, -1.135138, - 1.716656, - 0.532035, - 1.664513,0.139541, -1.093326,-1.369334, -1.508289]])

    st.success(f'The prediction is: {pred}')
    # st.success(f'The prediction is: {pred2}')






