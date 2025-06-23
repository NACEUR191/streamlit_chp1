import streamlit as st
import pandas as pd

model = joblib.load("churn_model.pkl")

st.title(" Telecom Churn Prediction App")

st.header("Enter Customer Details:")

region = st.selectbox("REGION", options=list(range(10)))  
tenure = st.slider("TENURE (months)", 0, 60, 12)
regularity = st.slider("REGULARITY", 0, 100)
top_pack = st.selectbox("TOP_PACK (encoded)", options=list(range(10)))

#montant = st.number_input("MONTANT (Recharge Amount)", min_value=0.0)
freq_rech = st.slider("FREQUENCE_RECH", 0, 100)
#revenue = st.number_input("REVENUE (Monthly Income)", min_value=0.0)
arpu = st.number_input("ARPU_SEGMENT", min_value=0.0)
frequence = st.slider("FREQUENCE", 0, 100)
data_volume = st.slider("DATA_VOLUME", 0, 1000)
#on_net = st.slider("ON_NET", 0, 500)
#orange = st.slider("ORANGE", 0, 500)
#tigo = st.slider("TIGO", 0, 500)
#zone1 = st.slider("ZONE1", 0, 500)
#zone2 = st.slider("ZONE2", 0, 500)
#mrg = st.slider("MRG", 0, 500)

freq_top_pack = st.slider("FREQ_TOP_PACK", 0, 100)


if st.button(" Predict Churn"):
    input_data = pd.DataFrame([[
        region, tenure,# montant, freq_rech, revenue, arpu,
        #frequence, data_volume, on_net, orange, tigo,
        #zone1, zone2, mrg,
        regularity, top_pack#, freq_top_pack
    ]], columns=[
        'REGION', 'TENURE',# 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
        #'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
        #'ZONE1', 'ZONE2', 'MRG',
          'REGULARITY', 'TOP_PACK'#, 'FREQ_TOP_PACK'
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error(" This customer is likely to CHURN.")
    else:
        st.success(" This customer is likely to STAY.")
