import pickle
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils.file_operations import filedownload
from utils.descriptors import desc_calc
from utils.data_processing import remove_low_variance
from utils.visualization import model_graph_analysis


def build_model(input_data, load_data, selected_model, selected_model_name):
    
   # try:
        load_model = pickle.load(open(selected_model, 'rb'))
        prediction = load_model.predict(input_data)
        st.header(f'**Saída das predições - Bioatividade em relação ao modelo {selected_model_name}**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_id = pd.Series(load_data[1], name='id_molecula')
        molecule_name = pd.Series(load_data[2], name='nome_molecula')
        df = pd.concat([molecule_id, molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)
    #except Exception as e:
        #st.error(f'Erro ao construir o modelo: {e}')


def model_generation(molecules_processed, variance, estimators, model_name):
    try: 
        selection = ['canonical_smiles','molecule_chembl_id']
        df_final_selection = molecules_processed[selection]
        df_final_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)
        with st.spinner("Calculando descritores..."):
            desc_calc()
        df_fingerprints = pd.read_csv('descriptors_output.csv')
        st.header("Descritores")
        df_fingerprints
        df_fingerprints = df_fingerprints.drop(columns = ['Name'])
        df_Y = molecules_processed['pIC50']
        df_training = pd.concat([df_fingerprints, df_Y], axis=1)
        df_training = df_training.dropna()
        X = df_training.drop(['pIC50'], axis=1)
        Y = df_training.iloc[:, -1]
        X = remove_low_variance(X, variance)
        X.to_csv(f'descriptor_lists/{model_name}_descriptor_list.csv', index = False)
        model = RandomForestRegressor(estimators, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        mse = mean_squared_error(Y, Y_pred)
        r2 = r2_score(Y, Y_pred)
        with st.spinner("Realizando análise do modelo: "):
            model_graph_analysis(Y, Y_pred, mse, r2)
        pickle.dump(model, open(f'models/{model_name}.pkl', 'wb'))
        st.success(f'Modelo {model_name} criado! Agora está disponível para predições.')
    except Exception as e:
        st.error(f'Falha na criação do modelo: {e}')

