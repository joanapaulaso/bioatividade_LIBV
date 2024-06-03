import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from chembl_webresource_client.new_client import new_client
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


#TODO: Tratamento de Exceções

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    try:
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')
    except Exception as e:
        st.error(f'Erro ao calcular descritores: {e}')
# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Baixar Predições</a>'
    return href

# Model building
def build_model(input_data):
    
   # try:
        # Reads in saved regression model
        load_model = pickle.load(open(selected_model, 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.header('**Saída das predições**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_id = pd.Series(load_data[1], name='id_molecula')
        molecule_name = pd.Series(load_data[2], name='nome_molecula')
        df = pd.concat([molecule_id, molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)
    #except Exception as e:
        #st.error(f'Erro ao construir o modelo: {e}')


def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop(columns = 'standard_value_norm')

    return x


def norm_value(input):
    norm = []

    for j in input['standard_value']:
        i = float(j)
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop(columns = 'standard_value')

    return x


def search_target(search):
    try:
        target = new_client.target
        target_query = target.search(search)
        targets = pd.DataFrame.from_dict(target_query)
        return targets
    except Exception as e:
        st.error(f'Erro na busca do alvo: {e}')
        return pd.DataFrame()


def select_target(selected_index):
    try:
        selected_index = int(selected_index)
        selected_target = targets.loc[selected_index]['target_chembl_id']
        activity = new_client.activity
        res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
        df = pd.DataFrame.from_dict(res)
        df2 = df[df.standard_value.notna()]
        df2 = df2[df.canonical_smiles.notna()]
        df2_nr = df2.drop_duplicates(['canonical_smiles'])
        selection = ['molecule_chembl_id','canonical_smiles','standard_value']
        df3 = df2_nr[selection]
        df_norm = norm_value(df3)
        df_final = pIC50(df_norm)

        return df_final
    except Exception as e:
        st.error(f'Erro na seleção do alvo: {e}')
        return pd.DataFrame()

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

def model_generation(X, Y):
    try:
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X, Y)
        pickle.dump(model, open(f'models/{search}.pkl', 'wb'))

    except Exception as e:
        st.error(f'Erro na geração do modelo: {e}')


image = Image.open('logo.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Aplicação de Predição de Bioatividade


Essa aplicação permite prever a bioatividade em relação a um alvo cujo modelo de Deep Learning foi previamente preparado.

**Credits**
- Aplicativo originalmente desenvolvido em Python + Streamlit por [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descritores calculados utilizando [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Leia o Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

st.header("Criação do modelo (CHEMBL)")


col1, col2 = st.columns([3,1])
if 'targets' not in st.session_state:
    st.session_state['targets'] = pd.DataFrame()
with col1:
    search = st.text_input("Alvo")
with col2:
    if st.button('Buscar'):
        with st.spinner("Buscando..."):
            st.session_state['targets'] = search_target(search)


targets = st.session_state['targets']
target_molecules = pd.DataFrame()


#TODO: Alinhar input e botão 

container1 = st.container()
with container1:
    if not targets.empty:
        st.write(targets)

    
if not targets.empty:
    selected_index = st.text_input("Índice do alvo selecionado")

    if selected_index:
        target_molecules = select_target(selected_index)
        target_molecules


if not target_molecules.empty:
    if st.button("Gerar modelo"):
        selection = ['canonical_smiles','molecule_chembl_id']
        df_final_selection = target_molecules[selection]
        df_final_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)
        with st.spinner("Calculando descritores..."):
            desc_calc()
        df_fingerprints = pd.read_csv('descriptors_output.csv')
        st.header("Descritores")
        df_fingerprints
        df_fingerprints = df_fingerprints.drop(columns = ['Name'])
        df_Y = target_molecules['pIC50']
        df_training = pd.concat([df_fingerprints, df_Y], axis=1)
        df_training = df_training.dropna()
        X = df_training.drop(['pIC50'], axis=1)
        Y = df_training.iloc[:, -1]
        X = remove_low_variance(X, threshold=0.1)
        X.to_csv(f'{search}_descriptor_list.csv', index = False)
        try:
            with st.spinner("Gerando modelo..."):
                model_generation(X, Y)
                st.success(f'Modelo {search} criado! Agora está disponível para predições.')
        except:
            st.error('Falha na criação do modelo: {e}')








# Sidebar
models = os.listdir('models')

with st.sidebar.header('1. Selecione o modelo a ser utilizado (alvo): '):
    selected_model_name = st.sidebar.selectbox("Modelo", models).removesuffix(".pkl")
    selected_model = f'models/{selected_model_name}.pkl'

with st.sidebar.header('2. Faça upload dos dados em CSV:'):
    uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo de entrada", type=['txt'])
    st.sidebar.markdown("""
[Exemplo de arquivo de entrada](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Prever'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep = ' ', header = False, index = False)

    st.header('**Dados originais de entrada**')
    st.write(load_data)

    with st.spinner("Calculando descritores..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Cálculo de descritores moleculares realizado**')
    desc = pd.read_csv('descriptors_output.csv', )
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subconjunto de descritores de modelos preparados previamente**')
    Xlist = list(pd.read_csv(f'{selected_model_name}_descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Utilize a barra lateral para selecionar o modelo e realizar o upload dos dados de entrada!')
