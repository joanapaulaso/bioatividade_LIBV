import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Baixar Predições</a>'
    return href

# Model building
def build_model(input_data):
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

# Logo image
image = Image.open('logo.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Aplicação de Predição de Bioatividade


Essa aplicação permite prever a bioatividade direcionada para a inibição de uma enzima cujo modelo de Deep Learning foi previamente preparado.

**Credits**
- Aplicativo originalmente desenvolvido em Python + Streamlit por [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descritores calculados utilizando [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Leia o Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

# Sidebar
models = os.listdir('models')

with st.sidebar.header('1. Selecione o modelo a ser utilizado (alvo): '):
    selected_model = f"models/{st.sidebar.selectbox("Modelo", models)}"

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
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Utilize a barra lateral para selecionar o modelo e realizar o upload dos dados de entrada!')
