import streamlit as st
import pandas as pd
from PIL import Image
import os


from utils.data_search import search_target, select_target
from utils.data_processing import pIC50, norm_value, label_bioactivity
from utils.descriptors import lipinski, desc_calc
from utils.model import build_model, model_generation
from utils.visualization import molecules_graph_analysis


if not os.path.isdir("models"):
        os.mkdir("models")

if not os.path.isdir("descriptor_lists"):
                    os.mkdir("descriptor_lists")

image = Image.open('logo.png')
selected_index = ''

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Aplicação de Predição de Bioatividade


Essa aplicação permite predizer a bioatividade em relação a um alvo cujo modelo de Deep Learning foi previamente preparado.

**Credits**
- Aplicativo originalmente desenvolvido em Python + Streamlit por [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descritores calculados utilizando [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Leia o Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

st.header("Criação de modelo (CHEMBL)")


col1, col2 = st.columns([3,1])
if 'targets' not in st.session_state:
    st.session_state['targets'] = pd.DataFrame()
with col1:
    search = st.text_input("Alvo")
with col2:
    if st.button('Buscar'):
        selected_index = ''
        with st.spinner("Buscando..."):
            st.session_state['targets'] = search_target(search)


targets = st.session_state['targets']
molecules_processed = pd.DataFrame()


#TODO: Alinhar input e botão 

container1 = st.container()
with container1:
    if not targets.empty:
        st.write(targets)

    
if not targets.empty:
    selected_index = st.text_input("Índice do alvo selecionado")

    if selected_index:
        with st.spinner("Selecionando base de dados de moléculas: "):
            selected_molecules = select_target(selected_index, targets)
        with st.spinner("Processando base de dados: "):
            df_labeled = label_bioactivity(selected_molecules)
            #df_clean_smiles = clean_smiles(df_labeled)
            df_lipinski = lipinski(df_labeled)
            df_combined = pd.concat([df_labeled, df_lipinski], axis=1)
            df_norm = norm_value(df_combined)
            molecules_processed = pIC50(df_norm)
            st.header("Moléculas Processadas")
            molecules_processed

        
        
        
        if not molecules_processed.empty:
            
            if st.button("Realizar análise gráfica"):
                st.header("Análise Gráfica")
                molecules_graph_analysis(molecules_processed)
            
            model_col1, model_col2 = st.columns([0.5, 0.5])
            with model_col1:
                variance_input = st.number_input("Limite de variância:", min_value = 0.0, value = 0.1)
            with model_col2:
                estimators_input = st.number_input("Número de estimadores:", min_value = 1, value = 500)
            model_name = st.text_input("Nome para salvamento do modelo: ")
            if st.button("Gerar modelo"):
                with st.spinner("Gerando modelo"):
                    model_generation(molecules_processed, variance_input, estimators_input, model_name)



models = os.listdir('models')

if len(os.listdir('models')) != 0:
    with st.sidebar.header('1. Selecione o modelo a ser utilizado (alvo): '):
        selected_model_name = st.sidebar.selectbox("Modelo", models).removesuffix(".pkl")
        selected_model = f'models/{selected_model_name}.pkl'

    with st.sidebar.header('2. Faça upload dos dados em CSV:'):
        uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo de entrada", type=['txt'])
        st.sidebar.markdown("""
    [Exemplo de arquivo de entrada](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
    """)

    if st.sidebar.button('Predizer'):
        st.header('Cálculo de predição:')
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
        Xlist = list(pd.read_csv(f'descriptor_lists/{selected_model_name}_descriptor_list.csv').columns)
        desc_subset = desc[Xlist]
        st.write(desc_subset)
        st.write(desc_subset.shape)

        # Apply trained model to make prediction on query compounds
        build_model(desc_subset, load_data, selected_model, selected_model_name)
    else:
        st.info('Utilize a barra lateral para selecionar o modelo e realizar o upload dos dados de entrada!')
else:
    with st.sidebar.header('Não há modelos disponíveis para predição'):
        pass