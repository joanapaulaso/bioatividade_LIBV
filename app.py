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
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import seaborn as sns
import matplotlib.pyplot as plt


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
        st.header(f'**Saída das predições - Bioatividade em relação ao modelo {selected_model_name}**')
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

def lipinski(smiles, verbose=False):

    try:
        moldata= []
        for elem in smiles:
            mol=Chem.MolFromSmiles(elem) 
            moldata.append(mol)
        
        baseData= np.arange(1,1)
        i=0  
        for mol in moldata:        
        
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
            
            row = np.array([desc_MolWt,
                            desc_MolLogP,
                            desc_NumHDonors,
                            desc_NumHAcceptors])   
        
            if(i==0):
                baseData=row
            else:
                baseData=np.vstack([baseData, row])
            i=i+1      
        
        columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
        descriptors = pd.DataFrame(data=baseData,columns=columnNames, index=df_labeled.index)
        
        return descriptors
    
    except Exception as e:
        st.error(f'Erro no cálculo dos descritores de Lipinski: {e}')
        return pd.DataFrame()



def search_target(search):
    try:
        target = new_client.target
        target_query = target.search(search)
        targets = pd.DataFrame.from_dict(target_query)
        return targets
    except Exception as e:
        st.error(f'Erro na busca do alvo: {e}')
        return pd.DataFrame()
    
def clean_smiles(df):
    df_no_smiles = df.drop(columns = 'canonical_smiles')
    smiles = []

    for i in df.canonical_smiles.tolist():

        cpd = str(i).split('.')
        cpd_longest = max(cpd, key=len)
        smiles.append(cpd_longest)

    smiles = pd.Series(smiles, name = 'canonical_smiles', index=df_no_smiles)
    df_clean_smiles = pd.concat([df_no_smiles, smiles], axis=1)
    return df_clean_smiles


def label_bioactivity(df_selected):
    
    bioactivity_threshold = []
        
    for i in df_selected.standard_value:
        if float(i) >= 10000:
            bioactivity_threshold.append("inativo")
        elif float(i) <= 1000:
            bioactivity_threshold.append("ativo")
        else:
            bioactivity_threshold.append("intermediário")
    bioactivity_class = pd.Series(bioactivity_threshold, name='class', index = df_selected.index)
    
    
    
    df_labeled = pd.concat([df_selected, bioactivity_class], axis=1)
    return df_labeled


def select_target(selected_index):
    try:
        selected_index = int(selected_index)
        selected_target = targets.loc[selected_index]['target_chembl_id']
        activity = new_client.activity
        res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
        df = pd.DataFrame.from_dict(res)

        st.header("Dados das moléculas")
        df
        medidas = df['standard_units'].value_counts()
        st.write("Medidas:")
        st.write(medidas)

        df_clean = df[df.standard_value.notna()]
        df_clean = df_clean[df.canonical_smiles.notna()]
        df_clean = df_clean.drop_duplicates(['canonical_smiles'])

        selection = ['molecule_chembl_id','canonical_smiles','standard_value']
        df_selected = df_clean[selection]

        return df_selected
    
    except Exception as e:
        st.error(f'Erro na seleção do alvo: {e}')
        return pd.DataFrame()

def remove_low_variance(input_data, threshold=0.1):
    
    try:
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]
    except Exception as e:
        st.error('Erro na remoção de baixa variância: {e}')
        return pd.DataFrame()

def model_generation():
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
        X = remove_low_variance(X, threshold=0.1)
        X.to_csv(f'descriptor_lists/{model_name}_descriptor_list.csv', index = False)
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X, Y)
        Y_pred = model.predict(X)
        mse = mean_squared_error(Y, Y_pred)
        r2 = r2_score(Y, Y_pred)
        with st.spinner("Realizando análise do modelo: "):
            model_graph_analysis(Y, Y_pred, mse, r2)
        pickle.dump(model, open(f'models/{model_name}.pkl', 'wb'))
        st.success(f'Modelo {model_name} criado! Agora está disponível para predições.')
    except:
        st.error('Falha na criação do modelo: {e}')

    

def model_graph_analysis(Y, Y_pred, mse, r2, ):
    try:
        st.header("Análise do modelo")
        st.write(f'Mean squared error: {mse}')
        st.write(f'Coeficiente de determinação: {r2}')

        plt.clf()
        plt.figure(figsize=(5,5))
        plt.scatter(x=Y, y=Y_pred, c="#7CAE00", alpha=0.3)
        z = np.polyfit(Y, Y_pred, 1)
        p = np.poly1d(z)

        plt.plot(Y, p(Y),"#F8766D")
        plt.ylabel('pIC50 predito')
        plt.xlabel('pIC50 experimental')
        st.pyplot(plt)
    except Exception as e:
        st.error(f'Erro na análise do modelo: {e}')

def molecules_graph_analysis():
        try:
            graph1, graph2 = st.columns([0.5, 0.5])
            
            sns.set(style='ticks')
            
            plt.figure(figsize=(5.5, 5.5))
            sns.countplot(x='class', data=molecules_processed, edgecolor='black')
            plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
            plt.ylabel('Frequency', fontsize=14, fontweight='bold')
            
            with graph1:
                st.write("Frequências")
                with st.spinner("Gerando gráfico de frequências"):
                    st.pyplot(plt)

            
            plt.clf()
            sns.scatterplot(x='MW', y='LogP', data=molecules_processed, hue='class', size='pIC50', edgecolor='black', alpha=0.7)

            plt.xlabel('MW', fontsize=14, fontweight='bold')
            plt.ylabel('LogP', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            with graph2:
                st.write("MW x LogP")
                with st.spinner("Gerando gráfico MW x LogP"):
                    st.pyplot(plt)

            


            

            st.header("Classes x Descritores de Lipinski")
            graph3, graph4, graph5, graph6, graph7 = st.columns([0.20, 0.20, 0.20, 0.20, 0.20])
            
            plt.clf()
            sns.boxplot(x = 'class', y = 'pIC50', data = molecules_processed)
            plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
            plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
            
            with graph3:
                st.write("Classe x pIC50")
                with st.spinner("Gerando gráfico de Classe x pIC50"):
                    st.pyplot(plt)

            plt.clf()
            sns.boxplot(x = 'class', y = 'LogP', data = molecules_processed)
            plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
            plt.ylabel('LogP', fontsize=14, fontweight='bold')

            with graph4:
                st.write("Classe x LogP")
                with st.spinner("Gerando gráfico de Classe x LogP"):
                    st.pyplot(plt)

            
            plt.clf()
            sns.boxplot(x = 'class', y = 'MW', data = molecules_processed)
            plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
            plt.ylabel('MW', fontsize=14, fontweight='bold')

            with graph5:
                st.write("Classe x MW")
                with st.spinner("Gerando gráfico de Classe x MW"):
                    st.pyplot(plt)

            
            plt.clf()
            sns.boxplot(x = 'class', y = 'NumHDonors', data = molecules_processed)
            plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
            plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

            with graph6:
                st.write("Classe x NumHDonors")
                with st.spinner("Gerando gráfico de Classe x NumHDonors"):
                    st.pyplot(plt)

            
            plt.clf()
            sns.boxplot(x = 'class', y = 'NumHAcceptors', data = molecules_processed)
            plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
            plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

            with graph7:
                st.write("Classe x NumHAcceptors")
                with st.spinner("Gerando gráfico de Classe x NumHAcceptors"):
                    st.pyplot(plt)
        
        except Exception as e:
            st.error(f'Erro na criação dos gráficos: {e}')
        

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
            selected_molecules = select_target(selected_index)
        with st.spinner("Processando base de dados: "):
            df_labeled = label_bioactivity(selected_molecules)
            #df_clean_smiles = clean_smiles(df_labeled)
            df_lipinski = lipinski(df_labeled.canonical_smiles)
            df_combined = pd.concat([df_labeled, df_lipinski], axis=1)
            df_norm = norm_value(df_combined)
            molecules_processed = pIC50(df_norm)
            st.header("Moléculas Processadas")
            molecules_processed

        
        
        
        if not molecules_processed.empty:
            
            if st.button("Realizar análise gráfica"):
                st.header("Análise Gráfica")
                molecules_graph_analysis()
            
            model_name = st.text_input("Nome para salvamento do modelo: ")
            if st.button("Gerar modelo"):
                with st.spinner("Gerando modelo"):
                    model_generation()



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
        build_model(desc_subset)
    else:
        st.info('Utilize a barra lateral para selecionar o modelo e realizar o upload dos dados de entrada!')
else:
    with st.sidebar.header('Não há modelos disponíveis para predição'):
        pass