import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_selection import VarianceThreshold



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


def remove_low_variance(input_data, threshold=0.1):
    
    try:
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]
    except Exception as e:
        st.error('Erro na remoção de baixa variância: {e}')
        return pd.DataFrame()
