import streamlit as st
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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


def molecules_graph_analysis(molecules_processed):
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
 