import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from unidecode import unidecode
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud


"""
 ARQUIVO CONCENTRA AS FUNÇÕES DESENVOLVIDAS PARA O TRATAMENTO E AGRUPAMENTO DE DADOS
 DAS BASES DE EMPREENDIMENTOS E ACIDENTES. COMO AS FUNÇÕES FORAM UTILIZADAS PARA TRATAR
 A BASE PODE SER ENCONTRADO NOS SEGUINTES NOTEBOOKS:
    - tratamento_de_dados/tratamento_nulos.ipynb
    - tratamento_de_dados/tratamento_empreendimentos.ipynb
    - tratamento_de_dados/tratamento_acidentes.ipynb
    - tratamento_de_dados/agrupamento_municipio_ano.ipynb

"""


################## FUNÇÕES NO NOTEBOOK DE TRATAMENTO DE NULOS ##################


def processa_coluna_texto(df, coluna):
    """
    Função para tratar uma coluna de texto de um DataFrame. A função aplica a biblioteca 'unidecode' para remover acentos,
    converte a string para maiúsculas, remove apóstrofos e substitui hífens por espaços. 

    Parâmetros:
    df (pd.DataFrame): DataFrame que contém a coluna a ser processada.
    coluna (str): Nome da coluna a ser processada.

    Retorna:
    df (pd.DataFrame): DataFrame com a coluna processada.
    """
    df[coluna] = df[coluna].apply(lambda x: unidecode(x).upper().replace("'", "").replace("-", " ").strip() if pd.notnull(x) else x)
    return df

def atualiza_codigo_ibge(df_empreend, df_municipios, codigo_ibge, colunas_on_merge=['mun_MUNNOMEX', 'uf_SIGLA_UF']):
    """
    Função para atualizar o código IBGE no DataFrame de empreendimentos baseado no DataFrame de municípios.

    Parâmetros:
    df_empreend (pd.DataFrame): DataFrame que contém a coluna 'Código IBGE' a ser atualizada.
    df_municipios (pd.DataFrame): DataFrame que contém os códigos IBGE corretos.
    codigo_ibge (int): O código IBGE que precisa ser atualizado.
    colunas_on_merge (list): Lista de colunas para fazer o merge.

    Retorna:
    df_empreend (pd.DataFrame): DataFrame com a coluna 'Código IBGE' atualizada.
    """
    mask = df_empreend['Código IBGE'] == codigo_ibge
    df_empreend.loc[mask, 'Código IBGE'] = df_empreend[mask].merge(df_municipios, on=colunas_on_merge, how='left')['Código IBGE_y']
    return df_empreend

def merge_e_atualiza_nulos_com_codigo_ibge(df_empreend, df_municipios, colunas_descartar=['mun_MUNNOMEX', 'uf_SIGLA_UF'], merge_on='Código IBGE'):
    """
    Função para mesclar dois DataFrames e atualizar as colunas do primeiro DataFrame (a base de empreendimentos)
    com base no segundo DataFrame (a base de municípios).

    Parâmetros:
    df_empreend (pd.DataFrame): DataFrame principal que será atualizado.
    df_municipios (pd.DataFrame): DataFrame que contém as informações corretas.
    colunas_descartar (list): Lista de colunas a serem descartadas do df_municipios antes do merge - o valor default é mun_MUNOMEX e uf_SIGLA_UF, que têm poucos nulos.
    merge_on (str): Coluna para fazer o merge.

    Retorna:
    df_empreend (pd.DataFrame): DataFrame atualizado após o merge.
    """
    df_municipios_selecionado = df_municipios.copy().drop(colunas_descartar, axis=1)
    df_merge = df_empreend.merge(df_municipios_selecionado, on=merge_on, how='left', suffixes=('_empreend', '_municipios'))

    for coluna in df_empreend.columns.intersection(df_municipios_selecionado.columns):
        if coluna not in [merge_on]:
            df_merge = df_merge.drop(columns=coluna + '_empreend')
            df_merge = df_merge.rename(columns={coluna + '_municipios': coluna})

    return df_merge

def merge_e_atualiza_nulos_com_nome(df_empreend, df_municipios, merge_on=['mun_MUNNOMEX', 'uf_SIGLA_UF']):
    """
    Função para mesclar o dataframe de empreendimentos com o de municípios onde o 'Código IBGE' é nulo e 'mun_MUNNOMEX'
    não é nulo no dataframe de empreendimentos). As colunas do dataframe de empreendimentos que são repetidas com o
    de municípios são descartadas após o merge.

    Parâmetros:
    df_empreend (pd.DataFrame): DataFrame principal que será atualizado.
    df_municipios (pd.DataFrame): DataFrame que contém as informações corretas.
    merge_on (list): Lista de colunas para fazer o merge.

    Retorna:
    df_empreend (pd.DataFrame): DataFrame atualizado após o merge.
    """
    mascara = (df_empreend['Código IBGE'].isnull()) & (df_empreend['mun_MUNNOMEX'].notnull())
    print(f"Quantidade de linhas com 'Código IBGE' nulo e 'mun_MUNNOMEX' não nulo: {len(df_empreend[mascara])}")

    # Onde a mascara (filtro) é verdadeira, mescla os dataframes
    df_merge = pd.merge(df_empreend[mascara], df_municipios, on=merge_on, suffixes=('_empreend', ''))

    for coluna in df_merge.columns:
        if coluna.endswith('_empreend'):
            df_merge = df_merge.drop(coluna, axis = 1)

    df_empreend.set_index('cod_mdr', inplace=True)
    df_merge.set_index('cod_mdr', inplace=True)

    # Atualiza df_empreend com os valores de df_merge
    df_empreend.update(df_merge)

    df_empreend.reset_index(inplace=True)
    df_merge.reset_index(inplace=True)

    return df_empreend


def treinar_e_validar_modelos_classificacao(X, y, cv=5):
    """
    Função para treinar vários modelos e calcular a acurácia usando validação cruzada.

    Parâmetros:
    X (pd.DataFrame): DataFrame com os dados de entrada.
    y (pd.DataFrame): DataFrame com os dados de saída (rótulos).
    cv (int): Número de divisões para a validação cruzada.

    Retorna:
    None
    """

    modelos = [
        ('Regressão Logística', LogisticRegression()),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Support Vector Machine', SVC()),
        ('Árvore de Decisão', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier())
    ]

    for nome, modelo in modelos:
        # Validação cruzada para calcular a acurácia
        scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')

        mean_score = np.mean(scores)

        print(f'{nome} Acurácia: {mean_score * 100}%')


def otimizar_hiperparametros_knn(X, y, cv=5, scoring='accuracy'):
    """
    Função para otimizar os hiperparâmetros de um classificador K-Nearest Neighbors usando GridSearchCV.

    Parâmetros:
    X (pd.DataFrame): DataFrame com os dados de entrada.
    y (pd.DataFrame): DataFrame com os dados de saída (rótulos).
    cv (int): Número de divisões para a validação cruzada.
    scoring (str): Métrica de avaliação do desempenho do modelo.

    Retorna:
    dict: Dicionário com os melhores hiperparâmetros encontrados.
    float: Melhor acurácia encontrada.
    """
    param_grid = {
        'n_neighbors': range(1, 110),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring=scoring)

    grid_search.fit(X, y)

    melhores_hiperparametros = grid_search.best_params_
    melhor_acuracia = grid_search.best_score_

    return melhores_hiperparametros, melhor_acuracia

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Para visualizar a evolução da acurácia conforme o número
def plot_acuracia_n_vizinhos(df, variavel_preditora, variavel_predita):
    """
    Função para visualizar a evolução da acurácia de um classificador K-Nearest Neighbors
    no treinamento com diferentes números de vizinhos.

    Parâmetros:
    df (pd.DataFrame): DataFrame com os dados.
    variavel_preditora (str): Nome da coluna no DataFrame usada como variável preditora.
    variavel_predita (str): Nome da coluna no DataFrame usada como variável predita (rótulo).

    Retorna:
    accuracies (list): Lista com as acurácias para cada número de vizinhos.
    """

    X = df[[variavel_preditora]]
    y = df[variavel_predita]

    accuracies = []

    n_vizinhos_range = range(1, 110)

    # Treina o classificador K-Nearest Neighbors com cada um dos números de vizinhos em n_vizinhos_range e guarda a acurácia
    for n in n_vizinhos_range:
        knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean', weights = 'uniform')

        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

        mean_score = np.mean(scores)

        accuracies.append(mean_score)
    
    max_accuracy = max(accuracies)
    best_n_neighbors = accuracies.index(max_accuracy) + 1
    print(f'O número de vizinhos com a maior acurácia ({max_accuracy * 100}%) é: {best_n_neighbors}')

    # Plota a evolução da acurácia com o número de vizinhos
    plt.figure(figsize=(10, 6))
    plt.plot(n_vizinhos_range, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Evolução da Acurácia com o Número de Vizinhos')
    plt.xlabel('Número de Vizinhos')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.show()

    return accuracies


def preencher_nulos_com_predicoes(df, variavel_preditora, variavel_predita, n_neighbors=106, metric='euclidean', weights='uniform'):
    """
    Função para preencher valores nulos em uma coluna de um DataFrame usando previsões de um classificador K-Nearest Neighbors (KNN).
    A função recebe os hiperparâmetro do KNN que devem ser utilizados na predição.

    Parâmetros:
    df (pd.DataFrame): DataFrame que contém os dados.
    variavel_preditora (str): Nome da coluna no DataFrame usada como variável preditora.
    variavel_predita (str): Nome da coluna no DataFrame usada como variável predita (rótulo).
    n_neighbors (int): Número de vizinhos a serem usados pelo classificador K-Nearest Neighbors.
    metric (str): Métrica de distância a ser usada pelo classificador K-Nearest Neighbors.
    weights (str): Função de peso a ser usada pelo classificador K-Nearest Neighbors.

    Retorna:
    df (pd.DataFrame): DataFrame com a coluna de variável predita atualizada.
    """
    # Cria o modelo K-Nearest Neighbors com os hiperparâmetros especificados
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)

    # Treina o modelo nos dados não nulos
    df_not_null = df.dropna(subset=[variavel_predita, variavel_preditora])
    X = df_not_null[[variavel_preditora]]
    y = df_not_null[variavel_predita]
    model.fit(X, y)

    # Faz previsões para os dados nulos
    null_data = df[df[variavel_predita].isnull() & df[variavel_preditora].notnull()]
    null_predictions = model.predict(null_data[[variavel_preditora]])

    # Preenche os valores nulos com as previsões
    df.loc[df[variavel_predita].isnull() & df[variavel_preditora].notnull(), variavel_predita] = null_predictions

    return df

def atualiza_populacao_nula(df_empreend, df_populacao_colunas_selecionadas):
    """
    Função para atualizar os valores nulos da coluna 'Populacao' em um DataFrame com base em outro DataFrame.

    Parâmetros:
    df_empreend (pd.DataFrame): DataFrame que contém a coluna 'Populacao' a ser atualizada (base de empreendimentos).
    df_populacao_colunas_selecionadas (pd.DataFrame): DataFrame que contém as informações corretas (dados de população por município e por ano).

    Retorna:
    df_empreend (pd.DataFrame): DataFrame com a coluna 'Populacao' atualizada.
    """

    mascara = (df_empreend['Populacao'].isnull()) & (df_empreend['Código IBGE'].notnull()) & (df_empreend['ano'].notnull())
    print(f"Quantidade de linhas com 'Populacao' nulo, 'Código IBGE' não nulo e 'ano' não nulo: {len(df_empreend[mascara])}")

    df_merge = pd.merge(df_empreend[mascara], df_populacao_colunas_selecionadas, on=['Código IBGE', 'ano'], suffixes=('_empreend', ''))

    for coluna in df_merge.columns:
        if coluna.endswith('_empreend'):
            df_merge = df_merge.drop(coluna, axis = 1)

    df_empreend.set_index('cod_mdr', inplace=True)
    df_merge.set_index('cod_mdr', inplace=True)

    df_empreend.update(df_merge)

    df_empreend.reset_index(inplace=True)
    df_merge.reset_index(inplace=True)

    return df_empreend

################## FUNÇÕES NO NOTEBOOK DE TRATAMENTO/CATEGORIZAÇÃO DA COLUNA EMPREENDIMENTO ##################

def trata_texto(texto):
    """
    Função para tratar um texto em português. Realiza as seguintes operações:
    1. Remove acentos.
    2. Converte o texto para minúsculas.
    3. Remove apóstrofos e substitui hífens por espaços.
    4. Substitui espaços duplos por um único espaço.
    5. Remove as palavras de parada (stopwords) do português.

    Parâmetros:
    texto (str): O texto a ser tratado.

    Retorna:
    str: O texto tratado.

    """
    sem_acentos = unidecode(texto).lower().replace("'", "").replace("-", " ").strip() if pd.notnull(texto) else texto
    sem_espacos_duplos = re.sub(r'\s+', ' ', sem_acentos).strip()
    sw = stopwords.words('portuguese')
    texto_processado = ' '.join([palavra for palavra in sem_espacos_duplos.split(" ") if palavra not in sw])
    return texto_processado if texto_processado else None


def plota_nuvem_de_palavras(df_parametro, palavras_excluidas = None, empreendimento = True):
    df = df_parametro.copy()
    if empreendimento:
        contagem  = df.empreendimento.str.split().explode('empreendimento').apply(lambda x: trata_texto(x)).value_counts()
    else: 
        contagem = df['new_empreend'].str.split(',').explode().value_counts()

    contagem_dict = dict(zip(contagem.index, contagem.values))

    # Remove palavras em palavras_excluidas do dicionário de contagem
    if palavras_excluidas:
        for palavra in palavras_excluidas:
            contagem_dict.pop(palavra, None)

    wordcloud = WordCloud(random_state=65)
    wordcloud.generate_from_frequencies(frequencies = contagem_dict)

    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud, interpolation = 'bilinear') # plotagem da nuvem de palavras
    plt.axis('off') # remove as bordas
    plt.show()

    return contagem

################## FUNÇÕES NO NOTEBOOK DE TRATAMENTO DA BASE DE ACIDENTES ##################

def trata_df_emprend(df, keywords, mostrar_colunas_novas=True):
    """
    Esta função processa um DataFrame de empreendimentos, procurando por palavras-chave específicas nos empreendimentos e criando novas colunas com base nessas palavras-chave.

    Parâmetros:
    df (pandas.DataFrame): DataFrame de entrada que contém os dados dos empreendimentos.
    keywords (dict): Dicionário onde as chaves são as palavras-chave a serem procuradas e os valores são listas de sinônimos dessas palavras-chave.
    mostrar_colunas_novas (bool, opcional): Se True, retorna o DataFrame original com as novas colunas adicionadas. Se False, retorna o DataFrame sem as colunas auxiliares criadas durante o processamento.

    Retorna:
    pandas.DataFrame: DataFrame processado com as novas colunas adicionadas.

    """

    # Para cada palavra-chave e seus sinônimos
    for key, values in keywords.items():
        # Cria uma nova coluna no DataFrame para essa palavra-chave
        df[f'aux_emp_{key.lower().replace(" ", "_")}'] = df['empreendimento'].apply(
            lambda x: key if any(re.search(v, x, re.IGNORECASE) for v in values) else np.nan
        )

    # Cria uma nova coluna 'new_empreend' que é a junção das colunas auxiliares criadas
    df['new_empreend'] = df.filter(like='aux_emp_').apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

    # Se mostrar_colunas_novas for True, retorna o DataFrame com todas as colunas
    if mostrar_colunas_novas:
        return df

    # Se mostrar_colunas_novas for False, retorna o DataFrame sem as colunas auxiliares
    return df.loc[:, ~df.columns.str.startswith('aux')]


def calcular_taxas_acidentes(df):
    """
    Função para calcular as taxas de acidentes por modalidade de transporte.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo os dados de acidentes e população.

    Retorna:
    df (pandas.DataFrame): DataFrame atualizado com as taxas calculadas.
    """

    # Dicionário mapeando as colunas de taxa para as modalidades de transporte correspondentes.
    correspondencia = {
        'taxa_mun_mortes': 'total_mortes',
        'taxa_mun_pedestre_mortes': 'Pedestre_mortes',
        'taxa_mun_ciclista_mortes': 'Ciclista_mortes',
        'taxa_mun_motociclista_mortes': 'Motociclista_mortes',
        'taxa_mun_triciclo_mortes': 'Ocup_triciclo_motor_mortes',
        'taxa_mun_automovel_mortes': 'Ocup_automovel_mortes',
        'taxa_mun_caminhonete_mortes': 'Ocup_caminhonete_mortes',
        'taxa_mun_veiculo_pesado_mortes': 'Ocup_veic_transp_pesado_mortes',
        'taxa_mun_onibus_mortes': 'Ocup_onibus_mortes',
        'taxa_mun_outros_mortes': 'Outros_mortes',
        'taxa_mun_feridos': 'total_feridos',
        'taxa_mun_pedestre_feridos': 'Pedestre_feridos',
        'taxa_mun_ciclista_feridos': 'Ciclista_feridos',
        'taxa_mun_motociclista_feridos': 'Motociclista_feridos',
        'taxa_mun_triciclo_feridos': 'Ocup_triciclo_motor_feridos',
        'taxa_mun_automovel_feridos': 'Ocup_automovel_feridos',
        'taxa_mun_caminhonete_feridos': 'Ocup_caminhonete_feridos',
        'taxa_mun_veiculo_pesado_feridos': 'Ocup_veic_transp_pesado_feridos',
        'taxa_mun_onibus_feridos': 'Ocup_onibus_feridos',
        'taxa_mun_outros_feridos': 'Outros_feridos',
    }

    for taxa_col, num_col in correspondencia.items():
        # para evitar divisão por zero, se população for zero (há casos)
        df[taxa_col] = np.where(df['Populacao_populacao'] != 0, 
                                df[num_col] / df['Populacao_populacao'] * 100, 
                                0)
    return df

################## FUNÇÕES NO NOTEBOOK DE AGRUPAMENTO POR MUNICÍPIO E ANO ##################

def agrupar_base_empreendimentos(df_empreend, ano=True):
    """
    Esta função agrupa um DataFrame de empreendimentos por município (e opcionalmente por ano),
    calculando várias estatísticas para cada grupo.

    Parâmetros:
    df_empreend (pandas.DataFrame): DataFrame de entrada que contém os dados dos empreendimentos.
    ano (bool, opcional): Se True, o DataFrame será agrupado por município e ano. Se False, será agrupado apenas por município.

    Retorna:
    pandas.DataFrame: DataFrame agrupado com novas colunas calculadas.

    """
    df = df_empreend.copy()

    aux_emp_columns = df.filter(like='aux_emp').columns

    agg_dict = {
        'vlr_investimento': 'sum',
        'Populacao': 'mean',
        'pop_beneficiada': 'sum',
        'Código IBGE': 'size'
    }

    # Para calcular o total de investimentos em cada categoria de empreendimentos, vamos, para cada coluna aux_emp, adicionar
    # uma operação de agregação ao dicionário e criar uma nova coluna no DataFrame que é igual ao valor de investimento se a coluna aux_emp não for nula, e zero caso contrário
    # Em seguida, quando agruparmos as colunas vamos somar o valor de investimento em cada categoria
    for column in aux_emp_columns:
        agg_dict[column] = lambda x: x.notnull().sum()

        new_column_name = column + '_vlr_investimento'
        
        # Calcula a soma dos valores da coluna "vlr_investimento" onde a coluna aux_emp correspondente não é nula
        df[new_column_name] = df.apply(lambda row: row['vlr_investimento'] if pd.notnull(row[column]) else 0, axis=1)
        
        agg_dict[new_column_name] = 'sum'

    # Cria colunas dummy para cada valor distinto na coluna 'programa' - Em seguida, faremos o mesmo que fizemos
    # com as categorias de empreendimentos para obter o valor do investimento em cada programa
    programa_dummies = pd.get_dummies(df['programa'], prefix='programa')

    df = pd.concat([df, programa_dummies], axis=1)

    for column in programa_dummies.columns:
        agg_dict[column] = 'sum'
        
        new_column_name = column + '_vlr_investimento'
        
        # Calcula a soma dos valores da coluna "vlr_investimento" onde a coluna 'programa' corresponde a cada valor distinto
        df[new_column_name] = df.apply(lambda row: row['vlr_investimento'] if row['programa'] == column.replace('programa_', '') else 0, axis=1)
        
        agg_dict[new_column_name] = 'sum'
    
    if ano:
        colunas_agrupar = ['Código IBGE', 'ano']
    else:
        colunas_agrupar = ['Código IBGE']

    df_agrupado_municipio_empreend = df.groupby(colunas_agrupar).agg(agg_dict)

    df_agrupado_municipio_empreend.rename(columns={'Código IBGE': 'num_total_empreendimentos' }, inplace=True)

    # Calcula o valor de investimento e a população beneficiada per capita
    df_agrupado_municipio_empreend['vlr_investimento_per_capita'] = df_agrupado_municipio_empreend['vlr_investimento'] / df_agrupado_municipio_empreend['Populacao']
    df_agrupado_municipio_empreend['pop_beneficiada_per_capita'] = df_agrupado_municipio_empreend['pop_beneficiada'] / df_agrupado_municipio_empreend['Populacao']

    df_agrupado_municipio_empreend = df_agrupado_municipio_empreend.reset_index()

    return df_agrupado_municipio_empreend


def merge_dataframes_empreend_acidentes(df_agrupado_municipio_empreend, df_acidentes, merge_on = ['Código IBGE', 'ano']):
    """
    Função que realiza a junção de dois dataframes, um contendo dados de empreendimentos e outro contendo dados de acidentes.
    
    Parâmetros:
    df_agrupado_municipio_empreend (pandas.DataFrame): DataFrame contendo dados de empreendimentos agrupados por município.
    df_acidentes (pandas.DataFrame): DataFrame contendo dados de acidentes.
    merge_on (list, opcional): Lista de colunas para realizar a junção. Por padrão é ['Código IBGE', 'ano'].
    
    Retorna:
    df_merge_empreendimentos_acidentes (pandas.DataFrame): DataFrame resultante da junção 'outer'.
    df_merge_empreendimentos_acidentes_sem_nulo (pandas.DataFrame): DataFrame resultante da junção 'inner', contendo apenas os conjuntos município e ano
                                                                    que contém tanto algum valor de empreendimento quanto o número de acidentes.
    """
    df_merge_empreendimentos_acidentes = pd.merge(df_agrupado_municipio_empreend, df_acidentes, on=merge_on, how= 'outer')
    
    # Para evitar que o modelo seja treinado com muitos dados nulos, vamos também fazer um inner join para selecionar
    # apenas os conjuntos município e ano que contém tanto algum valor de empreendimento quanto o número de acidentes
    df_merge_empreendimentos_acidentes_sem_nulo = pd.merge(df_agrupado_municipio_empreend, df_acidentes, on=merge_on, how='inner')
    
    return df_merge_empreendimentos_acidentes, df_merge_empreendimentos_acidentes_sem_nulo