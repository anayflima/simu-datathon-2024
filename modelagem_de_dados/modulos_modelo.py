import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from linearmodels.panel import PanelOLS, compare, RandomEffects, PooledOLS
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import f as scipyStatsF
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os


"""
 ARQUIVO CONCENTRA AS FUNÇÕES DESENVOLVIDAS PARA A MODELAGEM DOS DADOS E EXTRAÇÃO DOS COEFICIENTES
 QUE REPRESENTAM A RELAÇÃO ENTRE A BASE DE EMPREENDIMENTOS E DE ACIDENTES. COMO AS FUNÇÕES FORAM UTILIZADAS
 PARA MODELAR OS DADOS PODE SER ENCONTRADO NOS SEGUINTES NOTEBOOKS:
    - analise_de_dados/analise_painel.ipynb
    - analise_de_dados/analise_coeficientes_modelo.ipynb
"""

################## FUNÇÕES NO NOTEBOOK DE MODELAGEM DOS DADOS ##################

def matrix_LI(matrix):
    """
    Seleciona colunas linearmente independentes de uma matriz de entrada.

    Esta função remove colunas com todos os valores zero, adiciona uma constante e, em seguida, 
    seleciona colunas linearmente independentes, começando com as duas primeiras colunas da matriz original.

    Parâmetros:
    matrix (pd.DataFrame): O DataFrame de entrada, contendo as colunas e um MultiIndex.

    Retorna:
    pd.DataFrame: Um novo DataFrame contendo apenas colunas linearmente independentes, 
    preservando o MultiIndex original.
    """
    # Removendo colunas com todos os valores zero
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    matrix = sm.add_constant(matrix).astype(float)

    # Selecionar as duas primeiras colunas da matriz original
    initial_columns = matrix.iloc[:, :2]
    remaining_matrix = matrix.iloc[:, 2:].values

    # Inicializa a matriz LI e lista de índices
    LI_matrix = initial_columns.values
    cols_li = initial_columns.columns.tolist()

    for i in range(remaining_matrix.shape[1]):
        # Adiciona uma coluna candidata
        nova_coluna = remaining_matrix[:, i].reshape(-1, 1)
        # Concatena a nova coluna à matriz de colunas LI
        temp_matrix = np.hstack([LI_matrix, nova_coluna])
        # Verifica o posto da nova matriz
        if np.linalg.matrix_rank(temp_matrix) > np.linalg.matrix_rank(LI_matrix):
            LI_matrix = temp_matrix
            cols_li.append(matrix.columns[i + 2])
        # Verificar se já temos o número máximo de colunas LI (igual ao número de linhas)
        if LI_matrix.shape[1] == LI_matrix.shape[0]:
            break

    # Convertendo de volta para DataFrame do Pandas com o MultiIndex original
    new_matrix = pd.DataFrame(LI_matrix, index=matrix.index, columns=cols_li)
    
    return new_matrix


def prepare_panel_data(df, selected_columns=[], check_rank=True):
    """
    Prepara os dados de um DataFrame para análise em painel.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem preparados.
    selected_columns : list, opcional
        Lista de colunas selecionadas para a análise. Se não fornecida, todas as colunas numéricas serão utilizadas.
    check_rank : bool, opcional
        Flag para indicar se a verificação do posto da matriz deve ser realizada.

    Retornos:
    --------
    df_panel : pd.DataFrame
        DataFrame preparado para análise em painel, com as colunas independentes, dependentes e eixos do painel selecionados.
    X : pd.DataFrame
        DataFrame contendo as variáveis independentes com uma constante adicionada.
    y1 : pd.Series
        Série contendo os valores da variável dependente 'total_mortes'.
    y2 : pd.Series
        Série contendo os valores da variável dependente 'total_feridos'.
    """

    # Fixos
    eixos_painel = ['Código IBGE', 'ano']
    y1_name = 'total_mortes'
    y2_name = 'total_feridos'

    if len(selected_columns) == 0:
        selected_columns = df.select_dtypes(include='number').columns.tolist()

    df_panel = df[pd.unique([y1_name, y2_name, *eixos_painel, *selected_columns]).tolist()].copy()
    df_panel['Código IBGE'] = df_panel['Código IBGE'].astype('category')
    df_panel.set_index(['Código IBGE', 'ano'], inplace=True)

    selected_columns = df_panel.columns
    x_columns_selected = selected_columns.drop([y1_name, y2_name]).tolist()

    X = df_panel[x_columns_selected]

    # Resolver problema de posto da matriz
    if check_rank:
        X = matrix_LI(X)
        selected_columns = [y1_name, y2_name, *X.columns.drop('const').tolist()]
        df_panel = df_panel[selected_columns]

    X = sm.add_constant(X).astype(float)
    y1 = df_panel[y1_name].astype(float)
    y2 = df_panel[y2_name].astype(float)

    return df_panel, X, y1, y2

def fit_panel_data_model(X, y, show_summary=True,drop_absorbed=True,cov_error_clusterized=False):
    """
    Ajusta um modelo de dados em painel usando o método de Mínimos Quadrados Ordinários (OLS).

    Parâmetros:
    ----------
    X : pd.DataFrame
        DataFrame contendo as variáveis independentes.
    y : pd.Series
        Série contendo a variável dependente.
    show_summary : bool, opcional
        Flag para indicar se o resumo do modelo deve ser exibido. O padrão é True.

    Retornos:
    --------
    model : PanelOLS
        Modelo ajustado de dados em painel.
    """
    # # Cria e ajusta o modelo de dados em painel com efeitos fixos para entidades
    model = PanelOLS(y, X, entity_effects=True,drop_absorbed=drop_absorbed)
    # if cov_error_clusterized:
    #     model = model.fit(cov_type="clustered", cluster_entity=True)
    # else:
    model = model.fit()
    
    # Exibe o resumo do modelo, se solicitado
    if show_summary:
        print(model.summary)
    
    return model


def chow_test(data, break_point, y_var, x_vars):
    """
    Realiza o teste de Chow para verificar a existência de uma quebra estrutural nos dados.

    Parâmetros:
    ----------
    data : pd.DataFrame
        DataFrame contendo os dados em painel com MultiIndex (Código IBGE e ano).
    break_point : int
        Ano que representa o ponto de quebra para o teste de Chow.
    y_var : str
        Nome da variável dependente no DataFrame.
    x_vars : list of str
        Lista dos nomes das variáveis independentes no DataFrame.

    Retorno:
    -------
    dict
        Dicionário contendo o valor do estatístico F e o p-valor do teste de Chow.
    """
    # Dividindo os dados em dois subconjuntos com base no ponto de quebra
    df_pre_break = data.loc[data.index.get_level_values('ano') < break_point]
    df_post_break = data.loc[data.index.get_level_values('ano') >= break_point]

    # Extraindo as variáveis dependente e independente para os dois subconjuntos
    y_pre, X_pre = df_pre_break[y_var], df_pre_break[x_vars]
    y_post, X_post = df_post_break[y_var], df_post_break[x_vars]

    # Adicionando uma constante (intercepto) aos modelos
    X_pre = sm.add_constant(X_pre)
    X_post = sm.add_constant(X_post)
    
    # Ajustando os modelos de regressão para os dois subconjuntos
    model_pre = PooledOLS(y_pre, X_pre,check_rank=False).fit()
    model_post = PooledOLS(y_post, X_post,check_rank=False).fit()
    
    # Ajustando o modelo de regressão para o conjunto completo de dados
    y, X = data[y_var], data[x_vars]
    X = sm.add_constant(X)
    model_full = PooledOLS(y, X,check_rank=False).fit()

    # Calculando os resíduos quadrados dos modelos
    RSS_pre = sum(model_pre.resids ** 2)
    RSS_post = sum(model_post.resids ** 2)
    RSS_full = sum(model_full.resids ** 2)

    # Calculando o número de parâmetros
    k = X_pre.shape[1]
    N_pre, N_post = X_pre.shape[0], X_post.shape[0]

    # Calculando o estatístico F do teste de Chow
    chow_stat = ((RSS_full - (RSS_pre + RSS_post)) / k) / ((RSS_pre + RSS_post) / (N_pre + N_post - 2 * k))
    df1 = k
    df2 = N_pre + N_post - 2 * k
    p_value = scipyStatsF.sf(chow_stat, df1, df2)

    return {"F-statistic": chow_stat, "p-value": p_value}



def run_chow_tests(data, y_var, x_vars):
    """
    Executa o teste de Chow para cada ano no intervalo de anos presente nos dados.

    Parâmetros:
    ----------
    data : pd.DataFrame
        DataFrame contendo os dados em painel com MultiIndex (Código IBGE e ano).
    y_var : str
        Nome da variável dependente no DataFrame.
    x_vars : list of str
        Lista dos nomes das variáveis independentes no DataFrame.

    Retorno:
    -------
    pd.DataFrame
        DataFrame contendo os resultados do teste de Chow para cada ano.
    """
    # Obtendo o ano inicial e final do índice 'ano'
    start_year = data.index.get_level_values('ano').min()
    end_year = data.index.get_level_values('ano').max()
    
    # Lista para armazenar os resultados do teste de Chow
    results = []
    
    # Iterando sobre os anos no intervalo especificado
    for year in range(int(start_year)+1, int(end_year)):
        # Executando o teste de Chow para o ano atual
        result = chow_test(data, year, y_var, x_vars)
        result['Year'] = year
        results.append(result)
    
    # Retornando um DataFrame com os resultados do teste de Chow para cada ano
    return pd.DataFrame(results)



def teste_BP(data, y_var, x_vars):
    """
    Executa o teste de Breusch-Pagan para heteroscedasticidade nos resíduos de um modelo OLS.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados.
        y_var (str): Nome da variável dependente no DataFrame.
        x_vars (list of str): Lista dos nomes das variáveis independentes no DataFrame.

    Returns:
        None: Imprime os resultados do teste de Breusch-Pagan, incluindo as estatísticas LM e F e seus valores p correspondentes.
    """
    # Extraindo as variáveis dependente e independente
    y = data[y_var]
    X = data[x_vars]

    # Adicionando uma constante (intercepto) ao modelo
    X = sm.add_constant(X)

    # Ajustando o modelo de regressão OLS
    # model_ols = sm.OLS(y, X).fit()
    model_ols = PooledOLS(y, X).fit()

    # Realizando o teste de Breusch-Pagan para verificar a heteroscedasticidade dos resíduos
    LM, LM_pv, F, F_pv = het_breuschpagan(model_ols.resids.values, model_ols.model.exog.dataframe)

    # Definindo rótulos para os valores retornados pelo teste
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

    # Exibindo os resultados do teste de Breusch-Pagan
    print(f"\n\nBreusch-Pagan Test for Total {y_var}:")
    print(dict(zip(labels, (LM, LM_pv, F, F_pv))))

    # Interpretando os resultados do teste de homoscedasticidade
    # A hipótese nula do teste é de homoscedasticidade (variância constante dos resíduos).
    # Se ambos os valores p (LM_pv e F_pv) forem maiores que 0,05, não rejeitamos a hipótese nula.
    # Caso contrário, rejeitamos a hipótese nula, indicando heteroscedasticidade no modelo Pooled.
    print(f"\nHipótese de Homoscedasticidade do modelo Pooled {'Não' if ((LM_pv > .05) and (F_pv > .05)) else ''} Rejeitada")


def plot_residuos(model, y_name):
    """
    Plota a distribuição dos resíduos e a relação entre resíduos e valores ajustados para um modelo de regressão.

    Parâmetros:
    ----------
    model : PanelOLS
        Modelo de dados em painel ajustado.
    y_name : str
        Nome da variável dependente para ser exibido nos títulos dos gráficos.

    """
    # Plota a distribuição dos resíduos
    plt.hist(model.resids, bins=50, edgecolor='k')
    plt.title(f'Distribuição dos Resíduos - {y_name}')
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.show()

    # Plota os resíduos versus os valores ajustados
    plt.scatter(model.fitted_values, model.resids, alpha=0.5)
    plt.title(f'Resíduos vs Valores Ajustados - {y_name}')
    plt.xlabel('Valores Ajustados')
    plt.ylabel('Resíduos')
    plt.show()

def VIF(X):
    """
    Esta função calcula o Fator de Inflação da Variância (VIF) para cada variável independente em um conjunto de dados.
    
    Parâmetros:
    X (pandas.DataFrame): DataFrame contendo as variáveis independentes.
    
    Retorna:
    vif_data (pandas.DataFrame): DataFrame contendo o nome de cada variável independente e seu respectivo VIF.
    """
    X_vif = sm.add_constant(X)  # Adicionar constante
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    return vif_data

def compare_random_fixed_panel(X, y):
    """
    Compara os modelos de efeitos fixos e aleatórios para um conjunto de dados em painel.

    Parâmetros:
    ----------
    X : pd.DataFrame
        DataFrame contendo as variáveis independentes.
    y : pd.Series
        Série contendo a variável dependente.

    """
    # Ajusta o modelo de efeitos fixos
    fe_model = PanelOLS(y, X, entity_effects=True,drop_absorbed=True).fit()
    
    # Ajusta o modelo de efeitos aleatórios
    re_model = RandomEffects(y, X).fit()
    
    # Compara os dois modelos
    comparison = compare({'Fixed Effects': fe_model, 'Random Effects': re_model})
    
    # Exibe a comparação dos modelos
    print(comparison)

    # Cálculo do teste de Hausman
    b = fe_model.params
    B = re_model.params
    v_b = fe_model.cov
    v_B = re_model.cov
    chi2 = np.dot((b - B).T, np.linalg.inv(v_b - v_B).dot(b - B))
    p_value = stats.chi2(b.size-1).sf(chi2)
    
    print(f"Teste de Hausman\nEstatística chi2: {chi2}\nP-valor: {p_value}")
    print(f"Hipótese Nula: {'Não' if p_value > 0.05 else ''} Rejeitada (Efeitos aleatórios são preferidos)")

def fit_random_effect(X, y, show_summary=True):
    """
    Ajusta um modelo de Efeitos Aleatórios aos dados fornecidos.

    Parâmetros:
    -----------
    X : pandas.DataFrame ou numpy.ndarray
        Matriz de variáveis independentes.
    y : pandas.Series ou numpy.ndarray
        Vetor da variável dependente.
    show_summary : bool, opcional
        Se True, exibe o resumo do modelo ajustado. O padrão é True.

    Retorna:
    --------
    fitted_model : RandomEffectsResults
        O modelo ajustado de Efeitos Aleatórios.
    """
    # Modelo de Efeitos Aleatórios
    fitted_model = RandomEffects(y, X).fit()
    if show_summary:
        print(fitted_model.summary)
    return fitted_model

def salva_coeficientes(model):
    """
    Salva os coeficientes de um modelo estatístico em um arquivo CSV.

    Parâmetros:
    model (Statsmodels object): O modelo estatístico.

    Retorna:
    None
    """
    variavel_dependente = pd.DataFrame(model.summary.tables[0][0]).iloc[1]

    variavel_dependente = str(variavel_dependente.values[0])

    summary_df = pd.DataFrame(model.summary.tables[1])

    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df[1:]

    summary_df.rename(columns={summary_df.columns[0]: variavel_dependente}, inplace=True)

    if not os.path.exists('resultados_modelo'):
            os.makedirs('resultados_modelo')

    summary_df.to_csv(f'resultados_modelo/geral_{variavel_dependente}.csv', index=False)

def get_top_municip(df_empreend, pop=100000):
    """
    Filtra os municípios com população maior que um valor especificado e retorna os principais municípios
    ordenados pela população.

    Parâmetros:
    -----------
    df_empreend : pandas.DataFrame
        DataFrame contendo os dados dos municípios, incluindo a população e o código IBGE.
    
    pop : int, opcional
        O limite mínimo de população para considerar um município como "principal". O valor padrão é 100000.

    Retornos:
    ---------
    df_top_munic : pandas.DataFrame
        DataFrame contendo os municípios com população acima do limite especificado, ordenados pela população
        em ordem decrescente. Inclui as colunas 'Código IBGE', 'Município' e 'Populacao'.
    
    id_cidades_principais : numpy.ndarray
        Array contendo os códigos IBGE dos municípios selecionados.
    """
    # Filtra o DataFrame para incluir apenas os municípios com população maior que 'pop'
    df_top_munic = df_empreend[df_empreend.Populacao > pop][['Código IBGE', 'Município', 'Populacao']]
    
    # Agrupa os dados pelos códigos IBGE e nome dos municípios, e seleciona a maior população registrada para cada grupo
    df_top_munic = df_top_munic.groupby(['Código IBGE', 'Município']).max()['Populacao']
    
    # Ordena os municípios pela população em ordem decrescente
    df_top_munic = df_top_munic.sort_values(ascending=False).reset_index()
    
    # Extrai os códigos IBGE dos municípios selecionados para um array
    id_cidades_principais = df_top_munic['Código IBGE'].values
    
    # Retorna o DataFrame dos principais municípios e o array dos códigos IBGE
    return df_top_munic, id_cidades_principais

def get_capitais(df_empreend):
    """
    Filtra os municípios que são capitais de estado e retorna esses municípios.

    Parâmetros:
    -----------
    df_empreend : pandas.DataFrame
        DataFrame contendo os dados dos municípios, incluindo uma coluna que indica se o município é uma capital.
    
    Retornos:
    ---------
    df_capitais : pandas.DataFrame
        DataFrame contendo os municípios que são capitais de estado. Inclui as colunas 'Código IBGE' e 'Município'.
    
    id_cidades_principais : numpy.ndarray
        Array contendo os códigos IBGE dos municípios que são capitais de estado.
    """
    # Filtra o DataFrame para incluir apenas os municípios que são capitais
    df_capitais = df_empreend[df_empreend['mun_CAPITAL'].astype(str) == 'S'][['Código IBGE', 'Município']]
    
    # Remove duplicatas e valores ausentes
    df_capitais = df_capitais.drop_duplicates().dropna().reset_index(drop=True)
    
    # Extrai os códigos IBGE dos municípios selecionados para um array
    id_cidades_principais = df_capitais['Código IBGE'].values
    
    # Retorna o DataFrame dos municípios capitais e o array dos códigos IBGE
    return df_capitais, id_cidades_principais

def matrix_LI_com_QR(matrix):
    """
    Esta função realiza a decomposição QR em uma matriz e retorna uma nova matriz
    contendo apenas as colunas linearmente independentes (LI).
    
    Parâmetros:
    matrix (pandas.DataFrame): DataFrame contendo os dados a serem processados.
    
    Retorna:
    LI_matrix (pandas.DataFrame): DataFrame contendo apenas as colunas linearmente independentes da matriz original.
    """
    # Removendo colunas com todos os valores zero
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    matrix = sm.add_constant(matrix).astype(float)

    # Realiza a decomposição QR
    Q, R = np.linalg.qr(matrix.values)

    # Seleciona as colunas com base nos elementos da diagonal de R
    cols_li_indices = np.where(np.abs(np.diag(R)) > 1e-10)[0]
    
    # Seleciona os nomes das colunas correspondentes
    cols_li = matrix.columns[cols_li_indices]
    
    # Cria a nova matriz apenas com as colunas LI
    LI_matrix = matrix[cols_li]

    return LI_matrix

def matrix_LI_com_SVD(matrix):
    """
    Esta função realiza a decomposição em valores singulares (SVD) em uma matriz
    e retorna uma nova matriz contendo apenas as colunas linearmente independentes (LI).
    
    Parâmetros:
    matrix (pandas.DataFrame): DataFrame contendo os dados a serem processados.
    
    Retorna:
    LI_matrix (pandas.DataFrame): DataFrame contendo apenas as colunas linearmente independentes da matriz original.
    
    """
    # Removendo colunas com todos os valores zero
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    matrix = sm.add_constant(matrix).astype(float)

    # Realiza a decomposição SVD
    U, S, Vt = np.linalg.svd(matrix.values)
    
    # Seleciona as colunas com base na matriz Vt e os valores singulares
    # Verifica as colunas em Vt correspondentes a valores singulares não nulos
    cols_li_indices = np.where(np.abs(S) > 1e-10)[0]
    
    # Seleciona os nomes das colunas correspondentes
    cols_li = matrix.columns[cols_li_indices]
    
    # Cria a nova matriz apenas com as colunas LI
    LI_matrix = matrix[cols_li]

    return LI_matrix


################## FUNÇÕES NO NOTEBOOK DE ANÁLISE DE COEFICIENTES ##################


def processa_resultados(df_resultados_parametro):
    """
    Processa o DataFrame de resultados dos coeficientes do modelo, separando-os em
    resultados válidos e não válidos, de acordo com o p-valor,
    e ainda divide os resultados válidos em negativos e positivos, de acordo com o coeficiente.
    Parâmetros:
    df_resultados (pd.DataFrame): DataFrame contendo os resultados a serem processados.

    Retorna:
    df_resultados_validos_negativos (pd.DataFrame): DataFrame contendo os resultados válidos e negativos.
    df_resultados_validos_positivos (pd.DataFrame): DataFrame contendo os resultados válidos e positivos.
    df_resultados_nao_validos (pd.DataFrame): DataFrame contendo os resultados não válidos.
    """
    df_resultados = df_resultados_parametro.copy()
    df_resultados[df_resultados.columns[1:]] = df_resultados[df_resultados.columns[1:]].apply(pd.to_numeric)
    df_resultados.set_index(df_resultados.columns[0], inplace=True)
    
    # ordena resultados de acordo com o impacto - os que precisam de menos investimento para efeito
    coluna_investimento_impacto = 'Investimento para 1 de impacto'
    df_resultados[coluna_investimento_impacto] = 1/df_resultados['Parameter']
    df_resultados = df_resultados.reindex(df_resultados[coluna_investimento_impacto].abs().sort_values().index)

    df_resultados_nao_validos = df_resultados[df_resultados['P-value'] > 0.05]
    df_resultados_validos = df_resultados[df_resultados['P-value'] < 0.05]
    df_resultados_validos_negativos = df_resultados_validos[df_resultados_validos['Parameter'] < 0]
    df_resultados_validos_positivos = df_resultados_validos[df_resultados_validos['Parameter'] >= 0]
    return df_resultados_validos_negativos, df_resultados_validos_positivos, df_resultados_nao_validos


def calcular_impacto_total(df_resultados_validos, variaveis_valores):
    """
    Calcula o impacto total no número de acidentes de um conjunto de investimentos
    com base nos coeficientes de variáveis explicativas.

    Parâmetros:
    df_resultados_validos (pandas.DataFrame): Um dataframe que contém os resultados válidos. 
                                              Espera-se que este dataframe tenha uma coluna chamada 'Parameter' 
                                              que contém os coeficientes das variáveis explicativas.

    variaveis_valores (dict): Um dicionário onde as chaves são as variáveis explicativas e os valores 
                              são os respectivos valores de investimento.

    Retorna:
    impacto_total (float): O impacto total calculado como a soma dos produtos do coeficiente de cada 
                           variável explicativa e seu respectivo valor de investimento.
    """
    impacto_total = 0
    for variavel_explicativa, valor_investimento in variaveis_valores.items():
        coeficiente = df_resultados_validos.loc[variavel_explicativa, 'Parameter']
        impacto = coeficiente * valor_investimento
        impacto_total += impacto
    return impacto_total