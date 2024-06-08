import numpy as np
import hvplot.pandas
import panel as pn
from bokeh.models import HoverTool


def plot_mapa_interativo(df):
    """
    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo os dados a serem plotados, resultado do agrupamento feito da base de empreendimentos e acidentes por município e ano.
    Deve conter colunas de longitude, latitude, população, acidentes, valores de investimento nos empreendimentos, entre outros.
    
    Funcionalidade:
    A função permite ao usuário selecionar variáveis específicas para empreendimentos e acidentes.
    Além disso, permite filtrar os dados com base na população, se o município é uma capital, a unidade federativa e o ano.
    
    O gráfico resultante é um mapa de pontos geográficos onde o tamanho do ponto é proporcional ao
    valor da variável de empreendimentos e a cor do ponto é determinada pelo valor da variável de acidentes.
    O gráfico é interativo, permitindo ao usuário explorar os dados de maneira mais detalhada.
    
    Retorna:
    Um painel interativo contendo o gráfico e os widgets para seleção de variáveis e filtragem de dados.
    """
    colunas_empreendimentos = df.filter(like='vlr_investimento').columns.tolist()
    colunas_acidentes = df.columns[df.columns.str.contains('mortes|feridos')].tolist()
    variavel_empreendimentos_select = pn.widgets.Select(name='Empreendimentos', options=colunas_empreendimentos, value='vlr_investimento', width=320)
    variavel_acidentes_select = pn.widgets.Select(name='Acidentes', options=colunas_acidentes, value='total_mortes', width=320)

    empreendimentos_min_input = pn.widgets.FloatInput(name=f'Mínimo Empreendimentos', value=df[variavel_empreendimentos_select.value].min(), width=150)
    empreendimentos_max_input = pn.widgets.FloatInput(name=f'Máximo Empreendimentos', value=df[variavel_empreendimentos_select.value].max(), width=150)
    acidentes_min_input = pn.widgets.FloatInput(name=f'Mínimo Acidentes', value=df[variavel_acidentes_select.value].min(), width=150)
    acidentes_max_input = pn.widgets.FloatInput(name=f'Máximo Acidentes', value=df[variavel_acidentes_select.value].max(), width=150)
    variavel_populacao = 'Populacao'

    pop_min, pop_max = np.log10(df[variavel_populacao].min()), np.log10(df[variavel_populacao].max())
    pop_range_slider = pn.widgets.RangeSlider(start=pop_min, end=pop_max, value=(pop_min, pop_max), step=0.1, name='População (escala 10^x)')
    pop_min_input = pn.widgets.IntInput(name=f'População Mínima', value=int(10**pop_min), width=100)
    pop_max_input = pn.widgets.IntInput(name=f'População Máxima', value=int(10**pop_max), width=100)
    capital_filter = pn.widgets.Select(name='Município é capital', options=['Todos', 'S', 'N'], value='Todos', width=320)
    ano_min, ano_max = df['ano'].min(), df['ano'].max()
    ano_range_slider = pn.widgets.RangeSlider(start=ano_min, end=ano_max, value=(ano_min, ano_max), step=1, name='Ano')
    regiao_filter = pn.widgets.MultiChoice(name='Região', options=df['Região'].unique().tolist(),
                                           value=df['Região'].unique().tolist(), width=600)
    uf_filter = pn.widgets.MultiChoice(name='Unidade Federativa', options=df['uf_SIGLA_UF'].unique().tolist(),
                                       value=df['uf_SIGLA_UF'].unique().tolist(), width=600)

    df_parametro = df.copy()
    df_parametro[variavel_populacao] = df_parametro[variavel_populacao].astype(int)

    @pn.depends(variavel_empreendimentos_select.param.value, watch=True)
    def update_empreendimentos_min_max(variavel_empreendimentos):
        empreendimentos_min_input.value = df[variavel_empreendimentos].min()
        empreendimentos_max_input.value = df[variavel_empreendimentos].max()

    @pn.depends(variavel_acidentes_select.param.value, watch=True)
    def update_acidentes_min_max(variavel_acidentes):
        acidentes_min_input.value = df[variavel_acidentes].min()
        acidentes_max_input.value = df[variavel_acidentes].max()

    # Função chamada quando as variáveis de empreendimentos e acidentes são alteradas com base na seleção do usuário.
    @pn.depends(variavel_empreendimentos_select.param.value, variavel_acidentes_select.param.value)
    def update_plot(variavel_empreendimentos, variavel_acidentes):
        variavel_empreendimentos_normalizado = variavel_empreendimentos + '_normalizado'

        @pn.depends(pop_range_slider.param.value, watch=True)
        def update_inputs(value):
            pop_min_input.value = int(10**value[0])
            pop_max_input.value = int(10**value[1])

        @pn.depends(pop_min_input.param.value, watch=True)
        def update_slider_min(value):
            pop_range_slider.value = (np.log10(value), pop_range_slider.value[1])

        @pn.depends(pop_max_input.param.value, watch=True)
        def update_slider_max(value):
            pop_range_slider.value = (pop_range_slider.value[0], np.log10(value))
        
        # Quando o valor do filtro regiao_filter muda, atualiza valores e opções disponíveis de UF de acordo com a região selecionada
        @pn.depends(regiao_filter.param.value, watch=True)
        def update_uf_filter(regiao_value):
            uf_filter.options = df[df['Região'].isin(regiao_value)]['uf_SIGLA_UF'].unique().tolist()

            uf_filter.value = df[df['Região'].isin(regiao_value)]['uf_SIGLA_UF'].unique().tolist()

        # Função chamada quando algum valor dos filtros é alterado pelo usuário.
        # @pn.depends(pop_range_slider.param.value, capital_filter.param.value, uf_filter.param.value, ano_range_slider.param.value, regiao_filter.param.value)
        @pn.depends(pop_range_slider.param.value, capital_filter.param.value, uf_filter.param.value, ano_range_slider.param.value, regiao_filter.param.value, empreendimentos_min_input.param.value, empreendimentos_max_input.param.value, acidentes_min_input.param.value, acidentes_max_input.param.value)
        # def filtered_plot(pop_range, capital_value, uf_value, ano_range, regiao_value):
        def filtered_plot(pop_range, capital_value, uf_value, ano_range, regiao_value, empreendimentos_min, empreendimentos_max, acidentes_min, acidentes_max):
            df_filtered = df_parametro[(df_parametro[variavel_populacao] >= 10**pop_range[0]) & (df_parametro[variavel_populacao] <= 10**pop_range[1])]
            df_filtered = df_filtered[(df_filtered[variavel_empreendimentos] >= empreendimentos_min) & (df_filtered[variavel_empreendimentos] <= empreendimentos_max)]
            df_filtered = df_filtered[(df_filtered[variavel_acidentes] >= acidentes_min) & (df_filtered[variavel_acidentes] <= acidentes_max)]
            if capital_value != 'Todos':
                df_filtered = df_filtered[df_filtered['mun_CAPITAL'] == capital_value]
            df_filtered = df_filtered[df_filtered['Região'].isin(regiao_value)]
            df_filtered = df_filtered[df_filtered['uf_SIGLA_UF'].isin(uf_value)]
            df_filtered = df_filtered[(df_parametro['ano'] >= ano_range[0]) & (df_parametro['ano'] <= ano_range[1])]
            colunas_sum = colunas_empreendimentos+colunas_acidentes+['pop_beneficiada', 'num_total_empreendimentos',]
            colunas_mean = ['Populacao','densidade_populacional']
            colunas_first = ['uf_SIGLA_UF', 'mun_MUNNOME', 'mun_LONGITUDE','mun_LATITUDE', 'mun_CAPITAL','Região', 'mun_AREA', 'mun_FRONTEIRA', 'mun_AMAZONIA']

            agg_dict = {col: 'sum' for col in colunas_sum}
            agg_dict.update({col: 'mean' for col in colunas_mean})
            agg_dict.update({col: 'first' for col in colunas_first})

            # Agrupa dados pelo Código IBGE, para agrupar os anos selecionados em um único ponto
            df_filtered = df_filtered.groupby('Código IBGE').agg(agg_dict).reset_index()

            valor = df_filtered[variavel_empreendimentos].median()

            # Se a mediana for zero, substitui pelo valor da média. Isso é feito ara evitar que scale_factor seja igual a infinito
            if valor == 0:
                valor = df_filtered[variavel_empreendimentos].mean()

            # Se a média também for zero, define o scale_factor como 1
            # Caso contrário, calcula o scale_factor com base no valor
            scale_factor = 1 if valor == 0 else np.power(10.0, -np.floor(np.log10(valor)))

            df_filtered[variavel_empreendimentos_normalizado] = df_filtered[variavel_empreendimentos] * scale_factor
     
            hover_tool = HoverTool(tooltips=[
                ('Nome do Município', '@mun_MUNNOME'),
                (variavel_acidentes, '@{'+ variavel_acidentes +'}{0,0}'),  # formata como inteiro, para mostrar 1,000,000 ao inves de 1e+6, por exemplo
                (variavel_empreendimentos, '@{'+ variavel_empreendimentos +'}{0,0}'), 
                ('População', '@{'+ variavel_populacao +'}{0,0}') 
            ])

            return df_filtered.hvplot.points('mun_LONGITUDE', 'mun_LATITUDE',
                                             size=variavel_empreendimentos_normalizado,
                                             geo=True,
                                             tiles='OSM',
                                             color=variavel_acidentes,
                                             # cmap='magma',
                                             tools=[hover_tool],
                                             hover_cols=['mun_MUNNOME', variavel_acidentes, variavel_empreendimentos, variavel_populacao],
                                             height=700, width=700,
                                             title="Relação entre investimentos em empreendimentos e acidentes no trânsito\n"
                                             ).opts(legend_position='bottom_right')

        
        explicacao = "### O raio dos pontos é proporcional ao valor da variável de empreendimentos.\n### A cor é mais escura quanto maior o valor da variável de acidentes"
        
        return pn.Row(filtered_plot, pn.Column(pn.pane.Markdown(f"{explicacao}"), variavel_empreendimentos_select, pn.Row(empreendimentos_min_input, empreendimentos_max_input), variavel_acidentes_select, pn.Row(acidentes_min_input, acidentes_max_input), pn.Row(pop_min_input, pop_max_input, pop_range_slider), capital_filter, ano_range_slider, regiao_filter, uf_filter))

    return pn.Row(update_plot)
