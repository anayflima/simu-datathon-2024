import os
import pandas as pd
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from modulos_mapa_interativo import plot_mapa_interativo

current_path = os.getcwd()

if 'mapa_interativo' in current_path:
    pasta_dados = '../dados/'
else:
    pasta_dados = './dados/'

df_municipios = pd.read_csv(os.path.join(pasta_dados, 'simu_carteira_municipios.csv'))
df_agrupado = pd.read_csv(os.path.join(pasta_dados, 'tratados/agrupamento_empreend_acidentes_por_municipio_e_ano_sem_nulos.csv'))

df_agrupado_merge = df_agrupado.merge(df_municipios, on='Código IBGE', how='left')

def modify_doc(doc):
    plot = plot_mapa_interativo(df_agrupado_merge)
    
    doc.add_root(plot.get_root(doc))

apps = {'/': Application(FunctionHandler(modify_doc))}

server = Server(apps, port=5000)
server.start()

if __name__ == '__main__':
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()