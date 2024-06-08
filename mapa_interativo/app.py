import os
import pandas as pd
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from modulos_mapa_interativo import plot_mapa_interativo

pasta_dados = './dados/'

df_agrupado_merge = pd.read_csv(os.path.join(pasta_dados, 'merge_agrupamento_municipio_ano.csv'))


def modify_doc(doc):
    plot = plot_mapa_interativo(df_agrupado_merge)
    
    doc.add_root(plot.get_root(doc))
    doc.title = 'Mapa Empreendimentos vs Acidentes'

apps = {'/': Application(FunctionHandler(modify_doc))}

server = Server(apps, port=8080)
server.start()

if __name__ == '__main__':
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()