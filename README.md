# DATATHON DE MOBILIDADE URBANA 2024 

**Repositório da competição DATATHON DE MOBILIDADE URBANA 2024**

## Participantes:
- Ana Yoon Faria de Lima
- Richard Sousa Antunes

## Apresentação em vídeo e slides

- Apresentação: Apresentacao_Datathon.pdf

- Link para o vídeo: 
https://www.youtube.com/watch?v=n7Tc9D83uS0

## Estrutura de pastas e arquivos

Este projeto está dividido em 6 pastas:

- dados
- analise_de_dados
- tratamento_de_dados
- modelagem_de_dados
- mapa_interativo
- extra


**Dados** 

A pasta dados contém as bases utilizadas no projeto:
- Bases de acidentes de transportes baixadas do SIMU:
    - https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIMU/temas/simu-acidentes-transportes-mun-T.zip
    - https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIMU/bases_dados/FERIDOS/simu-feridos-mun-T.zip
    - https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIMU/bases_dados/MORTES/simu-mortes-mun-T.zip
- Base da PEMOB baixadas do SIMU:
    - https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIMU/bases_dados/PEMOB/simu-pemob-mun_T.zip
- Base de empreendimentos baixadas do SIMU:
    - https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIMU/bases_dados/CARTEIRA/simu-carteira-mun-T.zip
- Base de municípios baixadas do SIMU:
    - https://pcdas.icict.fiocruz.br/wp-content/uploads/2024/05/simu_carteira_municipios.csv
- Base de População baixadas do SIMU:
    - https://pcdas.icict.fiocruz.br/wp-content/uploads/2024/05/simu_carteira_populacao.csv
- Base Extra resultante do notebook da pasta *extras*
- Pasta tratados (contém as bases filtradas e agrupadas).

**Analise de dados**

analise_de_dados contém os seguintes notebooks.
- (analise_exploratoria_simu.ipynb): contém a análise exploratória inicial que deu início aos primeiros insights para o projeto.
- (analise_inicial_correlacao.ipynb): exibe a análise inicial das correlações entre as variáveis, contribuindo nos insights e possíveis caminhos a seguir, 
- (sao_paulo_empreendimentos_versus_acidentes.ipynb): mostra a correlação dos empreendimenteos em São Paulo com os acidentes no trânsito a fim de mostrar possíveis peculiaridades desta grande cidade.

**Tratamento de dados**

tratamento_de_dados contém os códigos usados para fazer os tratamento das bases de dados.
- (agrupamento_municipio_ano.ipynb): combina as bases de acidentes e empreendimentos e agrupa por município e ano.
- (tratamento_acidentes.ipynb): realiza o tratamento na base de acidentes.
- (tratamento_empreendimentos.ipynb): trata informações de empreendimentos com tratamento de textos e agrupa em categorias de emprendimentos.
- (tratamento_nulos.ipynb): trata os nulos da base de empreendimentos.
- (modulos_tratamento.py): compilado de funções usado nos notebooks de tratamento acima.

**Modelagem de dados**

modelagem_de_dados contém:
- (analise_painel.ipynb): implementação de modelos sobre os dados tratados, além da realização dos testes estatísticos pertinentes ao contexto a fim de validar o processo e para escolher o modelo mais adequado.
- (analise_coeficientes_modelo.ipynb): realiza a análise dos coeficientes obtidos no modelo escolhido.
- (modulos_modelo.py): compilado de funções utilizado nos notebooks acima.

**Mapa interativo**

mapa_interativo contém
- (mapa_geoespacial_interativo.ipynb): notebook com o mapa interativo que mostra os dados ao longo do mapa do Brasil, o qual permite visualizar de forma interativa diferentes variáveis ao longo espaço e tempo.
- (modulos_mapa_interativo.py): compilado de funções utilizada no notebook acima.

**Extras**

extras contém:
- (codigos_extras_nao_selecionados.ipynb): notebook com os algoritmos avançados que foram inicialmente considerados para tratar os empreendimentos, porém pela sua complexidade e baixa reprodutibilidade, forma descartados.