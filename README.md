# DATATHON DE MOBILIDADE URBANA 2024 

**Repositório da competição DATATHON DE MOBILIDADE URBANA 2024**

## Participantes
- Ana Yoon Faria de Lima
- Richard Sousa Antunes

## Apresentação em vídeo e slides

- Relatório: [Relatório_SIMU_Datathon_2024.pdf](https://github.com/anayflima/simu-datathon-2024/blob/main/Relatório_SIMU_Datathon_2024.pdf)

- Apresentação: [Apresentacao_Datathon.pdf](https://github.com/anayflima/simu-datathon-2024/blob/main/Apresentacao_Datathon.pdf)

- Link para o vídeo: 
https://www.youtube.com/watch?v=n7Tc9D83uS0


## Requisitos

Para clonar esse repositório e tê-lo em uma pasta local, pode-se executar o seguinte comando:

```
git clone https://github.com/anayflima/simu-datathon-2024
```

A execução dos notebooks e funções presentes nesse repositório exigem a instalação de algumas bibliotecas do Python.
As principais utilizadas foram listadas no arquivo requirements.txt, na pasta root, e podem ser instaladas com o seguinte comando:

```
pip install -r requirements.txt
```


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

**Bases originais:**

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

**Bases geradas no projeto:**

- Base extra resultante do notebook da pasta *extras*
    - https://github.com/anayflima/simu-datathon-2024/blob/main/dados/teste_empreendimentos_codigo_corretor.csv
- Subpasta tratados (contém as bases resultantes dos tratamentos e agrupamentos)
    - https://github.com/anayflima/simu-datathon-2024/tree/main/dados/tratados

**Análise de dados**

analise_de_dados contém os seguintes notebooks:
- *analise_exploratoria_simu.ipynb*: contém a análise exploratória inicial que deu início aos primeiros insights para o projeto.
- *analise_inicial_correlacao.ipynb*: exibe a análise inicial das correlações entre as variáveis, contribuindo nos insights e possíveis caminhos a seguir.
- *sao_paulo_empreendimentos_versus_acidentes.ipynb*: mostra a relação dos empreendimentos em São Paulo com os acidentes no trânsito a fim de ilustrar possíveis peculiaridades desta grande cidade.

**Tratamento de dados**

tratamento_de_dados contém os códigos usados para fazer os tratamentos das bases de dados:
- *tratamento_nulos.ipynb*: trata os nulos da base de empreendimentos.
- *tratamento_empreendimentos.ipynb*: trata informações de empreendimentos com tratamento de textos e agrupa em categorias de emprendimentos.
- *tratamento_acidentes.ipynb*: realiza o tratamento na base de acidentes.
- *agrupamento_municipio_ano.ipynb*: combina as bases de acidentes e empreendimentos e agrupa por município e ano.
- *modulos_tratamento.py*: compilado das funções usadas nos notebooks de tratamento e agrupamento.

Os notebooks de tratamento devem ser executados na ordem apresentada acima, pois alguns notebooks esperam como entrada o resultado do notebook anterior.

**Modelagem de dados**

modelagem_de_dados contém:
- *analise_painel.ipynb*: implementa modelos de regressão para dados em painel sobre os dados tratados, além de realizar  testes estatísticos pertinentes ao contexto a fim de validar o processo e escolher o modelo mais adequado.
- *analise_coeficientes_modelo.ipynb*: realiza a análise dos coeficientes obtidos no modelo escolhido.
- *modulos_modelo.py*: compilado das funções utilizadas nos notebooks de modelagem.

**Mapa interativo**

mapa_interativo contém:
- *mapa_geoespacial_interativo.ipynb*: notebook com o mapa interativo que mostra os dados ao longo do mapa do Brasil, o qual permite visualizar de forma interativa diferentes variáveis de empreendimentos e acidentes por municípios e anos.
- *app.py*, *modulos_mapa_interativo.py*, *fly.toml*, *Procfile*, *requirements.txt*: arquivos utilizados para disponibilizar o mapa interativo em um servidor.

O mapa foi subido em um servidor e pode ser acessado pelo link:

- https://mapa-empreendimentos-acidentes.fly.dev/

Há duas formas possíveis de se executar o mapa interativo localmente:

1 - rodando o notebook *mapa_geoespacial_interativo.ipynb*

2 - executando o arquivo *app.py* dentro da pasta mapa_interativo:

```
cd mapa_interativo
python3 app.py
```

Ambas as opções irão rodar o mapa em uma porta no localhost.

**Extras**

Extras contém:
- *codigos_extras_nao_selecionados.ipynb*: notebook com os algoritmos avançados que foram inicialmente considerados para tratar o nome dos empreendimentos, porém foram descartados pela sua complexidade e baixa reprodutibilidade.