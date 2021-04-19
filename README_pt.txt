Uma breve descrição dos arquivos e pastas.

Arquivos ".ipynb" no Diretório principal: uma observação importante é que todos os notebooks tem um número indicador de prioridade em seu nome, e.g., "(2)". Os números indicam em que ordem os notebooks devem ser executados. Por exemplo, um notebook "(2)" deve ser executado antes de um notebook "(3)". Notebooks com a mesma numeração não têm ordem de prioridade. Vamos descrever brevemente qual a funcionalidade dos notebooks de acordo com sua numeração:
    (0) Nesse notebook treinamos um modelo não supervisionado para descobrirmos quais conjunto de 2, 3, e 4 palavras devem ser consideradas tokens únicos;
    (1) Nesses notebooks treinamos quatro representações para textos: word2vec, doc2vec, bert, e tfidf;
    (2) Nesses notebooks utilizamos as representações já aprendidas para extrair as features do conjunto de dados rotulado e já prepará-lo para uso;
    (3) Nesse notebook utilizamos grid search para escolher os melhores valores para os hiperparâmetros dos classificadores;
    (4) Nesse notebook treinamos e avaliamos os modelos finais e otimizados. Além disso geramos gráficos e tabelas para o artigo;
    (5) Nesse notebook obtemos resultados de interpretabilidade;
    
Arquivos ".py" Diretório principal: esses arquivos foram utilizados como auxiliares enquanto rodamos os notebooks, afim de tornar tudo mais enxuto e organizado.
    - packages.py: arquivo que abre todos os pacotes e funções utilizadas no artigo;
    - tokenizer.py: arquivo que contém o tokenizador utilizados em conjunto com word2vec, doc2vec, e tfidf;
    - clean_functions.py: arquivo que contém as funções utilizadas para limpar os textos;
    - random_state.py: arquivo que rodamos de forma a garantir uma semente fixa e resultados reprodutíveis;
    - fit_models.py: arquivo para o treinamento de modelos;
    
Pastas no Diretório principal:
    - data: contémos as bases de dados utilizadas no artigo;
    - models: contém todos os modelos treinados na confecção do artigo;
    - plots: contém os gráficos gerados na confecção do artigo;
    - hyper: contém tabelas com melhores valores para os hiperparâmetros;