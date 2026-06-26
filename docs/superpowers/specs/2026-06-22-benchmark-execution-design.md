# Especificacao de Design: Fase de Simplificacao da Execucao em `validation/benchmark.py`

## Contexto

Depois das fases de `validation_core`, `validator`, `model_loader` e compatibilidade legada, `validation/benchmark.py` continua sendo um dos pontos mais acoplados do modulo `validation/`.

Hoje, `ModelBenchmark.run_benchmark()` mistura:

- loop de iteracao sobre modelos
- instanciacao de `ModelValidator`
- execucao de validacao por modelo
- tratamento de erro por modelo
- preenchimento de `self.results`
- criacao do `comparison_df`
- decisao de persistencia final

Esse metodo ja representa um recorte natural para a proxima fase, porque concentra a orquestracao principal do benchmark.

## Problema

Nesta fase, a execucao de benchmark ainda e menos simples e menos testavel do que o restante do fluxo porque:

- a validacao de um unico modelo esta embutida dentro do loop principal
- o tratamento de sucesso e erro por modelo nao tem fronteira propria
- `run_benchmark()` faz mais coisas do que precisa para ser testado isoladamente
- ainda nao existe um ponto pequeno e reutilizavel para representar a execucao por modelo

## Objetivos

- simplificar `ModelBenchmark.run_benchmark()`
- extrair a execucao de um unico modelo para um helper interno pequeno
- preservar o contrato publico de `run_benchmark()`
- manter `self.results` e `self.comparison_df` como estado publico do objeto
- aumentar a testabilidade da fase de execucao antes de atacar persistencia e visualizacoes

## Nao Objetivos

- refatorar `_save_benchmark_results()` nesta fase
- refatorar os plots e heatmaps nesta fase
- alterar o formato do `comparison_df`
- criar um novo modulo separado para benchmark agora

## Escopo Aprovado

Esta fase modifica apenas:

- `validation/benchmark.py`
- `tests/validation/test_benchmark.py`

Podera haver ajuste minimo em outros testes de `validation/` apenas se surgir regressao direta.

## Arquitetura

### `ModelBenchmark.run_benchmark()`

`run_benchmark()` continua sendo a entrada publica principal, mas passa a ter responsabilidade mais estreita:

- iterar sobre `model_paths`
- delegar a execucao por modelo a um helper interno
- preencher `self.results`
- criar `self.comparison_df`
- acionar persistencia existente quando `save_results=True`

### Helper interno por modelo

Um helper interno novo fica responsavel por executar o benchmark de um unico modelo.

Responsabilidades desse helper:

- instanciar `ModelValidator`
- chamar `validate_dataset(..., save_results=False)`
- coletar `metrics` e `model_info`
- devolver uma estrutura previsivel de sucesso
- capturar excecao e devolver uma estrutura previsivel de erro

Esse helper nao salva resultados, nao cria dataframe e nao gera visualizacoes.

## Fluxo

1. `run_benchmark()` recebe `model_paths`.
2. Para cada modelo, chama o helper interno.
3. O helper devolve um payload de sucesso ou erro para aquele modelo.
4. `run_benchmark()` popula `self.results` com esse payload.
5. `run_benchmark()` chama `_create_comparison_dataframe()`.
6. Se `save_results=True`, o fluxo de persistencia atual continua sendo usado sem mudanca estrutural.

## Estrategia de Refatoracao

Sequencia desta fase:

1. Adicionar testes vermelhos focados na execucao por modelo.
2. Extrair a menor estrutura interna possivel para representar sucesso e erro por modelo.
3. Adaptar `run_benchmark()` para delegar ao helper novo.
4. Fechar com a suite focal do benchmark e dos modulos de `validation` afetados.

## Estrategia de Testes

Prioridade: testes pequenos, sem `pandas`, `matplotlib` ou benchmark real pesado alem do necessario para o contrato do metodo.

### Testes novos

Cobrir:

- helper interno devolve sucesso com `metrics` e `model_info`
- helper interno devolve erro estruturado quando `ModelValidator` falha
- `run_benchmark()` popula `self.results` usando o helper
- `run_benchmark()` chama `_create_comparison_dataframe()`
- `run_benchmark()` respeita `save_results=False` sem chamar persistencia

### Nao entram nesta fase

- testes de CSV/JSON detalhados
- testes dos plots
- benchmark fim a fim com modelos reais

## Riscos e Mitigacoes

- Risco: mudar sem querer o formato esperado em `self.results`.
  Mitigacao: cobrir explicitamente os payloads de sucesso e erro nos testes.

- Risco: extrair logica demais e abrir uma arquitetura maior do que o necessario.
  Mitigacao: limitar a extracao a um unico helper interno de execucao por modelo.

- Risco: misturar esta fase com persistencia e visualizacao.
  Mitigacao: manter `_save_benchmark_results()` e `_generate_benchmark_visualizations()` fora do escopo.

## Criterios de Sucesso

- `run_benchmark()` fica menor e mais legivel
- a execucao por modelo fica isolada e testavel
- `self.results` e `self.comparison_df` preservam o comportamento publico atual
- a suite focal da fase permanece verde

## Fase Seguinte

Depois desta fase, os proximos encaixes naturais sao:

- extrair persistencia de benchmark
- reduzir acoplamento com visualizacoes
- avaliar discovery compartilhado com `domain_shift`
