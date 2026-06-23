# Especificacao de Design: Fase de Refatoracao de validator.py com Runner + Persistence

## Contexto

O arquivo `validation/validator.py` continua concentrando varias responsabilidades no mesmo objeto `ModelValidator`:

- carga e inicializacao do modelo
- inferencia em lote no `DataLoader`
- coleta de previsoes, probabilidades e labels
- calculo de metricas
- persistencia de metricas e predicoes
- geracao de visualizacoes

Depois da fase de `Validation CLI + Benchmark`, o modulo passou a ter fronteiras melhores na entrada, mas o miolo de `validator.py` ainda esta muito acoplado e pouco testavel.

## Problema

O `validator.py` atual nao e suficientemente isolado porque:

- o metodo `validate_dataset()` mistura inferencia, metrica e filesystem
- a logica de salvar resultados e gerar plots fica acoplada ao fluxo de execucao
- testes do comportamento interno exigem doubles demais ou dependencias pesadas
- o `ModelValidator` faz trabalho de fachada e de implementacao concreta ao mesmo tempo

## Objetivos

- manter `ModelValidator` como interface publica
- separar a execucao de inferencia em lote da persistencia de artefatos
- tornar o fluxo interno testavel com stubs simples
- preservar o contrato atual de `validate_dataset()`
- preparar o modulo para fases futuras em carga de modelo e visualizacao

## Nao Objetivos

- refatorar totalmente a carga de modelo nesta fase
- alterar o formato de retorno de `validate_dataset()`
- reescrever `validation/metrics.py`
- mover todo o codigo de visualizacao para outro modulo nesta etapa

## Escopo Aprovado

Esta fase criara dois arquivos centrais novos:

- `validation/validation_runner.py`
- `validation/validation_persistence.py`

O arquivo existente:

- `validation/validator.py`

sera adaptado para delegar inferencia e persistencia internamente, sem mudar sua interface publica principal.

## Arquitetura

### `validation/validation_runner.py`

Responsabilidades:

- receber `model`, `device` e `test_loader`
- executar inferencia em lote
- coletar `y_true`, `y_pred` e `y_prob`
- devolver uma estrutura simples com os arrays brutos

Fora de escopo:

- carregar modelo
- calcular metricas
- salvar arquivos

### `validation/validation_persistence.py`

Responsabilidades:

- salvar `metrics.json`
- salvar `predictions.json`
- acionar geracao de visualizacoes

Fora de escopo:

- executar inferencia
- carregar modelo
- calcular metricas

### `validation/validator.py`

Responsabilidades nesta fase:

- continuar expondo `ModelValidator`
- carregar o modelo
- chamar o runner para obter resultados brutos
- criar `ClassificationMetrics`
- delegar persistencia quando `save_results=True`
- retornar a mesma estrutura publica de resultados

## Fluxo

1. `ModelValidator` carrega o modelo.
2. `ModelValidator.validate_dataset()` chama o runner.
3. O runner devolve `y_true`, `y_pred` e `y_prob`.
4. `ModelValidator` calcula metricas via `ClassificationMetrics`.
5. `ModelValidator` chama a persistence quando necessario.
6. `ModelValidator` retorna o mesmo shape de resultado usado hoje.

## Estrategia de Refatoracao

Sequencia desta fase:

1. Criar `validation_runner.py` com a execucao de inferencia.
2. Cobrir o runner com testes de unidade.
3. Criar `validation_persistence.py` com salvamento de metricas e predicoes.
4. Cobrir a persistencia com testes de unidade.
5. Adaptar `validator.py` para delegar ao runner e a persistence.
6. Fechar com regressao focal do `ModelValidator`.

## Estrategia de Testes

Prioridade: unidades puras e regressao focal.

### Testes de `validation_runner.py`

Cobrir:

- coleta correta de `y_true`
- coleta correta de `y_pred`
- coleta correta de `y_prob`
- uso de `device` no loop de inferencia

Tecnica:

- modelo fake
- loader fake
- tensores ou objetos stubados minimos

### Testes de `validation_persistence.py`

Cobrir:

- salvamento de `metrics.json`
- salvamento de `predictions.json`
- chamada de funcoes de visualizacao sem renderizacao real

Tecnica:

- `tmp_path`
- doubles para geracao de plots
- stubs para `metrics_calculator`

### Testes de `validator.py`

Cobrir:

- delegacao para o runner
- delegacao para a persistence
- preservacao do contrato de retorno de `validate_dataset()`

Nao cobrir nesta fase:

- checkpoint real pesado
- dataset real completo
- renderizacao real de `matplotlib`

## Riscos e Mitigacoes

- Risco: alterar o contrato esperado por `benchmark.py` e pela nova CLI.
  Mitigacao: manter o shape atual do retorno de `validate_dataset()`.

- Risco: extrair cedo demais e criar camadas artificiais.
  Mitigacao: limitar a fase a runner + persistence, mantendo `ModelValidator` como fachada.

- Risco: visualizacao continuar parcialmente acoplada.
  Mitigacao: aceitar esse acoplamento residual nesta fase e deixar a limpeza total de plots para uma iteracao propria.

## Criterios de Sucesso

- `ModelValidator.validate_dataset()` continua funcionando
- inferencia e persistencia deixam de ficar no mesmo metodo
- runner e persistence ficam testaveis isoladamente
- o validator fica pronto para futuras limpezas em carga de modelo e visualizacoes

## Fase Seguinte

Depois desta fase, os proximos encaixes naturais sao:

- isolar carga de modelo e checkpoint
- separar visualizacoes do `validator.py`
- reduzir acoplamento residual com `metrics.py`
