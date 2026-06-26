# Especificacao de Design: Fase de Simplificacao de `validation/validate_models.py`

## Contexto

Depois da introducao de `validation_core.py`, `validation_cli.py`, `validation_runner.py`, `validation_persistence.py` e `validation/model_loader.py`, o arquivo `validation/validate_models.py` continua existindo como ponte de compatibilidade.

Hoje ele ainda concentra duplicacao entre pares de funcoes publicas:

- `validate_single_entry()` e `validate_single_model()`
- `run_benchmark_entry()` e `run_benchmark()`

Esses pares repetem parte do setup de `device`, `test_loader`, `ModelValidator` e `ModelBenchmark`, o que deixa a borda legada maior do que precisa ser.

## Problema

Nesta fase, `validation/validate_models.py` ainda e menos simples e menos testavel do que o restante do modulo porque:

- a mesma logica aparece em duas funcoes publicas por fluxo
- defaults de `output_dir` e `device` ficam espalhados
- a borda de compatibilidade conhece detalhes demais de execucao
- pequenas mudancas no fluxo `single` ou `benchmark` exigem tocar em mais de um lugar

## Objetivos

- simplificar `validation/validate_models.py` sem quebrar compatibilidade publica
- consolidar a logica duplicada de `single` em um helper interno unico
- consolidar a logica duplicada de `benchmark` em um helper interno unico
- manter `main()` como delegador fino para `validation_cli.py`
- aumentar a testabilidade com testes pequenos e focados em delegacao

## Nao Objetivos

- refatorar `validation/benchmark.py` nesta fase
- alterar a interface publica de `validation_cli.py`
- mover descoberta de modelos para outro modulo agora
- alterar o formato publico dos retornos ja consumidos pelos testes existentes

## Escopo Aprovado

Esta fase modifica apenas:

- `validation/validate_models.py`
- `tests/validation/test_validation_cli.py`

Podera haver ajuste minimo em outros testes de `validation/` apenas se surgir regressao direta de compatibilidade.

## Arquitetura

### `validation/validate_models.py`

O arquivo continua publico e importavel, mas passa a ter uma estrutura mais simples:

- helpers internos para fluxo `single`
- helpers internos para fluxo `benchmark`
- funcoes publicas atuais como wrappers finos
- `main()` delegando para `validation_cli.py`

### Helpers internos

Helpers internos previstos:

- um helper para executar a validacao single com `ModelValidator`
- um helper para executar benchmark com `ModelBenchmark`

Esses helpers recebem os argumentos ja normalizados e concentram a logica compartilhada.

### Funcoes publicas preservadas

As seguintes funcoes continuam existindo:

- `validate_single_entry()`
- `validate_single_model()`
- `run_benchmark_entry()`
- `run_benchmark()`
- `find_trained_models()`
- `main()`

O contrato externo permanece o mesmo. A mudanca e interna: as funcoes deixam de duplicar a execucao real.

## Fluxo

### Fluxo `single`

1. A chamada entra por `validate_single_entry()` ou `validate_single_model()`.
2. A funcao publica aplica apenas compatibilidade de assinatura e defaults.
3. O helper interno resolve `device`, monta `test_loader` e executa `ModelValidator`.
4. A funcao publica devolve o formato que ja devolvia antes.

### Fluxo `benchmark`

1. A chamada entra por `run_benchmark_entry()` ou `run_benchmark()`.
2. A funcao publica aplica apenas compatibilidade de assinatura e defaults.
3. O helper interno resolve `device`, monta `test_loader` e executa `ModelBenchmark`.
4. A funcao publica devolve o formato que ja devolvia antes.

## Estrategia de Refatoracao

Sequencia desta fase:

1. Adicionar testes vermelhos para garantir que os pares publicos passam pela mesma logica compartilhada.
2. Extrair a menor estrutura interna possivel para eliminar a duplicacao.
3. Manter `find_trained_models()` intacta, salvo se surgir ajuste minimo inevitavel.
4. Rodar a suite focal de `validation`.

## Estrategia de Testes

Prioridade: preservar comportamento publico com o menor numero de testes novos.

### Testes novos

Cobrir:

- `validate_single_entry()` e `validate_single_model()` delegam para a mesma logica interna
- `run_benchmark_entry()` e `run_benchmark()` delegam para a mesma logica interna
- `main()` continua delegando para `validation_cli.py`

### Testes existentes que devem continuar verdes

- `tests/validation/test_validation_cli.py`
- `tests/validation/test_validation_core.py`
- `tests/validation/test_validator.py`
- `tests/validation/test_model_loader.py`

Nao entram nesta fase:

- testes com benchmark real pesado
- mudancas em `benchmark.py`
- mudancas no comportamento de `find_trained_models()`

## Riscos e Mitigacoes

- Risco: mudar sem querer o formato de retorno de uma funcao legada.
  Mitigacao: cobrir explicitamente o retorno esperado nos testes focais.

- Risco: extrair helpers demais e criar uma mini-arquitetura desnecessaria.
  Mitigacao: limitar a mudanca a dois helpers internos simples e wrappers finos.

- Risco: aumentar o escopo para `benchmark.py` ou `validation_cli.py`.
  Mitigacao: tratar esta fase apenas como limpeza da borda legada.

## Criterios de Sucesso

- `validation/validate_models.py` fica menor e com menos duplicacao
- os pares publicos `single` e `benchmark` passam a compartilhar execucao real
- `main()` continua funcionando como delegador fino
- a suite focal de `validation` permanece verde

## Fase Seguinte

Depois desta fase, os proximos encaixes naturais sao:

- refinar `validation/benchmark.py`
- avaliar extracao de discovery compartilhado com `domain_shift`
- iniciar a mesma estrategia incremental em `domain_shift/`
