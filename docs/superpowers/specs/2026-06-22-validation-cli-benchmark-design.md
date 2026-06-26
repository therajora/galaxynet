# Especificacao de Design: Fase de Refatoracao de Validation CLI + Benchmark

## Contexto

O modulo `validation/` concentra hoje responsabilidades demais em poucos pontos de entrada, principalmente em:

- `validation/validate_models.py`
- `validation/benchmark.py`
- `validation/validator.py`

O fluxo atual mistura parsing de argumentos, resolucao de device, montagem de dataset e dataloader, descoberta de modelos treinados, execucao de benchmark e impressao de resultados.

Esse desenho dificulta testes unitarios e torna a evolucao do modulo mais arriscada, especialmente quando combinado com imports pesados e dependencia de dados reais.

## Problema

A fase atual de `Validation` nao e suficientemente testavel porque:

- `validate_models.py` mistura CLI e orquestracao
- setup de dataset, dataloader e device aparece acoplado ao fluxo principal
- benchmark e descoberta de modelos dependem demais de filesystem e ambiente real
- os modos `single`, `benchmark` e `find` nao passam por um nucleo unificado
- o retorno dos fluxos ainda depende muito de `print` e pouco de estruturas previsiveis

## Objetivos

- introduzir uma CLI Python unica para `Validation`
- criar um nucleo reutilizavel para `single`, `benchmark` e `find`
- mover descoberta de modelos e setup de runtime para o core
- manter `benchmark.py` e `validator.py` reaproveitaveis, sem reescreve-los por completo agora
- maximizar testes de unidades puras e integracoes leves
- manter compatibilidade razoavel com os modos e argumentos principais existentes

## Nao Objetivos

- refatorar completamente `validator.py` nesta fase
- reescrever o submodulo `validation/visualization/`
- alterar profundamente o formato de saida dos resultados salvos
- cobrir validacao real com modelos pesados e datasets completos

## Escopo Aprovado

Esta fase criara dois arquivos centrais novos:

- `validation/validation_cli.py`
- `validation/validation_core.py`

Os arquivos existentes:

- `validation/validate_models.py`
- `validation/benchmark.py`
- `validation/validator.py`

serao adaptados apenas no minimo necessario para consumirem ou serem consumidos pelo novo core.

## Arquitetura

### `validation/validation_cli.py`

Responsabilidades:

- definir os subcomandos ou modos `single`, `benchmark` e `find`
- fazer parsing dos argumentos
- construir um request simples para o core
- transformar erros e resultados em mensagens e codigos de saida

Fora de escopo:

- montar dataset ou dataloader
- instanciar `ModelBenchmark` diretamente
- concentrar regras de negocio do fluxo de validacao

### `validation/validation_core.py`

Responsabilidades:

- validar combinacoes de argumentos por modo
- resolver `device`
- montar dataset e dataloader de teste
- descobrir modelos treinados
- despachar os fluxos `single`, `benchmark` e `find`
- retornar estruturas previsiveis de resultado

Formato esperado de retorno:

- `success`
- `message`
- `output_dir`
- `results_path`
- `model_paths` quando aplicavel

## Fluxo

1. A CLI recebe o comando de validacao.
2. A CLI converte argumentos em um request simples.
3. O core valida e normaliza os argumentos.
4. O core resolve `device`, dataset, dataloader e descoberta de modelos quando necessario.
5. O core chama `ModelValidator` ou `ModelBenchmark`.
6. A CLI imprime a saida e define o codigo de retorno.

## Estrategia de Refatoracao

Sequencia desta fase:

1. Criar `validation_core.py` com request, resultado e despacho por modo.
2. Criar `validation_cli.py` como nova entrada principal.
3. Adaptar `validate_models.py` para delegar ao core ou a nova CLI.
4. Mover descoberta de modelos e setup de dataset/device para o core.
5. Manter `benchmark.py` e `validator.py` como componentes reutilizaveis nesta fase.
6. Fechar com testes de core, CLI e execucao direta dos scripts.

## Estrategia de Testes

Prioridade: unidades puras e integracoes leves.

### Testes de `validation_core.py`

Cobrir:

- selecao correta de `single`, `benchmark` e `find`
- validacao de argumentos por modo
- resolucao de `device`
- descoberta de modelos treinados
- montagem do fluxo de benchmark sem depender de modelo real

Tecnica:

- mocks e stubs para dataset, dataloader, benchmark, validator, `torch` e filesystem

### Testes de `validation_cli.py`

Cobrir:

- parsing dos modos principais
- traducao de argumentos para requests do core
- codigos de saida e mensagens de erro

### Testes de integracao leves

Cobrir apenas:

- `python validation/validation_cli.py single --help`
- `python validation/validation_cli.py benchmark --help`
- execucao direta do script desde a raiz do repositorio

Nao cobrir nesta fase:

- validacao real de modelos pesados
- benchmark real completo
- geracao detalhada de plots do submodulo `visualization`

## Riscos e Mitigacoes

- Risco: quebrar a interface atual de `validate_models.py`.
  Mitigacao: preservar modos principais e adaptar o arquivo existente como delegador fino.

- Risco: duplicar setup de device e dataset entre core e validator.
  Mitigacao: concentrar o setup no core nesta fase e deixar a limpeza interna de `validator.py` para a proxima.

- Risco: crescer o escopo para visualizacao e analise detalhada.
  Mitigacao: limitar esta fase a `CLI + benchmark`, sem incluir refatoracao profunda do submodulo `visualization`.

## Criterios de Sucesso

- existe uma nova entrada Python unica para `Validation`
- `validate_models.py` deixa de concentrar a orquestracao principal
- os fluxos `single`, `benchmark` e `find` ficam testaveis sem dados reais
- benchmark e descoberta de modelos passam por um core reutilizavel
- o modulo fica pronto para uma fase seguinte focada em `validator.py`

## Fase Seguinte

Depois desta fase, a proxima iteracao natural e:

- limpar `validator.py` internamente
- separar inferencia, coleta de metricas e persistencia
- atacar `validation/visualization/` como etapa propria
