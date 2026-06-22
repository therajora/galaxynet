# Especificacao de Design: Fase 1 de Refatoracao de Treino/CLI

## Contexto

O repositorio possui multiplos pontos de entrada para treino, com logica distribuida entre:

- `utils/scripts/run.sh`
- `utils/scripts/train_pretrained.sh`
- `utils/scripts/from_scratch.sh`
- `model/pretrained/train_pretrained.py`
- `model/from_scratch/train_from_scratch.py`

Hoje, esses fluxos misturam parsing de argumentos, validacao, resolucao de ambiente, selecao de device, configuracao de seed, orquestracao de treino e acoplamento com shell scripts. Isso dificulta testes unitarios, aumenta duplicacao e espalha regras de negocio por varios entrypoints.

## Problema

O modulo `Treino/CLI` nao e facilmente testavel porque:

- a camada Bash concentra parte da validacao e da orquestracao
- scripts Python acumulam responsabilidades demais
- ha forte dependencia de I/O real, GPU, dataset e ambiente
- o retorno dos fluxos e majoritariamente baseado em `print` e encerramento do processo
- existe duplicacao entre os fluxos `pretrained` e `from-scratch`

## Objetivos

- introduzir uma CLI Python unica para treino
- extrair um nucleo reutilizavel e testavel para a orquestracao
- reduzir a camada shell a wrappers finos
- manter execucao de `pretrained` e `from-scratch`
- maximizar testes de unidades puras na fase 1
- estabelecer um padrao reaproveitavel para os proximos modulos

## Nao Objetivos

- reorganizar o repositorio inteiro nesta fase
- reescrever o conteudo interno completo de `model/pretrained` ou `model/from_scratch`
- substituir todos os wrappers shell do projeto fora de `Treino/CLI`
- adicionar testes de treino real com GPU ou dataset completo

## Escopo Aprovado

A fase 1 criara apenas dois arquivos centrais novos:

- `model/training_cli.py`
- `model/training_core.py`

Os modulos existentes de treino permanecem no lugar e recebem apenas as extracoes minimas necessarias para serem chamados por `training_core.py`.

Os scripts:

- `utils/scripts/train_pretrained.sh`
- `utils/scripts/from_scratch.sh`

passam a ser wrappers finos para a nova CLI Python.

## Arquitetura

### `model/training_cli.py`

Responsabilidades:

- definir subcomandos como `pretrained` e `from-scratch`
- fazer parsing dos argumentos
- executar validacoes superficiais de interface
- chamar `training_core.py`
- transformar resultados e erros em mensagens e codigos de saida

Fora de escopo:

- carregar dataset
- instanciar modelos diretamente
- centralizar regras de negocio de treino

### `model/training_core.py`

Responsabilidades:

- receber opcoes ja parseadas
- normalizar argumentos e configuracoes
- resolver seed, device, paths e estrategia de execucao
- despachar para os fluxos `pretrained` e `from-scratch`
- retornar estruturas previsiveis de resultado

Formato esperado de retorno:

- `success`
- `message`
- `output_dir`
- `artifacts_path`
- `metrics_path` quando aplicavel

## Fluxo

1. A CLI recebe um comando de treino.
2. A CLI converte argumentos em uma estrutura simples de opcoes.
3. `training_core.py` valida e normaliza as opcoes.
4. `training_core.py` chama as funcoes ou classes existentes dos modulos de treino.
5. O core retorna um resultado estruturado para a CLI.
6. A CLI exibe a saida e define o codigo de retorno.

## Estrategia de Refatoracao

Sequencia da fase 1:

1. Criar `training_core.py` com interface estavel.
2. Criar `training_cli.py` como entrada principal.
3. Adaptar `train_pretrained.py` e `train_from_scratch.py` para delegarem ao core.
4. Reduzir `train_pretrained.sh` e `from_scratch.sh` a wrappers finos.
5. Remover duplicacoes evidentes de seed, device, config e montagem do fluxo.

## Estrategia de Testes

Prioridade: unidades puras.

### Testes de `training_core.py`

Cobrir:

- selecao correta do fluxo `pretrained` vs `from-scratch`
- resolucao de seed e device
- validacao de argumentos obrigatorios
- resolucao de configuracao por nome de modelo ou arquivo
- tratamento de falhas com retorno estruturado

Tecnica:

- mocks e stubs para treino real, dataset, Hugging Face, GPU e filesystem pesado

### Testes de `training_cli.py`

Cobrir:

- parsing de subcomandos
- traducao de argumentos para chamadas do core
- comportamento de erro e codigos de saida

### Testes de integracao leves

Cobrir apenas:

- subida do comando principal
- delegacao correta para o core

Nao cobrir nesta fase:

- treino completo
- benchmark de performance
- execucao dependente de dataset real grande

## Riscos e Mitigacoes

- Risco: duplicar logica durante a transicao.
  Mitigacao: usar o core como unica fonte de orquestracao nova e converter os entrypoints antigos em delegadores.

- Risco: quebrar comandos ja conhecidos.
  Mitigacao: manter wrappers shell temporarios.

- Risco: criar abstractions demais cedo.
  Mitigacao: limitar a fase 1 a `training_cli.py` e `training_core.py`.

## Criterios de Sucesso

- a nova CLI Python executa `pretrained` e `from-scratch`
- os wrappers shell deixam de conter regra de negocio
- o miolo do fluxo fica testavel sem dataset real e sem GPU
- existem testes unitarios cobrindo o fluxo principal do core
- a base da fase 2 fica pronta para continuar a simplificacao interna

## Fase Seguinte

Depois da fase 1, a proxima iteracao pode atacar modulo a modulo:

- limpeza interna de `model/pretrained`
- limpeza interna de `model/from_scratch`
- convergencia de configuracao, runtime e resultados
- adocao do mesmo padrao em `validation` e `domain_shift`
