# Especificacao de Design: Fase de Refatoracao da Carga de Modelo no Validator

## Contexto

Depois da extracao de runner e persistence, o `validation/validator.py` ainda concentra a carga de modelo no metodo `_load_model()`.

Hoje, esse metodo mistura:

- validacao de existencia do arquivo
- criacao da arquitetura via `create_pretrained_model()`
- leitura de checkpoint com `torch.load()`
- decisao entre `model_state_dict` e `state_dict` direto
- aplicacao dos pesos no modelo

Essa parte continua sendo um dos pontos mais acoplados do fluxo de validacao.

## Problema

A carga de modelo atual nao e suficientemente isolada porque:

- `_load_model()` mistura varias responsabilidades em um unico metodo
- a compatibilidade com formatos de checkpoint fica pouco testavel
- `ModelValidator` ainda conhece detalhes demais de restauracao de modelo
- testes dessa parte exigem stubs pesados quando feitos so pela fachada

## Objetivos

- extrair a carga de modelo para um componente reutilizavel
- manter `ModelValidator` como interface publica
- preservar compatibilidade com checkpoint contendo `model_state_dict`
- preservar compatibilidade com checkpoint contendo `state_dict` direto
- manter `ModelValidator` responsavel por `to(device)` e `eval()`

## Nao Objetivos

- mover toda a logica para `model_factory.py`
- alterar a assinatura publica de `ModelValidator`
- mudar o contrato dos resultados de validacao
- refatorar `benchmark.py` nesta fase

## Escopo Aprovado

Esta fase criara um novo arquivo central:

- `validation/model_loader.py`

O arquivo:

- `validation/validator.py`

sera adaptado para delegar a carga de modelo para esse novo componente.

## Arquitetura

### `validation/model_loader.py`

Responsabilidades:

- validar a existencia de `model_path`
- criar a arquitetura com `create_pretrained_model()`
- carregar checkpoint com `torch.load(..., map_location=device)`
- decidir qual `state_dict` usar
- aplicar os pesos ao modelo
- devolver o modelo pronto para uso

Entradas:

- `model_name`
- `model_path`
- `num_classes`
- `device`

Saida:

- modelo com pesos restaurados

### `validation/validator.py`

Responsabilidades nesta fase:

- continuar expondo `ModelValidator`
- delegar `_load_model()` ao novo loader
- manter `model.to(device)` e `model.eval()` na inicializacao

## Fluxo

1. `ModelValidator` inicializa.
2. `_load_model()` delega ao loader novo.
3. O loader valida o arquivo, cria a arquitetura, le o checkpoint e aplica os pesos.
4. O loader devolve o modelo restaurado.
5. `ModelValidator` move o modelo para `device` e chama `eval()`.

## Estrategia de Refatoracao

Sequencia desta fase:

1. Criar `validation/model_loader.py`.
2. Cobrir o loader com testes de unidade.
3. Adaptar `_load_model()` em `validator.py` para delegar ao loader.
4. Adicionar regressao para garantir `to(device)` e `eval()`.
5. Fechar com a suite focal de `Validation`.

## Estrategia de Testes

Prioridade: testes de unidade pequenos e regressao focal.

### Testes de `validation/model_loader.py`

Cobrir:

- erro quando `model_path` nao existe
- chamada correta de `create_pretrained_model()`
- chamada correta de `torch.load(..., map_location=device)`
- checkpoint com chave `model_state_dict`
- checkpoint sendo o `state_dict` direto

Tecnica:

- stubs para `torch`
- stubs para `create_pretrained_model()`
- modelos fake com `load_state_dict()`

### Testes de `validation/validator.py`

Cobrir:

- `_load_model()` delega ao loader novo
- `ModelValidator` continua chamando `to(device)` e `eval()`

Nao cobrir nesta fase:

- checkpoint real pesado
- `torchvision` real
- benchmark completo

## Riscos e Mitigacoes

- Risco: quebrar compatibilidade com formatos antigos de checkpoint.
  Mitigacao: testar explicitamente os dois formatos suportados.

- Risco: mover responsabilidade demais cedo.
  Mitigacao: limitar a extracao a um loader unico, sem mexer no restante da arquitetura do modulo.

- Risco: duplicar responsabilidade entre loader e validator.
  Mitigacao: deixar no loader somente a restauracao do modelo; `to(device)` e `eval()` permanecem no validator.

## Criterios de Sucesso

- `_load_model()` deixa de conter a logica detalhada de restauracao
- formatos de checkpoint ficam testaveis isoladamente
- `ModelValidator` continua funcionando sem alterar sua interface
- a suite focal de `Validation` continua verde

## Fase Seguinte

Depois desta fase, os proximos encaixes naturais sao:

- refinar `benchmark.py` para consumir fronteiras mais limpas
- reduzir acoplamento residual com `model_factory.py`
- extrair visualizacoes restantes do `validator.py`
