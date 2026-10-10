# CKP01 e CKP02 — Resíduos SP

Assistente de consulta às normas e orientações de coleta seletiva e descarte de
resíduos de São Paulo, capital.
Público: cidadãos que precisam entender as orientações de
Ecopontos, Cata-Bagulho e coleta seletiva. O domínio permite um recorte pequeno
com fontes oficiais e evolução para RAG no CKP02.

**Integrantes e RMs:** Luigi Borgheti Freire RM569958
(Grupo do Challenge é da 1CCA)

## Validação real e estado de entrega

O Gemma no Ollama Cloud foi validado com a configuração do `.env`:

- CKP01: oito casos aprovados, incluindo conversa de seis turnos; sem erros de execução.
- Contexto crescente: 15 execuções (cinco janelas, três repetições), todos os quatro
  fatos corretos. Entre 410 e 6.490 tokens estimados, não houve degradação observada.
- Meta prompting: versões original e proposta aprovaram os oito casos. Não foi
  demonstrado ganho; o prompt original permanece ativo.
- CKP02: `nomic-embed-text` instalado no Ollama local, com 768 dimensões, após
  autorização do usuário. Chat e juiz RAGAS continuam no Gemma Cloud. O pipeline,
  as citações e uma consulta real pela interface Gradio foram validados.

O resumo e os caminhos das execuções reais estão em `evidence/ckp01/README.md`.
O diagnóstico atual está em `evidence/ckp02/connection_check.json`; a tentativa
Cloud que retornou HTTP 401 está preservada em `connection_check_cloud.json`.
Os resultados quantitativos e a escolha da configuração estão em
`evidence/ckp02/README.md`. A suíte final possui **69 testes offline aprovados**.
Na comparação final, as **30 respostas foram geradas e avaliadas sem erros**,
com faithfulness média **1,0** nas três configurações. Foi selecionado **256/32
sem reranking**, com answer_relevancy média **0,8583**, por desempate numérico.
Os cinco casos de limite passaram; o notebook executou com serviços reais,
importando as métricas auditáveis dessa comparação.

## Executar

Python 3.11 ou superior. Desenvolvimento verificado em Python 3.14.
Execute dentro desta pasta.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.main
```

Acesse http://127.0.0.1:7860. A interface abre sem chave e sem chamada ao modelo.
Para conversar, copie `.env.example` para `.env`, preencha `OLLAMA_API_KEY` e
reinicie. Linux/macOS: `cp .env.example .env`; PowerShell:
`Copy-Item .env.example .env`.

### Ollama: configurar e verificar

O projeto conserva `OLLAMA_MODEL=gemma4:cloud`, exigido no PDF. O endpoint padrão
é `https://ollama.com`, com Bearer e chave do `.env`. A documentação atual diferencia
o alias da aplicação/CLI (`gemma4:cloud`) do identificador da API direta
(`gemma4:31b`). Para servidor local, configure a raiz do servidor,
como `http://localhost:11434`, e autentique o serviço Cloud conforme sua instalação.
A chave continua exigida pela aplicação.

Referências: [Ollama Cloud](https://docs.ollama.com/cloud) e
[autenticação](https://docs.ollama.com/api/authentication).

Verificação opcional real das duas chains, nunca chamada no startup/testes:

```bash
python -m app.check_connection
```

## Requisitos implementados

| Requisito | Implementação |
|---|---|
| Duas chains | `app/chain.py`: ConversationChain para chat e LCEL para análise |
| Templates e LCEL | ChatPromptTemplate com variáveis/mensagens e operador pipe |
| Memória | ConversationTokenBufferMemory, 1.200 tokens, sessão isolada |
| Pydantic v2 | AnaliseConsulta, enums, restrições, validação semântica e parser |
| Persona XML | `app/prompts.py`, fontes, jurisdição e limitações |
| Interface local | `python -m app.main`, Gradio, análise JSON e nova conversa |
| Context rot + métricas | `python -m app.context_rot`, tabela/JSON/CSV/gráfico |
| Meta prompting | `python -m app.meta_prompting`, proposta e comparação v1/v2 |
| Memória/escopo | `python -m evals.run_chat`, incluindo seis turnos |
| Entrega | `python scripts/package_ckp01.py`, ZIP sem .env/venv/caches |

### Justificativa da memória

TokenBuffer limita o histórico variável a 1.200 tokens estimados, evitando seu
crescimento indefinido e conservando turnos recentes. As consultas do domínio
envolvem município, resíduo e volume; respostas concisas preservam espaço para
vários turnos. O orçamento está na faixa de 800–1.500 tokens do PDF. Ele limita o
histórico, não o system prompt nem a chamada de análise.

Sessões isoladas têm expiração e limpeza. A poda remove mensagens antigas e
respostas órfãs. Fatos removidos não são garantidos: o assistente deve perguntar
novamente. Conversas não são persistidas em disco. Se uma chain falhar, o turno
é desfeito para o usuário poder repetir sem duplicação ou histórico invisível.

O contador usa tiktoken/cl100k_base com vocabulário local em `data/tokenizer/`.
Funciona offline, sem tokenizer remoto. É estimativa reprodutível, não tokenização
nativa do Gemma nem medida de cobrança.

### Context rot

```bash
python -m app.context_rot
# Opcional: ampliar após revisar os resultados e o limite do modelo:
python -m app.context_rot --windows 0 10 20 40 60 --repeats 3
```

Prompt e pergunta fixos consultam um registro alvo presente em todas as janelas.
Aumentam somente os registros de distração; não há poda de memória no experimento.
São três repetições por janela. Qualidade é a fração dos quatro fatos corretos,
comparada a gabarito independente. Salva entradas, respostas brutas, tokens,
latência, erros, CSV, resumo e gráfico em `evidence/ckp01/`.

Fixtures são fictícias e identificadas como experimento, sem uso como fonte jurídica.
Se não ocorrer queda, a conclusão informa isso; nunca gera degradação fictícia.

### Meta prompting e avaliação real

```bash
python -m evals.run_chat
python -m app.meta_prompting
# Apenas gerar a proposta:
python -m app.meta_prompting --generate-only
```

O modelo propõe uma revisão e as duas versões recebem os mesmos casos/parâmetros.
Salva solicitação, proposta, prompts, respostas e comparação. Não ativa propostas
automaticamente: revisar regras, fontes e regressões. Após revisão:

```bash
python -m app.main --system-prompt caminho/para/prompt_v2.md
```

Verificações de expressões/campos não comprovam sozinhas veracidade jurídica ou
resistência a toda injeção. Revisar respostas e fundamentação manualmente. Falhas
ficam nos artefatos; nenhum resultado é inventado.

## Testes offline e entrega

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check app tests evals scripts
ruff format --check app tests evals scripts
python scripts/package_ckp01.py
```

Testes usam modelos determinísticos exclusivos da suíte para verificar contratos,
memória, rollback, interface, configuração e artefatos. Não acessam Ollama nem
comprovam resultados reais do modelo. A aplicação usa somente o cliente real.

O ZIP vai para `dist/CKP01_residuos_sp_grupo.zip`, com código, contexto local,
vocabulário, testes e evidências disponíveis. Não inclui .env, venv ou caches.

## Fontes e limites

`data/contexto_oficial.md` contém descrições selecionadas de fontes oficiais,
com URLs, tipos e limites. Não realiza busca externa/RAG. Não contém valores de
multas, artigos completos, endereços ou horários por rua. Não certifica vigência
atual.

O pacote CKP01 mantém esse escopo; o RAG integra o pacote separado CKP02 abaixo.

## CKP02 — DocMind SP: RAG legislativo

Implementação com Chroma persistente, embeddings `nomic-embed-text`, geração
`gemma4:cloud` a temperatura zero, citações verificadas e interface Gradio.
O notebook de entrega é `notebooks/CKP02_DocMind_SP.ipynb`.

São cinco documentos oficiais municipais: Lei 13.478/2002, Decreto 55.113/2014,
Portaria SMSUB 11/2020, Lei 14.803/2008 e Portaria SMSUB 8/2024. Os HTMLs originais
estão em `data/raw/`; os recortes por artigo, em `data/processed/`. O manifesto
`data/sources.json` registra URLs, datas, recortes e SHA-256 de ambos. Texto marcado
como revogado é removido do corpus, mantendo o original para auditoria. Esta é uma
fotografia documental; as respostas não certificam vigência atual.

### Instalação e execução

Use a mesma `.venv` da CKP01 ou crie uma nova. Para instalação apenas em CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rag.txt
python -m app.rag.main
```

A interface abre em `http://127.0.0.1:7860` sem consultar Ollama no início.
Copie `.env.example` para `.env`, informe a chave e os endpoints antes de perguntar.
Geração e embeddings têm endpoints independentes. Não há substituição automática
de modelo ou provedor se o endpoint não oferecer `nomic-embed-text`.
O alias opcional `OLLAMA_API_MODEL` segue a orientação da CKP01; sua verificação
fica para a configuração do serviço.

Na configuração híbrida validada, manter Ollama local iniciado, executar
`ollama pull nomic-embed-text` e usar no `.env`:

```dotenv
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gemma4:cloud
OLLAMA_API_MODEL=gemma4:31b
OLLAMA_EMBED_BASE_URL=http://localhost:11434
```

Essa instalação de embeddings é local; não muda o modelo de geração para local.

O reranking usa um CrossEncoder CPU, carregado na primeira consulta. Seu primeiro
uso requer baixar o modelo do Hugging Face; `RAG_RERANK=false` desativa esse
diferencial. Há opções para fixar revisão e exigir cache local. Chroma fica em
`.chroma/` e é reconstruído automaticamente quando o corpus ou parâmetros mudam.

Comandos opcionais para executar depois da configuração:

```bash
python -m app.rag.check_connection
python -m app.rag.index
python -m evals.run_rag
```

### Comparação, filtros e avaliação

O executor compara chunks de **256/32** e **512/64** caracteres/sobreposição
(12,5%), com separadores `\n\n`, `\n`, `. `, espaço e string vazia. As mesmas dez
perguntas com gabarito documental são avaliadas em cada configuração, usando
RAGAS `faithfulness` e `answer_relevancy` com juiz e embeddings Ollama explícitos.
Uma terceira configuração compara reranking. Não precisa de chave OpenAI.

Respostas, contextos recuperados, citações, parâmetros, latência, métricas e erros
são salvos em `evidence/ckp02/`. A seleção exige todas as perguntas válidas e
faithfulness média ≥ 0,7; busca o maior valor, com relevância como desempate.
Uma média parcial não pode vencer. O gráfico do notebook usa apenas métricas reais;
sem uma configuração elegível, registra que a meta não foi alcançada.

Filtros por tipo, categoria, documento, artigo e ano de publicação são aplicados no Chroma.
Quando a pergunta nomeia uma única norma por número/ano, a busca é restrita a ela;
um artigo explicitamente citado também restringe a seção. Perguntas ambíguas ou
gerais mantêm a busca sem essas restrições. Nenhum gabarito entra na recuperação.
O notebook demonstra a Portaria de 2024 com filtro de Ecopontos a partir de 2020,
exibe ordenação antes/depois do reranking e executa cinco casos de limite do domínio.
A função `app.rag.pipeline.buscar(consulta, filtros)` também retorna estrutura
serializável, pronta para integração posterior como ferramenta.

### Notebook, testes e entrega

No Colab, envie o ZIP CKP02 quando a célula solicitar. A instalação está automatizada.
O parâmetro `RUN_LIVE=False` verifica corpus, hashes e chunking sem conexões.
Para a execução completa, configure o segredo `OLLAMA_API_KEY` no Colab, ajuste os
endpoints/alias na célula de parâmetros e altere `RUN_LIVE=True`. Execute todas as
células e salve os resultados. `LAUNCH_UI=True` ativa a demonstração Gradio;
no Colab, essa opção cria um link temporário compartilhado.

Para usar embeddings locais no Colab, instale/inicie Ollama no próprio runtime ou
conecte um [runtime local conforme a documentação do Colab](https://research.google.com/colaboratory/local-runtimes.html).
`localhost` em um runtime hospedado não aponta para seu computador. O notebook
executado pode ser aberto no Colab para leitura dos resultados salvos; a conexão
local não é compartilhada com quem recebe o link.

`REUSE_EVIDENCE` permite importar uma comparação real já incluída no ZIP, validando
hashes do corpus/perguntas e configuração dos modelos antes de mostrar suas métricas.
Deixe vazio para recalcular RAGAS. O notebook informa explicitamente a origem dos
resultados; geração de exemplo, filtros e casos de limite continuam executados.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check app tests evals scripts
ruff format --check app tests evals scripts
python scripts/validate_notebook.py
# Run All com serviços reais (recalcula métricas se não informar --reuse-evidence):
python scripts/validate_notebook.py --live
python scripts/package_ckp02.py
```

A validação automatizada padrão do notebook é offline; `--live` executa serviços reais.
Modelos determinísticos existem
somente nos testes; não entram no corpus nem produzem evidências de desempenho.
O código não inventa métricas caso Ollama esteja indisponível.

Para adicionar documentos: salve o HTML oficial em `data/raw/`, acrescente a ficha
e os artigos selecionados em `data/source_specs.json` e execute
`python scripts/prepare_sources.py`. Revise o recorte e atualize as perguntas.
O manifesto/hash e a identidade das coleções serão recalculados.

O ZIP é `dist/CKP02_residuos_sp_grupo.zip`, sem credenciais, índices ou caches.
Antes da entrega, preencha integrantes/RMs, execute a avaliação real e confira
o notebook salvo, as fontes e o link compartilhado do Colab. Prazo: **23:59 da
Aula 08**, conforme o enunciado. Esses passos de serviço e entrega ficam com o usuário.
