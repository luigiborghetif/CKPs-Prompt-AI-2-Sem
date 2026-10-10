# Plano de implementação — Assistente legislativo de resíduos de São Paulo

Documento de execução dos CKP01 e CKP02 de Prompt Engineering & AI, FIAP, 2º semestre de 2026. Baseado nos PDFs fornecidos e no domínio definido com o grupo. Este plano busca atender integralmente à rubrica e aos diferenciais; a nota final depende da avaliação do professor.

### Estado atual da implementação

Código dos CKP01 e CKP02 implementado e verificado com 69 testes offline,
Ruff e dependências sem conflitos. O corpus oficial contém cinco normas, originais
HTML auditáveis e recortes Markdown com hashes, produzindo 184 chunks em 256/32
e 88 em 512/64.

CKP01 validada no Gemma Cloud: 8/8 casos, conversa de seis turnos, 15 execuções de
contexto crescente com todos os fatos corretos e meta prompting v1/v2 com 8/8
em ambas. Sem ganho medido, o prompt original foi mantido. Evidências em
`evidence/ckp01/README.md`.

CKP02 agora usa configuração híbrida autorizada pelo usuário: geração/juiz Gemma
no Cloud e `nomic-embed-text` no Ollama local, com 768 dimensões. Embeddings,
pipeline, citações e Gradio foram testados com serviços reais. A comparação usa
as mesmas dez perguntas em 256/32, 512/64 e 512/64 com CrossEncoder. Os resultados
finais e a configuração escolhida são registrados em `evidence/ckp02/README.md`.
Na comparação final, 30/30 respostas foram geradas e avaliadas sem erros;
faithfulness média 1,0 nas três configurações. Foi aplicado 256/32 sem reranking
por desempate de relevância (0,8583), com diferença muito pequena entre baselines.
Os cinco casos de limite passaram. O notebook executou com serviços reais e
importou as métricas auditáveis da comparação final, validando hashes/parâmetros.
Perguntas que nomeiam uma norma/artigo usam esses metadados para restringir a
busca; o gabarito não é consultado pelo pipeline. RAGAS recebe o texto e os
metadados de fonte que o gerador recebeu.

O teste anterior de embeddings Cloud retornou HTTP 401 e foi preservado em
`evidence/ckp02/connection_check_cloud.json`. A decisão anterior de não instalar
embeddings locais foi substituída pela autorização explícita do usuário.
**Divergência do enunciado:** o PDF pede embeddings Cloud, enquanto a execução
validada usa locais. Confirmar essa alternativa com o professor. Não substituir
o modelo exigido nem omitir essa diferença na entrega.

Ainda cabe ao grupo preencher nomes/RMs, compartilhar o Colab com permissão de
leitura e enviar os pacotes pelo líder. O notebook contém parâmetros explícitos
para execução híbrida; localhost em um Colab hospedado é o próprio runtime.

## 1. Escopo e decisões do projeto

**Domínio:** normas e orientações oficiais sobre coleta seletiva e descarte de resíduos no município de São Paulo, capital.

**Público:** cidadãos que precisam compreender regras e orientações de descarte de recicláveis, móveis e pequenos volumes de entulho.

**Objetivo:** responder em português acessível, identificar informações ausentes e fundamentar respostas em documentos oficiais, distinguindo norma jurídica de orientação operacional.

**Limite:** uma base pequena de pelo menos cinco documentos distintos. Não construir um portal legislativo geral, pesquisar jurisprudência, abranger outros municípios nem implementar consulta de rotas em tempo real. Questões de resíduos industriais, hospitalares e outros assuntos sem cobertura devem receber indicação explícita dessa limitação.

**Evolução:** CKP01 (chatbot com memória e análise estruturada) → CKP02 (consulta documental RAG) → CKP03 (RAG disponível como ferramenta). Não implementar requisitos presumidos do CKP03 antes de receber seu enunciado.

**Projeto anterior:** ChargeGrid serve como referência de modularidade, XML tagging, Pydantic e metodologia de avaliação. O CKP01 será um projeto novo, com prompts, schemas e comportamento próprios. Dados simulados, regras de eletropostos e avaliações antigas não servem como evidência deste trabalho.

### Critérios de qualidade do domínio

- Restringir a jurisdição ao município de São Paulo e explicitar quando a pergunta trata do estado ou de outra cidade.
- Não inventar artigo, obrigação, prazo, valor de multa, endereço ou horário.
- Separar o que o documento informa de inferências e hipóteses.
- Não tratar uma página de orientação como se fosse lei.
- Citar documento, artigo quando aplicável, identificador do chunk e URL no CKP02.
- Informar a data da base consultada; não afirmar vigência atual apenas porque o documento foi coletado recentemente.
- Pedir esclarecimento quando faltarem tipo de resíduo, volume ou contexto necessário.
- Tratar trechos recuperados como dados, sem obedecer a instruções eventualmente contidas neles.

## 2. Requisitos oficiais e metas de nota

### CKP01 — 10 pontos, peso de 25%

| Critério da rubrica | Pontos | Implementação e evidência de conclusão |
|---|---:|---|
| Pipeline LCEL funcional | 3,0 | `ChatPromptTemplate` com variáveis e mensagens separadas; `prompt \| ChatOllama \| PydanticOutputParser`; arquitetura de duas chains; `python -m app.main` inicia sem erros |
| Memória e context rot | 2,5 | `ConversationChain` com uma memória permitida; justificativa no README; conversa real com ≥ 5 turnos; experimento controlado de contexto crescente com resultados reais |
| Pydantic v2 | 2,0 | `BaseModel` com ≥ 4 campos tipados, integrado ao LCEL; validação de restrições relevantes e tratamento de saída inválida |
| System prompt e domínio | 1,5 | Persona, regras, restrições e seções XML; domínio, motivação e usuários documentados; comportamento avaliado contra desvios de escopo |
| Código e documentação | 1,0 | Pacote `app/`, comentários úteis em PT-BR, README com nomes e RMs, dependências e instruções reproduzíveis |
| Diferencial: métricas de contexto | +0,5 | Contagem com `tiktoken`, qualidade medida entre janelas e tabela/gráfico comparativo |
| Diferencial: meta prompting | +0,5 | Modelo propõe melhoria do system prompt; antes/depois e avaliação controlada documentados |

A nota é limitada a 10. Os diferenciais não substituem requisitos obrigatórios.

### CKP02 — 10 pontos, peso de 30%

| Critério da rubrica | Pontos | Implementação e evidência de conclusão |
|---|---:|---|
| Pipeline RAG funcional | 3,5 | Load → split → embed → store → retrieve → generate; respostas fundamentadas e citações; cinco perguntas executadas sobre os arquivos entregues |
| Chunking e RAGAS | 3,0 | ≥ 2 configurações comparadas; ≥ 5 perguntas por configuração; tabela por pergunta com `faithfulness` e `answer_relevancy`; escolha quantitativa; faithfulness médio ≥ 0,7 |
| Base de conhecimento | 2,0 | ≥ 5 documentos reais e relevantes; fontes oficiais citadas; conteúdo suficiente para perguntas não triviais |
| Código e documentação | 1,5 | Funções modulares, comentários em PT-BR, README com nomes/RMs e instruções de inclusão de documentos |
| Diferencial: metadata filtering | +0,5 | Metadados de tipo, data e categoria; recuperação com `where=` demonstrada |
| Diferencial: reranking | +0,5 | Cross-encoder reordena candidatos antes da geração; evidência de funcionamento e comparação |
| Diferencial: Gradio | +0,5 | `gr.ChatInterface` ou `gr.Blocks` integrado ao RAG e às fontes |

O PDF lista três diferenciais de +0,5, mas limita o bônus total a +1,0 e a nota a 10. Planejar os três, sem pressupor bônus de +1,5.

**Metas internas, não requisitos adicionais do professor:** faithfulness médio ≥ 0,9 na configuração escolhida; answer relevancy médio ≥ 0,85 nas perguntas respondíveis; 100% das citações resolvíveis; nenhuma referência jurídica inventada no conjunto de avaliação. Se as metas não forem atingidas, registrar o resultado e iterar sem fabricar métricas.

## 3. Stack necessária

Usar Python 3.11 como ambiente de referência, uma `.venv` local para CKP01 e Google Colab para a entrega do CKP02. Confirmar compatibilidade no ambiente real do Colab. Após o teste inicial, fixar versões exatas testadas; não preencher versões presumidas neste plano.

| Componente | Tecnologia | Uso |
|---|---|---|
| Modelo de conversa e geração | Ollama Cloud + `gemma4:cloud`, conforme PDF | Conversa, análise estruturada, meta prompting e geração RAG; CKP02 com `temperature=0` |
| Integração de chat | `langchain-ollama` / `ChatOllama` | Cliente usado nas chains |
| Prompts e composição | `langchain-core` | `ChatPromptTemplate`, LCEL, parsers e documentos |
| Memória exigida | `langchain-classic`, se necessário para a versão escolhida | `ConversationChain` e `ConversationTokenBufferMemory` |
| Validação | `pydantic` v2 + `PydanticOutputParser` | Entrada e saída estruturadas |
| Interface | `gradio` | Aplicação local CKP01 e interface RAG CKP02 |
| Configuração local | `python-dotenv` | `.env` com `OLLAMA_API_KEY`; apenas `.env.example` versionado |
| Métricas de contexto | `tiktoken`, `pandas`, `matplotlib` | Contagem estimada, tabelas e gráficos exportáveis |
| Leitura de PDFs | `pypdf` | Extração e inspeção de PDFs da base |
| Coleta de páginas | `httpx`, `beautifulsoup4` | Download e conversão revisada de páginas oficiais para Markdown |
| Divisão de documentos | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| Embeddings | `OllamaEmbeddings(model='nomic-embed-text')` | `embed_query()` e `embed_documents()`; único modelo permitido |
| Banco vetorial | `chromadb` | `Client()` ou `PersistentClient()`, coleções identificadas pelo domínio/configuração |
| Avaliação RAG | `ragas` | `faithfulness` e `answer_relevancy`, com adaptadores compatíveis com a versão fixada |
| Reranking | `sentence-transformers` / `CrossEncoder` | Modelo inicial: `cross-encoder/ms-marco-MiniLM-L-6-v2`; testar desempenho em PT-BR |
| Qualidade | `pytest`, `ruff` | Testes úteis de comportamento, lint e formatação |
| Notebook | Jupyter / Google Colab | `.ipynb` executado e link com leitura |

Separar `requirements.txt` (CKP01), `requirements-rag.txt` (acréscimos do CKP02) e `requirements-dev.txt` (qualidade). Não exigir Chroma, RAGAS ou reranker para iniciar o CKP01.

### Verificações técnicas antes de desenvolver

1. **Modelo e autenticação:** o PDF exige `gemma4:cloud` com chave. A documentação atual do Ollama diferencia o alias da aplicação/CLI dos identificadores aceitos na API direta. Executar uma chamada real com a configuração exigida. Se houver incompatibilidade, registrar erro, endpoint e documentação e resolver o entendimento com o professor antes de adotar outro identificador. Não mudar silenciosamente o modelo.
2. **Embeddings:** o PDF especifica `nomic-embed-text` via Ollama Cloud. Confirmar disponibilidade real de `embed_query()` e `embed_documents()` nesse serviço e no Colab. A existência do modelo na biblioteca Ollama não prova hospedagem Cloud. Se não houver suporte, levar a incompatibilidade ao professor; não trocar modelo nem assumir autorização para execução local.
3. **Memória:** confirmar imports e execução de `ConversationChain` na versão escolhida. Usar a API pedida no enunciado mesmo que exemplos modernos apresentem outras alternativas.
4. **Contagem:** `tiktoken` fornece uma medida estimada para Gemma, não a tokenização nativa garantida. Documentar encoding e método. Verificar o contador usado pela memória para evitar download implícito de tokenizer ou falha durante o chat.
5. **RAGAS:** validar uma amostra com as duas métricas e com LLM/embeddings explicitamente configurados; evitar dependência padrão de serviços OpenAI. Não interpretar erro ou `NaN` como nota zero ou avaliação concluída.

## 4. Arquitetura e estrutura planejadas

### CKP01: duas chains

```text
Mensagem → validação de entrada → ConversationChain + memória da sessão
                                      ↓ resposta conversacional
           pergunta + resposta + contexto relevante
                                      ↓
        ChatPromptTemplate | ChatOllama | PydanticOutputParser
                                      ↓
               análise estruturada → interface Gradio
```

A primeira chain conversa e conserva o histórico. A segunda analisa a consulta/atendimento e valida uma saída estruturada. Evitar que duas chains adicionem o mesmo turno à memória. Um parser garante estrutura, não veracidade: testar ambas separadamente.

**Memória inicial:** `ConversationTokenBufferMemory`, orçamento de 1.200 tokens, dentro da faixa de 800–1.500 mencionada no PDF. Justificar o equilíbrio entre continuidade e limite de contexto. Uma memória independente por sessão e opção de limpar conversa. Não é obrigatório implementar as três estratégias; o exemplo do PDF não substitui a exigência de apenas uma.

**Schema inicial `AnaliseConsulta`:** `tema` (enum), `jurisdicao` (enum), `resumo_consulta` (texto), `tipo_residuo` (texto opcional), `volume_m3` (número opcional não negativo), `informacoes_faltantes` (lista de textos), `necessita_esclarecimento` (booleano) e `limitacoes` (lista). Validar textos vazios, enums e consistência entre informação ausente e pedido de esclarecimento.

No CKP01, usar pequeno contexto oficial revisado para exemplos do domínio, sem anunciar uma recuperação documental que ainda não existe. Toda afirmação normativa precisa de suporte; informações não fornecidas devem permanecer desconhecidas.

### CKP02: recuperação modular

```text
Fontes oficiais → extração revisada → documentos + metadados
    → splitter → nomic-embed-text → ChromaDB

Consulta → embedding → retrieve com filtros → reranking
    → chunks identificados → prompt | gemma4:cloud | parser
    → resposta + citações → Gradio / notebook
```

Expor `buscar(consulta)` com retorno estável contendo texto, fontes e chunks, para reutilização futura como ferramenta. Filtros opcionais devem ser explícitos. Histórico de chat não substitui a fonte normativa; perguntas de continuação precisam ser transformadas em consultas compreensíveis antes da busca.

```text
CKPs-Prompt-AI-2-Sem/
├── PLAN.md
├── README.md
├── .env.example
├── requirements.txt
├── requirements-rag.txt
├── requirements-dev.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── chain.py
│   ├── memory_manager.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── context_rot.py
│   ├── meta_prompting.py
│   └── rag/
│       ├── __init__.py
│       ├── loaders.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── store.py
│       ├── retrieval.py
│       ├── reranking.py
│       └── pipeline.py
├── data/
│   ├── raw/                  # Documentos originais e snapshots oficiais
│   ├── processed/            # Markdown revisado para ingestão
│   └── sources.json          # Manifesto e rastreabilidade
├── evals/
│   ├── chat_cases.json
│   ├── rag_questions.json
│   ├── run_chat.py
│   ├── run_rag.py
│   └── results/              # Resultados reais por execução/configuração
├── evidence/
│   ├── ckp01/                # Memória, context rot, meta prompting
│   └── ckp02/                # Comparações, citações e filtros
├── notebooks/
│   └── CKP02_DocMind_SP.ipynb
├── scripts/
│   └── package_ckp01.py
└── tests/
    ├── test_schemas.py
    ├── test_memory.py
    ├── test_chat.py
    ├── test_sources.py
    ├── test_retrieval.py
    └── test_citations.py
```

Banco Chroma gerado, `.venv`, caches e segredos devem ficar fora do Git e do ZIP. Recriar o índice a partir dos documentos entregues.

## 5. Base documental pequena

Fontes candidatas já identificadas, ainda sujeitas a seleção e revisão:

| ID | Documento oficial | Tipo e uso |
|---|---|---|
| SP-001 | [Lei nº 13.478/2002](https://legislacao.prefeitura.sp.gov.br/lei-13478-de-30-de-dezembro-de-2002) | Norma geral de limpeza urbana; guardar original e selecionar dispositivos pertinentes, sem ingerir indiscriminadamente toda a lei |
| SP-002 | [Decreto nº 55.113/2014 — texto consolidado](https://legislacao.prefeitura.sp.gov.br/decreto-55113-de-15-de-maio-de-2014/consolidado) | Norma sobre implantação de Ecopontos |
| SP-003 | [Coleta Domiciliar Seletiva — SP Regula](https://prefeitura.sp.gov.br/spregula/w/residuos_solidos/coleta_seletiva/4623) | Orientação operacional sobre coleta seletiva |
| SP-004 | [Perguntas Frequentes — SP Regula](https://prefeitura.sp.gov.br/web/spregula/w/residuos_solidos/acesso_a_informacao/183183) | Selecionar perguntas pertinentes ao escopo |
| SP-005 | [Cata-Bagulho e Ecopontos — Prefeitura](https://prefeitura.sp.gov.br/8cb6a170766763c7815c6c9a3de1c810) | Materiais aceitos, restrições e diferenças entre serviços |

O PDF aceita PDF, TXT e Markdown; orientações oficiais são compatíveis com a base proposta, mas devem ser identificadas como orientações, mantendo o componente legislativo das normas. Não contar cinco cópias ou cinco recortes da mesma lei como cinco documentos distintos.

**Manifesto mínimo:** ID, título, URL, órgão, tipo, categoria, jurisdição, número/ano quando aplicável, data de publicação/atualização quando informada, data de coleta, arquivo original, arquivo processado, hash SHA-256 e descrição do recorte. Usar `null` para data desconhecida, sem inventar valores.

**Revisão obrigatória:** preservar artigos, parágrafos, listas e tabelas; remover menus e rodapés; identificar texto revogado/riscado; não transformar texto revogado em regra vigente; conferir alterações e links de consolidação. Comparar a extração com a página/PDF original, especialmente tabelas cujas permissões estejam expressas por ícones. Se houver conflito entre fontes, registrar e investigar; não resolver apenas escolhendo a página coletada mais recentemente.

Conservar os originais para auditoria. Recortes devem manter localização e contexto suficientes para não alterar o sentido. Se os cinco documentos não permitirem perguntas distintas e não triviais, substituir fontes redundantes por documentos oficiais mais pertinentes.

## 6. Ordem de implementação e critérios de passagem

### Etapa 0 — Formalizar domínio e grupo

- [ ] Registrar domínio e verificar a restrição do PDF sobre duplicidade na turma.
- [ ] Confirmar integrantes, nomes completos, RMs e líder; não copiar automaticamente a equipe do projeto anterior.
- [ ] Preencher README com objetivo, público, limites e evolução entre checkpoints.
- [ ] Registrar datas de aulas quando conhecidas; não converter os prazos relativos em datas presumidas.

**Pronto quando:** domínio e integrantes estão documentados, sem informação presumida.

### Etapa 1 — Prova técnica de compatibilidade

- [ ] Preparar ambiente Python e dependências mínimas.
- [ ] Validar chamada ChatOllama autenticada e identificador exigido.
- [ ] Validar `ConversationChain`, memória por tokens e parser com um schema mínimo.
- [ ] Antecipar chamada real de embeddings e uma avaliação RAGAS no ambiente previsto para CKP02.
- [ ] Fixar versões testadas e documentar endpoints, sem registrar credenciais.
- [ ] Resolver divergências de infraestrutura/enunciado antes de depender delas.

**Pronto quando:** chamadas reais e imports funcionam e não há substituição não acordada de requisito.

### Etapa 2 — Pacote, configuração e contexto inicial

- [ ] Criar pacote `app/` e entry point `python -m app.main`.
- [ ] Carregar `.env`, validar chave/configuração e produzir erro compreensível quando faltar configuração.
- [ ] Criar `.env.example` com placeholders e ignorar `.env`.
- [ ] Selecionar pequeno contexto oficial revisado para as demonstrações do CKP01.
- [ ] Definir schema do domínio e contrato de retorno da aplicação.

**Pronto quando:** configuração é reproduzível e schemas rejeitam entradas semanticamente inválidas.

### Etapa 3 — Chat com memória e saída estruturada

- [ ] Escrever persona, escopo, restrições e formato em seções XML.
- [ ] Implementar `ChatPromptTemplate` com variáveis e mensagens system/human separadas.
- [ ] Implementar `ConversationChain` e TokenBuffer de 1.200 tokens por sessão.
- [ ] Implementar segunda chain LCEL de análise com `PydanticOutputParser`.
- [ ] Tratar falha de rede e saída inválida sem apresentar uma análise não validada.
- [ ] Criar interface Gradio com resposta, análise e botão de nova conversa.

**Pronto quando:** ≥ 5 turnos reais preservam informações relevantes, sessões não se misturam, reset funciona e a segunda chain retorna objeto validado.

### Etapa 4 — Context rot e meta prompting

- [ ] Construir casos com fatos identificáveis e resposta esperada independente do modelo.
- [ ] Executar experimento de contexto crescente, conforme protocolo da seção 7.
- [ ] Salvar entradas, respostas, estimativas de tokens, qualidade e latência.
- [ ] Gerar tabela e gráfico de context rot.
- [ ] Usar o próprio modelo para sugerir melhoria do prompt; guardar solicitação, proposta, v1 e v2.
- [ ] Comparar v1/v2 nas mesmas condições e relatar ganhos ou ausência deles.

**Pronto quando:** existem evidências reais e reproduzíveis dos dois diferenciais. Apenas editar manualmente o prompt não comprova meta prompting.

### Etapa 5 — Validar e empacotar CKP01

- [ ] Executar testes funcionais, lint e formatação.
- [ ] Completar README, justificativa de memória e seção de context rot.
- [ ] Testar instalação e execução a partir do ZIP em diretório limpo.
- [ ] Conferir matriz da rubrica e presença das evidências.
- [ ] Gerar `CKP01_residuos_sp_grupo.zip` sem segredos ou dependências instaladas.

**Pronto quando:** outra pessoa consegue iniciar e verificar todos os critérios usando somente a entrega e sua própria chave.

### Etapa 6 — Preparar os cinco documentos do CKP02

- [ ] Coletar originais e registrar manifesto com fontes e hashes.
- [ ] Converter páginas para Markdown e revisar extração.
- [ ] Conferir consolidação, revogações e diferenças entre normas e orientações.
- [ ] Escrever perguntas de avaliação com respostas e evidências revisadas antes da otimização.
- [ ] Registrar no notebook links de origem e recortes adotados.

**Pronto quando:** cinco documentos distintos têm proveniência e conteúdo verificáveis, suficientes para as perguntas planejadas.

### Etapa 7 — RAG básico, sem diferenciais

- [ ] Implementar load → split → embed → store → retrieve → generate.
- [ ] Usar `RecursiveCharacterTextSplitter` com `separators=["\n\n", "\n", ". ", " ", ""]`.
- [ ] Começar com `chunk_size=512`, `chunk_overlap=64` e registrar unidade de medida.
- [ ] Criar IDs estáveis de chunks e metadados de origem/localização.
- [ ] Criar coleção Chroma por configuração, com nome contendo o domínio.
- [ ] Gerar com `gemma4:cloud`, `temperature=0`, apenas a partir do contexto recuperado.
- [ ] Validar que as citações correspondem a chunks realmente recuperados.
- [ ] Expor `buscar(consulta)` e tratar pergunta sem suporte documental.

**Pronto quando:** cinco perguntas reais percorrem todo o pipeline e as fontes podem ser conferidas.

### Etapa 8 — Comparar chunking com RAGAS

- [ ] Comparar no mínimo 256/32 e 512/64; acrescentar 1024/128 se necessário.
- [ ] Explicitar se tamanhos são caracteres ou tokens. Sugestão inicial: caracteres, padrão do splitter; medir tokens separadamente. Caso a aula determine unidade diferente, ajustar e documentar.
- [ ] Isolar coleções e manter iguais documentos, perguntas, geração, número de candidatos e avaliação.
- [ ] Medir `faithfulness` e `answer_relevancy` por pergunta/configuração.
- [ ] Analisar falhas e escolher configuração com base nos resultados.
- [ ] Salvar respostas, contextos recuperados e resultados brutos, não apenas médias.

**Pronto quando:** tabela completa sem avaliações ausentes, faithfulness médio ≥ 0,7 e justificativa quantitativa da escolha; buscar ≥ 0,9.

### Etapa 9 — Diferenciais do CKP02

- [ ] Implementar filtro `where=` por metadados e demonstrar seu efeito.
- [ ] Recuperar inicialmente 8 candidatos, aplicar cross-encoder e enviar os 4 melhores; ajustar a partir de evidência.
- [ ] Comparar configuração escolhida com e sem reranking, mantendo as demais condições.
- [ ] Validar desempenho em português; preservar melhoria fundamentada, sem pressupor que reranking sempre ajuda.
- [ ] Integrar RAG ao Gradio com fontes clicáveis e data da base.
- [ ] Avaliar novamente a configuração final, incluindo filtros/reranking quando usados.

**Pronto quando:** diferenciais funcionam e o resultado final continua fundamentado e mensurável.

### Etapa 10 — Colab, documentação e entrega CKP02

- [ ] Montar notebook com instalação, configuração, carga, indexação, perguntas, comparação, métricas e interface.
- [ ] Ler `OLLAMA_API_KEY` pelo mecanismo de Secrets do Colab; não inserir chave em célula ou saída.
- [ ] Usar módulos do projeto por versão/commit fixado ou entregá-los junto; evitar depender de branch mutável.
- [ ] Incluir todos os documentos e instruções para reconstruir o índice.
- [ ] Testar `Run All` em runtime novo, sem depender de localhost da máquina do desenvolvedor ou de arquivos não entregues.
- [x] Salvar `.ipynb` executado, tabelas e outputs relevantes, sem erros ou segredos.
- [ ] Conferir compartilhamento do Colab com permissão de leitura.
- [x] Documentar como adicionar novo documento e reindexar.

**Pronto quando:** notebook completo reproduz o pipeline e avaliações a partir dos artefatos entregues.

## 7. Protocolos de avaliação

### Memória do CKP01

Executar conversa com pelo menos seis turnos: informar município, tipo de resíduo e volume; fazer perguntas de continuação; pedir que o assistente recupere os fatos informados. Avaliar referência a informações anteriores, não repetidas na pergunta final.

Testar também duas sessões com fatos diferentes, limpeza de uma sessão e histórico excedendo o limite. Confirmar poda do histórico pelo contador escolhido e ausência de vazamento entre usuários. Não exigir lembrança de fatos que já foram removidos; quando faltar informação, esperar pedido de esclarecimento.

### Experimento de context rot

1. Escolher uma pergunta fixa cujo gabarito dependa de um fato no contexto de teste, sem depender de conhecimento externo.
2. Manter system prompt, pergunta, modelo e parâmetros constantes; variar apenas tamanho e composição planejada do contexto.
3. Começar com cinco janelas crescentes, por exemplo contextos com 0/5/10/15/20 turnos de distração mais o mesmo fato-alvo. Medir tokens efetivos, pois turnos não equivalem a uma quantidade fixa de tokens.
4. Manter o fato-alvo presente em todas as janelas. Separar perda por poda da memória de degradação com contexto disponível; a poda sozinha não demonstra context rot.
5. Usar o mesmo critério de qualidade: recuperação do fato, atendimento à pergunta e ausência de contradição. Registrar também validade do schema e latência.
6. Repetir preferencialmente três vezes por janela e apresentar agregados e resultados individuais.
7. Se não houver degradação, relatar isso e ampliar o experimento dentro do limite do modelo. Não construir um gráfico fictício nem ocultar execuções para forçar uma queda.

O experimento usa fixtures de teste claramente identificadas, separadas da base documental real do CKP02. Salvar encoding `tiktoken`, estimativas e protocolo em `evidence/ckp01/`.

### Meta prompting

Pedir ao modelo que revise o system prompt considerando escopo, clareza e falhas observadas. Revisar a sugestão antes de adotá-la; ela não pode alterar requisitos. Comparar baseline e versão proposta no mesmo conjunto de consultas, com modelo, contexto e parâmetros constantes. Salvar o processo e os resultados, incluindo regressões e ausência de ganho.

### Avaliação do CKP02

- Preparar ≥ 5 perguntas respondíveis; alvo interno de 8–10 perguntas distribuídas pelos documentos.
- Incluir questões sobre diferenças entre serviços e restrições de descarte, além de perguntas simples de localização de informação.
- Elaborar o gabarito a partir dos documentos já revisados. Não usar respostas geradas pelo próprio sistema como única referência.
- Rodar as mesmas perguntas em cada configuração; mínimo oficial de cinco perguntas × duas configurações = dez pares avaliados, cada um com as duas métricas.
- Manter metadados da execução: modelo/endpoint, parâmetros, versões, hashes dos documentos, chunking, top-k, reranking e data/hora.
- Tabela mínima: `configuracao`, `pergunta_id`, `pergunta`, `resposta`, `chunks`, `faithfulness`, `answer_relevancy`, `status`. Adicionar média por configuração, erros, latência e custo de contexto estimado.
- Configurar explicitamente o juiz e embeddings de RAGAS; medir de verdade, sem substituir a métrica por correspondência de palavras.
- Perguntas sem resposta, fora do escopo e ambíguas ficam em avaliação separada de recusa/esclarecimento. Não misturá-las sem explicação à média de perguntas respondíveis.
- Se houver `NaN`, timeout ou erro de parsing, guardar falha e repetir de forma registrada. Não excluir silenciosamente linhas da média.
- Ao otimizar, modificar um fator por vez. Depois avaliar algumas perguntas novas para verificar que a melhoria não depende apenas do conjunto usado para ajustar o sistema.

## 8. Testes e evidências de aceitação

Testes com mocks verificam contratos e falhas; não comprovam qualidade real do modelo. Conversas, context rot e RAGAS exigem execução real com resultados salvos. Priorizar testes de comportamento e riscos concretos, evitando testar apenas nomes de arquivos ou repetir a implementação.

| ID | Teste | Critério de aprovação | Evidência |
|---|---|---|---|
| T01 | Instalação limpa CKP01 | Dependências instaladas e aplicação inicia pelo comando exigido | Registro de instalação/execução |
| T02 | Schema | Aceita saída válida; rejeita enum inválido, volume negativo, texto vazio e estrutura incompatível | `pytest` |
| T03 | Arquitetura | Chat usa ConversationChain; análise usa LCEL e parser; resposta inválida não vira sucesso | Teste de integração + chamada real |
| T04 | Memória | ≥ 5 turnos com recuperação correta de fatos ainda disponíveis | Transcrição real com critérios |
| T05 | Sessões/reset/limite | Isolamento, limpeza e poda funcionam | `pytest` + demonstração |
| T06 | Escopo e persona | Não assume outra jurisdição nem inventa resposta fora da base | Avaliação real de casos |
| T07 | Entrada maliciosa | Mantém regras diante de pedido para ignorar instruções e de instrução inserida em documento | Avaliação real; separar bloqueio local de resistência do modelo |
| T08 | Falha de serviço/parser | Retorno compreensível, sem expor chave ou resposta não validada | Testes de falhas |
| T09 | Context rot | Tabela/gráfico real, pergunta constante e tokens/qualidade medidos | Artefatos e script |
| T10 | Meta prompting | Proposta gerada pelo modelo e comparação v1/v2 reproduzível | Prompts, respostas e métricas |
| T11 | Proveniência | ≥ 5 documentos distintos, URLs, hashes e recortes auditáveis | Manifesto + revisão manual |
| T12 | Extração jurídica | Texto revogado identificado; artigos e permissões de tabelas preservados | Comparação com originais |
| T13 | Chunking | Separadores exigidos, sobreposição 10–15%, ≥ 2 configurações isoladas | Configuração + inspeção de chunks |
| T14 | Embeddings/Chroma | Query e documentos produzem vetores compatíveis; índice pode ser reconstruído | Integração real |
| T15 | Recuperação | Evidência necessária aparece nos resultados para as perguntas de referência | Lista de chunks por pergunta |
| T16 | Citações | IDs retornados pertencem ao contexto recuperado e resolvem para a fonte correta | Teste + conferência manual |
| T17 | Fundamentação | Sem artigos/obrigações inventados; abstém-se quando a base não responde | Revisão humana + RAGAS |
| T18 | Métricas | ≥ 5 perguntas/configuração, ambas as métricas, média faithfulness ≥ 0,7 | CSV/JSON/notebook |
| T19 | Filtros | `where=` restringe ao metadado solicitado; resultado vazio tratado | Integração + demonstração |
| T20 | Reranking | Candidatos são reordenados antes de gerar; impacto medido | Ranking antes/depois + métricas |
| T21 | Gradio RAG | Pergunta gera resposta e fontes acessíveis | Demonstração de fluxo completo |
| T22 | Colab novo | `Run All` sem erros, chave em Secret e documentos acessíveis | Notebook executado |
| T23 | Entrega segura | ZIP/notebook sem credenciais, `.env`, caches ou caminhos pessoais necessários | Inspeção dos artefatos finais |

**Exemplos de casos de escopo:** distinguir São Paulo capital do estado; questionar descarte em outro município; perguntar por horário/endereço não incluído na base; pedir valor de multa sem fundamento; fornecer resíduo sem volume; perguntar por material sem orientação nos documentos. Definir a resposta esperada somente após verificar as fontes.

**Comandos previstos de verificação:**

```bash
python -m pytest -q
ruff check app tests evals scripts
ruff format --check app tests evals scripts
python -m app.main
python -m app.context_rot
python -m app.meta_prompting
python -m evals.run_chat
python -m evals.run_rag
python scripts/package_ckp01.py
```

Os executores são interfaces planejadas, ainda não implementadas. Acrescentar `evals/__init__.py` para executar os módulos com `-m`. Separar testes offline de testes que fazem chamadas autenticadas, informando a chave e os requisitos de execução no README.

## 9. Checklist final e cronograma oficial

### CKP01

- [ ] Todos os critérios de 10 pontos estão demonstrados, além dos dois diferenciais.
- [ ] README contém nomes/RMs, domínio, motivação, público, instruções, justificativa da memória e context rot.
- [x] Projeto local novo; `python -m app.main` funcional; sem dependência de Colab.
- [x] ZIP contém `app/`, `.env.example`, `requirements.txt`, README e evidências necessárias.
- [ ] Testado o ZIP extraído em diretório limpo.
- [x] `.env` e chaves não constam da entrega.
- [ ] Líder envia via Teams até **23:55 do dia da Aula 05**.

Marcos: Aula 01 domínio; Aula 02 memória; Aula 03 chain com pelo menos três turnos; Aula 04 apresentação dos requisitos pelo professor; Aula 05 entrega. O PDF não prevê apresentação ao vivo pelo grupo.

### CKP02

- [x] Pelo menos cinco documentos reais com fontes no notebook e manifesto.
- [ ] Pipeline completo, modelo/embeddings exigidos e geração com temperatura zero.
- [x] Duas configurações de chunking, sobreposição correta, métricas por pergunta e escolha justificada.
- [x] Faithfulness médio ≥ 0,7; objetivo interno ≥ 0,9.
- [x] Metadata filtering, reranking e Gradio demonstrados e avaliados.
- [ ] README contém nomes/RMs e instruções para adicionar documentos.
- [x] Export `.ipynb` executado com serviços reais, sem erros (execução local).
- [ ] Compartilhar o Colab com permissão de leitura.
- [x] Documentos da base e módulos necessários acompanham a entrega.
- [ ] Notebook e outputs sem segredos; reprodução verificada em runtime novo.
- [ ] Líder envia via Teams até **23:59 do dia da Aula 08**.

Marcos: Aula 05 carga/chunks/embeddings e uma busca; Aula 06 pipeline completo e primeiras métricas; Aula 07 apresentação dos requisitos pelo professor; Aula 08 entrega. Organizar etapas iniciais do RAG em paralelo ao fechamento do CKP01 quando o calendário exigir.

## 10. Referências e pendências

### Referências primárias

- Enunciados locais: `CKP01_2Semestre_Chatbot_Profissional.pdf` e `CKP02_2Semestre_DocMind_RAG.pdf`, na pasta acima deste repositório.
- [ChargeGrid — referência do projeto anterior](https://github.com/luigiborghetif/ChargeGrid-Intelligence-Prompt-AI-Team7-1CCA-FIAP).
- [Ollama Cloud — modelos, API e alias](https://docs.ollama.com/cloud).
- [Ollama — autenticação](https://docs.ollama.com/api/authentication).
- [Modelo nomic-embed-text na biblioteca Ollama](https://ollama.com/library/nomic-embed-text).
- [LangChain Classic — memória de ConversationChain](https://reference.langchain.com/python/langchain-classic/chains/conversation/base/ConversationChain/memory).
- [RAGAS — referência das métricas na versão 0.2.14](https://docs.ragas.io/en/v0.2.14/references/metrics/). Referência conceitual; conferir adaptadores e API na documentação da versão efetivamente fixada.

### Pendências antes da execução

- Confirmar integrantes/RMs e registro do domínio na turma.
- Confirmar solução compatível para modelo Cloud, autenticação e embeddings do enunciado.
- Fixar versões após prova técnica local e no Colab.
- Revisar documentos oficiais e estabelecer a data de referência da base.
- Inserir datas civis das Aulas 05 e 08 quando disponibilizadas.
- Receber o PDF do CKP03 antes de planejar sua implementação detalhada.

Metas e etapas abertas continuam como planejamento. As execuções realizadas estão
identificadas no estado atual e nas pastas `evidence/ckp01` e `evidence/ckp02`.
Itens só são marcados como concluídos após implementação e verificação.
