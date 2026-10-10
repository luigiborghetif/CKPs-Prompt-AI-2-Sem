# Evidências CKP02 — execução híbrida real

Geração e juiz: Gemma no Ollama Cloud, temperatura zero. Embeddings: `nomic-embed-text` no Ollama local, 768 dimensões. Configuração autorizada pelo usuário em 10/10/2026.

## Comparação final

Pasta: `rag_comparison_20261010T232440Z_22eb7be5`. Dez perguntas iguais em cada configuração; todas as 30 respostas foram geradas, avaliadas e encontraram a fonte esperada. Não houve abstenção nem erro nesse conjunto.

| Configuração | Faithfulness média | Answer relevancy média | Latência média (s) | Erros |
|---|---:|---:|---:|---:|
| 256_32_baseline | 1.0000 | 0.8583 | 5.109 | 0 |
| 512_64_baseline | 1.0000 | 0.8581 | 5.211 | 0 |
| 512_64_rerank | 1.0000 | 0.8571 | 6.587 | 0 |

Escolha: **256/32 sem reranking**, aplicada ao `.env` e ao exemplo. Faithfulness empatou em 1,0; o critério usa relevância no desempate. A diferença entre os dois baselines é cerca de 0,00015, pequena demais para alegar superioridade robusta. O reranking permaneceu disponível e foi demonstrado; não houve vantagem medida neste conjunto final.

`results.json` guarda respostas, afirmações/citações, contextos e métricas por pergunta; `metrics.csv`, `summary.json`, `selection.json` e `comparison.png` apresentam resultados. `metadata.json` registra modelos/endpoints, parâmetros e hashes do corpus/perguntas. O juiz é o mesmo Gemma do gerador; as métricas não dispensam revisão de fundamentação nem garantem resultado idêntico em outras consultas.

## Verificações finais

- `connection_check.json`: embed_query/embed_documents, 768 dimensões, pipeline e RAGAS reais aprovados.
- `gradio_smoke.json`: consulta real pela interface localhost, com resposta e citações válidas.
- Na pasta final, `metadata_filter.json`: filtro tipo/categoria/ano e ordenação antes/depois do CrossEncoder; a ordenação mudou.
- Na pasta final, `boundary_cases.json`: 5/5 casos aprovados (outra cidade, horário/multa sem base, esclarecimento e injeção).
- `notebooks/CKP02_DocMind_SP.ipynb`: Run All local com serviços reais, geração, filtros e casos de limite. As métricas foram importadas da comparação final usando `REUSE_EVIDENCE`, com validação de hashes e parâmetros; não foram simuladas nem recalculadas desnecessariamente.
- 69 testes offline aprovados; Ruff e dependências sem conflitos.
- `corpus_chunks.png`: contagens reais do corpus (184 chunks em 256/32; 88 em 512/64), não métricas de resposta.

## Histórico de correções

`rag_comparison_20261010T231631Z_000621ee` é a tentativa inicial com falhas: JSON com conteúdo factual no campo message, reuso de cliente HTTP em loops encerrados e avaliação sem metadados de fonte. O código foi corrigido sem relaxar validação de citações/schema. Essa execução não é a comparação final.
`rag_comparison_20261010T232044Z_4899b3f3` concluiu sem erros, mas revelou abstenção em uma pergunta sobre custos. A recuperação passou a filtrar o artigo explicitamente citado pelo usuário; a comparação final respondeu essa pergunta nas três configurações. Todos os históricos foram preservados.

## Diferença do enunciado e entrega

`connection_check_cloud.json` registra a recusa HTTP 401 nos embeddings Cloud. A decisão posterior do usuário autorizou baixar e usar embeddings locais. O PDF pede embeddings Cloud: confirmar a alternativa com o professor e manter essa diferença explícita. Não foi trocado o modelo de embeddings nem de geração.
No Colab hospedado, localhost é o runtime do Colab. Instalar/iniciar Ollama nesse runtime ou conectar um runtime local antes de recalcular. O notebook salvo já contém outputs reais e pode ser compartilhado para leitura. Ainda preencher integrantes/RMs, criar o link de leitura no Colab e enviar pelo líder.
