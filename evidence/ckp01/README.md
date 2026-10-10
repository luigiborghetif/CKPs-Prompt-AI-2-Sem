# Execuções reais — CKP01

Executadas em 10/10/2026 com Gemma no Ollama Cloud, temperatura zero e configuração
registrada em cada `metadata.json`. Nenhuma credencial acompanha as evidências.

| Experimento | Pasta | Resultado |
|---|---|---|
| Consultas e memória | `chat_20261010T231144Z_c75e803b` | 8/8 casos aprovados, incluindo conversa de seis turnos; zero erros |
| Contexto crescente | `context_rot_20261010T231215Z_3f5f0b7c` | 15/15 execuções válidas; fração de fatos corretos = 1,0 em todas as janelas |
| Meta prompting | `meta_prompting_20261010T231226Z_7390305e` | v1: 8/8 e v2: 8/8; zero erros em ambas |

No contexto crescente, as entradas variaram de 410 a 6.490 tokens estimados,
com três repetições por janela. Não houve degradação observada neste experimento;
isso não comprova ausência de context rot em contextos maiores ou tarefas distintas.
O gráfico, as respostas brutas, os inputs e os resumos estão na pasta indicada.

No meta prompting, o modelo produziu uma proposta com os placeholders/seções
exigidos. A comparação não demonstrou ganho nos verificadores; o prompt original
foi mantido. A revisão das respostas encontrou limitações que os verificadores
não medem integralmente: a proposta recomenda medir volume sem fonte explícita,
e a indicação da página para localizar unidades vai além do recorte local. As
duas versões exigem revisão humana de fundamentação; 8/8 não significa ausência
de toda inferência ou resistência garantida a qualquer injeção.

As pastas `chat_20261010T231130Z_d8a6ff38`,
`context_rot_20261010T231157Z_943f3504` e
`meta_prompting_20261010T231216Z_873d354e` são tentativas com rede
bloqueada no sandbox. Não usar como evidência de qualidade do modelo. Os resultados
foram preservados para auditoria; as execuções posteriores acima usaram acesso
de rede autorizado.
