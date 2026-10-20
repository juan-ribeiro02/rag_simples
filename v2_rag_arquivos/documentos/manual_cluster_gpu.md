# Manual de Uso do Cluster de GPUs

## Solicitação de Acesso
Para solicitar acesso ao cluster de GPUs (placas de vídeo de alto desempenho), o aluno deve enviar um e-mail para o professor coordenador (`coordenador.lab@universidade.edu.br`) informando:
- Nome completo e matrícula;
- Nome do projeto de pesquisa ou disciplina;
- Estimativa de recursos necessários (memória VRAM e tempo de processamento).

## Regras de Utilização e Limites de Tempo
Cada usuário pode executar no máximo 2 jobs simultâneos na fila principal.
O limite máximo de tempo por execução contínua (job) é de **12 horas**. Processos que ultrapassarem esse limite sem autorização especial serão encerrados automaticamente pelo gerenciador SLURM.

## Armazenamento Temporário
Cada conta possui uma cota de 50 GB no diretório `/scratch`. Arquivos sem acesso há mais de 30 dias são removidos automaticamente.
