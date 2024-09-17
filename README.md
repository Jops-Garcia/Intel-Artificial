# Atividade: Aprendizado por Reforço - Lunar Lander e Car Racing

## Descrição
Este projeto faz parte da disciplina de **Inteligência Artificial** do curso de **Bacharelado em Sistemas de Informação**. O objetivo foi treinar uma Inteligência Artificial (IA) utilizando **Aprendizado por Reforço** nos ambientes **Lunar Lander** e **Car Racing**, ambos do Gymnasium.

## Objetivo
Treinar agentes para executar as seguintes tarefas:
- **Lunar Lander**: Controlar um módulo lunar para pousar suavemente em uma plataforma específica, evitando obstáculos.
- **Car Racing**: Dirigir um carro em uma pista, maximizando a pontuação ao percorrer a maior distância possível.

## Algoritmo Utilizado
O algoritmo utilizado para o treinamento foi o **Proximal Policy Optimization (PPO)**. O PPO é um dos algoritmos mais eficientes em aprendizado por reforço, especialmente quando se trata de ambientes contínuos como os utilizados nesta atividade.

### Principais Características do PPO:
- Atualizações restritas da política, o que melhora a estabilidade do treinamento.
- Equilíbrio entre exploração e exploração das políticas aprendidas.
- Compatível com ambientes de ação contínua como o Car Racing.

## Ferramentas Utilizadas
- **Linguagem**: Python
- **Bibliotecas**:
  - `Gymnasium`: Para simulação dos ambientes Lunar Lander e Car Racing.
  - `Stable-Baselines3`: Para implementação do algoritmo PPO.
  - `Matplotlib` (opcional): Para visualização dos resultados.

## Treinamento
### Lunar Lander
- O objetivo do agente é controlar o módulo lunar usando empuxo vertical e horizontal para pousar na plataforma de forma controlada.
- A recompensa é dada com base na proximidade da plataforma e pela suavidade do pouso.
- Penalidades são aplicadas para colisões com obstáculos.

### Car Racing
- O agente deve aprender a dirigir eficientemente ao longo da pista, maximizar a pontuação, e evitar sair da pista ou bater nas laterais.

## Resultados
- **Lunar Lander**: Após múltiplas iterações de treinamento, o agente conseguiu aprender a pousar consistentemente na plataforma, minimizando colisões e penalidades.
- **Car Racing**: O agente apresentou uma performance estável, conseguindo percorrer a maior parte das pistas sem sair dos limites, mas ainda pode melhorar em termos de eficiência de direção.

## Conclusão
A atividade demonstrou o poder do aprendizado por reforço em resolver problemas complexos de controle contínuo, como pousar uma nave ou dirigir um carro. A aplicação do algoritmo PPO mostrou-se adequada para lidar com as características dos ambientes Lunar Lander e Car Racing.
