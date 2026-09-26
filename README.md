# machine-learning-supervised

## "Black-Box Optimization with Gaussian Processes"
### FUNCAO ESCONDIDA
f(x)= sin(x) + 0.3 sin(3x) + 0.1x

Modelo não recebe essa função.

### PRIMEIRO TESTE
x = [0, 2, 4, 7, 10]
y = f(x)

Treina o GP -> GP decide proximo ponto -> Testa -> Adiciona novo Dado -> Treina -> Repete

### SEGUNDO TESTE
Upper Confidence Bound (UCB)

Primeira função de aquisição: UCB(x) = μ(x) + βσ(x)

onde:
- μ(x) = previsão do GP
- σ(x) = incerteza
- β = quanto queremos valorizar exploração

### TERCEIRO TESTE 
Aumento na Dimensao

e grafico de loop x y

### QUARTO TESTE
adicionar White Kernel
Separar o codigo main

## OBJETIVO
Depois algumas consultas, queremos descobrir o maior y.

Depois comparar Busca aleatória VS Busca usando GP