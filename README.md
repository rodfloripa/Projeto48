<p align="justify"><h1>1. Transformer com Memória Online por Cabeça de Atenção</h1></p>

<p align="justify">
Este projeto implementa uma arquitetura inspirada em mecanismos de <b>memória online</b> acoplados ao Transformer, onde cada cabeça de atenção possui sua própria memória treinável e atualizada dinamicamente durante o <i>forward pass</i>. A proposta aproxima conceitos discutidos em trabalhos recentes sobre memória associativa contínua e adaptação online sem necessidade de <i>backpropagation</i> completo para atualização da memória.
</p>

<p align="justify">
A principal ideia é permitir que cada cabeça de atenção mantenha um pequeno estado persistente capaz de armazenar correlações relevantes observadas ao longo da sequência. Diferente da atenção tradicional, onde toda a informação depende apenas do contexto instantâneo, aqui existe uma memória auxiliar capaz de reforçar padrões aprendidos durante a inferência.
</p>

---

<p align="justify"><h2>2. Objetivo</h2></p>

<p align="justify">
O objetivo foi investigar se pequenas matrizes de memória poderiam melhorar a capacidade do Transformer de recordar padrões recentes sem modificar drasticamente a arquitetura original.
</p>

<p align="justify">
A implementação adiciona:
</p>

* Memória independente por cabeça de atenção
* Atualização online da memória durante o forward
* Correção residual aplicada no output da atenção
* Controle de influência da memória via <i>gating</i>
* Reset explícito entre contextos diferentes

---

<p align="justify"><h2>3. Arquitetura Geral</h2></p>

```text
Entrada → Embedding → Transformer Layer
                                ↓
                      Multi-Head Attention
                                ↓
                     Memória por Cabeça
                                ↓
                 Correção Residual da Atenção
                                ↓
                           MLP + Norm
                                ↓
                              Saída
```

<p align="justify">
O fluxo principal do modelo continua seguindo a lógica tradicional do Transformer, porém agora existe um mecanismo auxiliar de memória atuando paralelamente à atenção.
</p>

<p align="justify">
A memória:
</p>

* recebe representações reduzidas do hidden state
* atualiza seus estados online
* devolve uma correção residual para a atenção

---

<p align="justify"><h2>4. Classe <code>HeadMemory</code></h2></p>

<p align="justify">
A classe <code>HeadMemory</code> implementa a unidade básica de memória online.
</p>

```python
class HeadMemory(nn.Module):

    def __init__(self, dim=8, lr=5e-3):
        super().__init__()

        self.dim = dim
        self.lr = lr

        self.register_buffer(
            "memory",
            torch.zeros(dim, dim)
        )

        self.key = nn.Parameter(
            torch.randn(dim) * 0.02
        )

        self.value = nn.Parameter(
            torch.randn(dim) * 0.02
        )

        self.W_q = nn.Parameter(
            torch.randn(dim) * 0.02
        )

        self.W_o = nn.Parameter(
            torch.randn(dim) * 0.02
        )
```

<p align="justify">
Essa classe possui:
</p>

| Componente | Função                        |
| ---------- | ----------------------------- |
| `memory`   | Matriz persistente da memória |
| `key`      | Vetor usado para projeção     |
| `value`    | Vetor alvo da atualização     |
| `W_q`      | Correção do espaço de queries |
| `W_o`      | Correção do output            |

---

<p align="justify"><h3>4.1 Memória Persistente</h3></p>

```python
self.register_buffer(
    "memory",
    torch.zeros(dim, dim)
)
```

<p align="justify">
A memória é registrada como <i>buffer</i>, e não como parâmetro treinável tradicional.
</p>

<p align="justify">
Isso significa que:
</p>

* ela não participa diretamente do gradiente
* continua sendo salva no checkpoint
* permanece no dispositivo correto (`cpu` ou `cuda`)

---

<p align="justify"><h3>4.2 Vetores Treináveis</h3></p>

```python
self.key = nn.Parameter(torch.randn(dim) * 0.02)
self.value = nn.Parameter(torch.randn(dim) * 0.02)

self.W_q = nn.Parameter(torch.randn(dim) * 0.02)
self.W_o = nn.Parameter(torch.randn(dim) * 0.02)
```

<p align="justify">
Esses são os únicos componentes aprendidos via backpropagation tradicional.
</p>

<p align="justify">
Eles funcionam como:
</p>

* mecanismos de leitura
* vetores-alvo
* controladores da influência da memória

---

<p align="justify"><h2>5. Atualização Online da Memória</h2></p>

```python
def update(self, h):

    h = h.detach()

    pred = self.memory @ self.key

    err = self.value - pred

    with torch.no_grad():
        self.memory += (
            self.lr *
            torch.outer(err, self.key)
        )
```

<p align="justify">
Esse trecho é o núcleo do aprendizado online.
</p>

<p align="justify">
A lógica funciona em quatro etapas:
</p>

| Etapa | Operação                             |
| ----- | ------------------------------------ |
| 1     | A memória gera uma predição          |
| 2     | Calcula-se o erro                    |
| 3     | O erro é transformado em atualização |
| 4     | A memória é corrigida                |

---

<p align="justify"><h3>5.1 Uso de <code>detach()</code></h3></p>

```python
h = h.detach()
```

<p align="justify">
A memória aprende fora do fluxo tradicional de gradiente.
</p>

<p align="justify">
O `detach()` impede:
</p>

* crescimento excessivo do grafo computacional
* propagação do gradiente pela memória online
* explosão de memória GPU

---

<p align="justify"><h3>5.2 Atualização Tipo Hebb/SGD</h3></p>

```python
self.memory += (
    self.lr *
    torch.outer(err, self.key)
)
```

<p align="justify">
A atualização lembra mecanismos:
</p>

* Hebbian learning
* Fast weights
* Associative memory
* SGD online local

<p align="justify">
A matriz aprende relações através do produto externo entre erro e chave.
</p>

---

<p align="justify"><h2>6. Leitura da Memória</h2></p>

```python
def read(self, h):

    out = h @ self.memory

    return (
        out * self.W_q,
        out * self.W_o
    )
```

<p align="justify">
Após a atualização, a memória pode ser consultada.
</p>

<p align="justify">
A leitura:
</p>

1. projeta o hidden state na memória
2. recupera padrões armazenados
3. gera vetores corretivos

---

<p align="justify"><h3>6.1 Correções Separadas</h3></p>

```python
return (
    out * self.W_q,
    out * self.W_o
)
```

<p align="justify">
O modelo separa:
</p>

| Correção | Uso               |
| -------- | ----------------- |
| `W_q`    | Ajuste de queries |
| `W_o`    | Ajuste do output  |

<p align="justify">
No experimento atual apenas a correção do output foi utilizada diretamente.
</p>

---

<p align="justify"><h2>7. MultiHeadMemory</h2></p>

```python
class MultiHeadMemory(nn.Module):

    def __init__(
        self,
        n_heads,
        mem_dim=8,
        lr=5e-3
    ):
```

<p align="justify">
Essa classe cria múltiplas memórias independentes.
</p>

---

<p align="justify"><h3>7.1 Criação das Memórias</h3></p>

```python
self.heads = nn.ModuleList([
    HeadMemory(
        mem_dim // n_heads,
        lr
    )
    for _ in range(n_heads)
])
```

<p align="justify">
Cada cabeça recebe:
</p>

* sua própria matriz
* seus próprios vetores
* seu próprio estado persistente

<p align="justify">
Isso reduz interferência entre representações diferentes.
</p>

---

<p align="justify"><h3>7.2 Atualização de Todas as Cabeças</h3></p>

```python
for b_idx in range(B):

    for head_idx in range(self.n_heads):

        h_for_head = (
            h_split_list[head_idx][b_idx]
        )

        self.heads[head_idx].update(
            h_for_head
        )
```

<p align="justify">
A atualização percorre:
</p>

1. batch
2. cabeça
3. token

<p align="justify">
Cada cabeça aprende independentemente.
</p>

---

<p align="justify"><h2>8. TransformerLayerWithMemory</h2></p>

```python
class TransformerLayerWithMemory(nn.Module):
```

<p align="justify">
Essa camada integra:
</p>

* atenção tradicional
* memória online
* correção residual
* MLP do Transformer

---

<p align="justify"><h3>8.1 Atenção Multi-Head Tradicional</h3></p>

```python
self.attn = nn.MultiheadAttention(
    hidden_dim,
    n_heads,
    batch_first=True
)
```

<p align="justify">
O núcleo original do Transformer foi preservado.
</p>

---

<p align="justify"><h3>8.2 Projeção para Espaço da Memória</h3></p>

```python
self.down_proj = nn.Linear(
    hidden_dim,
    mem_dim
)

self.up_proj = nn.Linear(
    mem_dim,
    hidden_dim
)
```

<p align="justify">
Essas projeções:
</p>

| Camada      | Função                |
| ----------- | --------------------- |
| `down_proj` | Reduz dimensão        |
| `up_proj`   | Retorna ao hidden_dim |

<p align="justify">
Isso reduz custo computacional da memória.
</p>

---

<p align="justify"><h3>8.3 Divisão por Cabeças</h3></p>

```python
h_split = h.chunk(
    self.n_heads,
    dim=-1
)
```

<p align="justify">
Cada cabeça recebe apenas seu próprio subespaço.
</p>

<p align="justify">
Esse comportamento mantém coerência com a atenção multi-head tradicional.
</p>

---

<p align="justify"><h3>8.4 Atualização Temporal</h3></p>

```python
for t in range(S):

    h_t = torch.cat([
        hs[:, t]
        for hs in h_split
    ], dim=-1)

    h_t_split = h_t.chunk(
        self.n_heads,
        dim=-1
    )

    self.memory.update_all(
        h_t_split
    )
```

<p align="justify">
A memória aprende token por token durante o forward.
</p>

<p align="justify">
Isso cria comportamento próximo de aprendizado contínuo.
</p>

---

<p align="justify"><h3>8.5 Leitura da Memória</h3></p>

```python
h_last_split = [
    hs[:, -1]
    for hs in h_split
]

corr_q, corr_o = self.memory.read_all(
    h_last_split
)
```

<p align="justify">
A leitura ocorre usando o último token da sequência.
</p>

<p align="justify">
A memória gera vetores corretivos para modificar a atenção.
</p>

---

<p align="justify"><h3>8.6 Reconstrução do Hidden Space</h3></p>

```python
corr_o_flat = corr_o.flatten(1)

corr_o_full = self.up_proj(
    corr_o_flat
)
```

<p align="justify">
As correções das cabeças são:
</p>

1. concatenadas
2. projetadas de volta
3. reinseridas no espaço original

---

<p align="justify"><h3>8.7 Correção Residual da Atenção</h3></p>

```python
attn_out[:, -1] += (
    corr_o_full * 1.0
)
```

<p align="justify">
Esse foi o ponto mais importante do experimento.
</p>

<p align="justify">
Inicialmente:
</p>

```python
corr_o_full * 0.1
```

<p align="justify">
A memória influenciava muito pouco.
</p>

<p align="justify">
Após aumentar para:
</p>

```python
1.0
```

<p align="justify">
o modelo passou a recuperar corretamente o próximo token.
</p>

---

<p align="justify"><h2>9. O Que Melhorou o Modelo</h2></p>

| Ajuste                | Impacto                                      |
| --------------------- | -------------------------------------------- |
| `corr_o_full * 1.0`   | Memória passou a afetar fortemente a atenção |
| `mem_dim=32`          | Maior capacidade de armazenamento            |
| Memória por cabeça    | Melhor separação de padrões                  |
| Atualização online    | Adaptação dinâmica                           |
| `lr=5e-3`             | Atualização rápida                           |
| Reset entre contextos | Evitou contaminação                          |

---

<p align="justify"><h2>10. Loop de Treinamento</h2></p>

```python
for step in range(500):

    opt.zero_grad()

    logits = model(
        seq,
        update_memory=True
    )

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 1000),
        seq[:, 1:].reshape(-1)
    )

    loss.backward()

    opt.step()
```

<p align="justify">
O treinamento usa predição autoregressiva tradicional:
</p>

* entrada → tokens anteriores
* alvo → próximo token

---

<p align="justify"><h2>11. Resultado Obtido</h2></p>

```text
Step 0, Loss: 6.834
Step 50, Loss: 0.371
Step 100, Loss: 0.111
Step 150, Loss: 0.059
Step 200, Loss: 0.038
Step 250, Loss: 0.027
Step 300, Loss: 0.020
Step 350, Loss: 0.016
Step 400, Loss: 0.013
Step 450, Loss: 0.010

Pred: 308
Real: 308
```

<p align="justify">
O modelo conseguiu:
</p>

* convergir rapidamente
* memorizar a sequência
* prever corretamente o próximo token

---

<p align="justify"><h2>12. Reset da Memória</h2></p>

```python
model.reset_memory()
```

<p align="justify">
Esse passo é obrigatório entre contextos independentes.
</p>

<p align="justify">
Sem reset:
</p>

* padrões antigos continuam armazenados
* ocorre interferência entre sequências
* a memória degrada

---

<p align="justify"><h2>13. Relação com Trabalhos Modernos</h2></p>

<p align="justify">
A arquitetura aproxima conceitos encontrados em:
</p>

* Fast Weights
* Associative Memory
* Online Learning
* Continual Learning
* Persistent Memory Transformers
* Retrieval-Free Memory

---

<p align="justify"><h2>14. Possíveis Melhorias Futuras</h2></p>

* Gating adaptativo aprendível
* Compressão dinâmica
* Memória hierárquica
* Controle de esquecimento
* Integração com Llama/Mistral
* Persistência entre sessões
* Atualização parcialmente diferenciável

---

<p align="justify"><h2>15. Conclusão</h2></p>

<p align="justify">
O experimento mostrou que pequenas memórias online podem modificar significativamente o comportamento do Transformer sem alterar os pesos principais do modelo.
</p>

<p align="justify">
O principal fator responsável pela melhora foi o aumento do fator de influência da memória:
</p>

```python
corr_o_full * 1.0
```

<p align="justify">
Isso permitiu que a memória deixasse de ser apenas um componente auxiliar e passasse a contribuir efetivamente para a predição.
</p>

<p align="justify">
Mesmo sendo um experimento compacto, a arquitetura já demonstra:
</p>

* adaptação online
* aprendizado contínuo
* memória persistente
* especialização por cabeça
* recuperação dinâmica de padrões

<p align="justify">
Os resultados sugerem que mecanismos leves de memória online podem ser uma direção promissora para ampliar a capacidade contextual de Transformers modernos sem necessidade de re-treinamento massivo.
</p>
