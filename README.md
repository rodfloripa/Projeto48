<p align="justify"><h2>11.1 Explicação Detalhada do Teste de Predição do Token 308</h2></p>

<p align="justify">
O teste final foi construído para verificar se a memória realmente conseguiu armazenar informação útil da sequência e reutilizá-la posteriormente durante a inferência.
</p>

<p align="justify">
O procedimento executado foi:
</p>

1. Treinar o modelo para memorizar uma sequência sintética
2. Limpar completamente a memória
3. Alimentar apenas parte da sequência
4. Verificar se o modelo consegue prever corretamente o próximo token

---

<p align="justify"><h3>11.1.1 Geração da Sequência</h3></p>

```python id="8x7n1p"
seq = torch.randint(
    0,
    500,
    (1, 20)
).to(device)
```

<p align="justify">
Foi criada uma sequência aleatória contendo:
</p>

* batch size = 1
* comprimento = 20 tokens
* tokens entre 0 e 499

<p align="justify">
Exemplo hipotético:
</p>

```text id="x9l8ra"
[14, 92, 7, 301, 88, 55, 401, 308, ...]
```

<p align="justify">
O objetivo do treinamento era fazer o Transformer aprender dependências autoregressivas dessa sequência.
</p>

---

<p align="justify"><h3>11.1.2 Processo de Treinamento</h3></p>

```python id="9u0v1x"
logits = model(
    seq,
    update_memory=True
)
```

<p align="justify">
Durante o treinamento:
</p>

* o modelo processa toda a sequência
* a memória é atualizada token por token
* as matrizes internas armazenam correlações úteis

<p align="justify">
A loss utilizada foi:
</p>

```python id="2zshxk"
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, 1000),
    seq[:, 1:].reshape(-1)
)
```

<p align="justify">
Isso significa:
</p>

| Entrada     | Alvo          |
| ----------- | ------------- |
| token atual | próximo token |

<p align="justify">
Ou seja, o modelo aprende a prever o próximo elemento da sequência.
</p>

---

<p align="justify"><h3>11.1.3 Reset da Memória</h3></p>

```python id="nq1lwb"
model.reset_memory()
```

<p align="justify">
Após o treinamento, toda a memória foi zerada.
</p>

<p align="justify">
Isso é extremamente importante.
</p>

<p align="justify">
Sem esse reset:
</p>

* o teste ficaria contaminado
* a memória ainda conteria estados antigos
* não seria possível saber se o modelo realmente reaprendeu durante a inferência

<p align="justify">
O reset garante um teste limpo.
</p>

---

<p align="justify"><h3>11.1.4 Alimentando Apenas o Prefixo</h3></p>

```python id="6cwl1d"
logits_prefix = model(
    seq[:, :-5],
    update_memory=True
)
```

<p align="justify">
Aqui acontece a parte mais importante do experimento.
</p>

<p align="justify">
O modelo NÃO recebe a sequência completa.
</p>

<p align="justify">
Ele recebe apenas:
</p>

```python id="s1g8kt"
seq[:, :-5]
```

<p align="justify">
Ou seja:
</p>

* os últimos 5 tokens foram escondidos
* o modelo precisa prever o próximo token usando apenas o contexto anterior
* a memória online continua sendo atualizada enquanto lê o prefixo

---

<p align="justify"><h3>11.1.5 Predição do Próximo Token</h3></p>

```python id="o6f7uv"
pred_next_token = (
    logits_prefix[:, -1, :]
    .argmax(-1)
)
```

<p align="justify">
Esse trecho pega:
</p>

* o último vetor de logits
* correspondente ao último token visível do prefixo

<p align="justify">
Depois:
</p>

* aplica `argmax`
* seleciona o token mais provável

<p align="justify">
Ou seja:
</p>

```text id="x7l6gh"
"Qual token o modelo acredita que vem depois?"
```

---

<p align="justify"><h3>11.1.6 Token Real Esperado</h3></p>

```python id="p7yk4h"
real_next_token = seq[:, -5]
```

<p align="justify">
O token correto era exatamente o primeiro token ocultado.
</p>

<p align="justify">
No experimento:
</p>

```text id="8r7y5v"
Real: 308
```

<p align="justify">
Isso significa:
</p>

* o próximo token correto da sequência era `308`
* o modelo precisava inferir isso apenas pelo contexto anterior

---

<p align="justify"><h3>11.1.7 Resultado Final</h3></p>

```text id="7s6d2w"
Pred: 308
Real: 308
```

<p align="justify">
O modelo acertou exatamente o próximo token esperado.
</p>

<p align="justify">
Isso demonstra que:
</p>

* a memória online armazenou informação útil
* a correção residual realmente influenciou a atenção
* o mecanismo conseguiu recuperar padrões aprendidos

---

<p align="justify"><h3>11.1.8 Por Que o Fator 1.0 Foi Importante</h3></p>

<p align="justify">
Antes da alteração:
</p>

```python id="0w6m1p"
corr_o_full * 0.1
```

<p align="justify">
a memória influenciava muito pouco a saída da atenção.
</p>

<p align="justify">
Mesmo que a memória armazenasse padrões corretos:
</p>

* o efeito residual era pequeno
* a atenção tradicional dominava
* o modelo frequentemente errava a predição

<p align="justify">
Após alterar para:
</p>

```python id="6l2k5f"
corr_o_full * 1.0
```

<p align="justify">
a contribuição da memória passou a ter magnitude suficiente para alterar significativamente o output final.
</p>

<p align="justify">
Foi exatamente essa mudança que permitiu:
</p>

* recuperar corretamente o token `308`
* reforçar padrões aprendidos
* estabilizar a recuperação contextual

---

<p align="justify"><h3>11.1.9 Interpretação do Experimento</h3></p>

<p align="justify">
O mais importante desse resultado não é apenas acertar um token aleatório.
</p>

<p align="justify">
O ponto central é que:
</p>

* a memória online conseguiu participar ativamente da inferência
* o Transformer passou a utilizar um estado persistente auxiliar
* houve aprendizado local durante o próprio forward pass

<p align="justify">
Isso aproxima o comportamento do modelo de mecanismos:
</p>

* associativos
* contínuos
* adaptativos
* persistentes

<p align="justify">
Mesmo sendo um experimento pequeno, ele demonstra que memórias online leves podem alterar significativamente a dinâmica interna de Transformers modernos.
</p>
