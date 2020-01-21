
<p><h1>SFW <small> - Stefanini Fashion Week</small></h1></p>

A Stefanini estava se preparando para realizar a primeira Fashion Week tecnológica do mundo, sua roupas são feitas com tecnologia de ponta, algumas contendo até inteligência artificial para se adaptar aos gostos que seus clientes. Porém, como foi a primeira vez que a Stefanini está realizando um evento desse estilo nem tudo poderia ser perfeito, as roupas que deveriam chegar uma semana antes estavam previstas para chegar algumas horas antes do evento.

Como é uma quantidade absurda de roupas, um ser humano não ia conseguir classificar e separar tudo a tempo, porém como a Stefanini tem funcionários experientes na área de inteligência artificial, foi então demandado a eles treinar um algoritmo capaz de classificar as roupas entre:

- 0 - Camiseta
- 1 - Calça
- 2 - Pulôver
- 3 - Vestido
- 4 - Casaco
- 5 - Sandália
- 6 - Camisa
- 7 - Tênis
- 8 - Bolsa
- 9 - Tornozeleira

O Dataset a ser utilizado para desenvolver esse modelo foi o famoso Fashion-Mnist e foi carregado pelo tensorflow.

É exigido que para esse modelo seja feito um <b>Multilayer Perceptron</b>. O Framework a ser escolhido é de preferência do desenvolvedor.

<b>É necessário por comentários explicando o código</b>

\* <small>Quem desenvolver o modelo em Numpy terá uma melhor avaliação que os demais</small>

### Importações necessárias

# Exercício
## Autor: Matheus Jericó Palhares
## Email: matheusjerico1994@hotmail.com
## linkedIn: https://www.linkedin.com/in/matheusjerico

## MLP com framework Keras;


```python
import matplotlib.pyplot  as plt
from tensorflow.keras.datasets import fashion_mnist
from keras import backend as K
```

    Using TensorFlow backend.


#### Carregando o dataset a ser trabalhado


```python
# Separando dados de treino e teste
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Verificando formato das imagens
print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("X_test.shape: {}".format(X_test.shape))
print("y_test.shape: {}".format(y_test.shape))
```

    X_train.shape: (60000, 28, 28)
    y_train.shape: (60000,)
    X_test.shape: (10000, 28, 28)
    y_test.shape: (10000,)



```python
# Mostrando o dataset
fig, axs = plt.subplots(3,10, figsize=(25,5))
axs = axs.flatten()

for img, ax in zip(X_train[:30], axs):
    ax.imshow(img, cmap='gray')
    ax.axis('off')

print('Exemplos de imagens: ')
plt.show()
```

    Exemplos de imagens: 



![png](imagens/output_5_1.png)


## To be continued...

## - Implementando MLP com o framework Keras


```python
# Importando biblioteca numpy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, multilabel_confusion_matrix
```

## 1. Pré-processamento dos dados

- Fazendo reshape:
    - Backend utilizando o tensor_flow: O formato da imagem é 'channels_last', coloca-se a profundidade da imagem no final;
    - Backend utilizando o Theano: O formato da imagem é 'channels_first', coloca-se a profundidade da imagem no início.
    
O código abaixo eu automatizo essa decisão, utilizando a função keras.backend.image_data_format(), a resposta da função é uma string: 'channels_first' para Theano como beckend e 'channels_last' para tensor_flow como beckend.

Como foi usado o tensor_flow de backend para o Keras, então o formato da imagem é 'channels_last'.

Já seto o parâmetro (input_shape) a ser passado na camada de entrada da MLP.


```python
imagem_linhas = 28
imagem_colunas = 28

if K.image_data_format() == 'channels_first':
    print("image_data_format: channels_first")
    X_train = X_train.reshape(X_train.shape[0], 1, imagem_linhas, imagem_colunas)
    X_test = X_test.reshape(X_test.shape[0], 1, imagem_linhas, imagem_colunas)
    inputShape = (1, imagem_linhas, imagem_colunas)
else:
    print("image_data_format: channels_last")
    X_train = X_train.reshape(X_train.shape[0], imagem_linhas, imagem_colunas, 1)
    X_test = X_test.reshape(X_test.shape[0], imagem_linhas, imagem_colunas, 1)
    inputShape = (imagem_linhas, imagem_colunas, 1)
```

    image_data_format: channels_last


- Após realizar o reshape da imagem adicionando a profundidade, verifico a modificação:


```python
# Confirmando as dimensões dos X's
print('X_train.shape: {}'.format(X_train.shape))
print('X_test.shape: {}'.format(X_test.shape))
```

    X_train.shape: (60000, 28, 28, 1)
    X_test.shape: (10000, 28, 28, 1)


#### 1.1 Pré-processamento das variáveis X's:
    - Transformo em array do tipo 'float' e divido por 255 para normalização os dados;
    - Os valores das matrizes das imagens que antes estavam entre 0 a 255, agora variam de 0 a 1.


```python
# Transformando dados em float32 e normalizando (dados com valores de 0 até 1)
X_train = np.array(X_train, dtype='float32') / 255.
X_test = np.array(X_test, dtype='float32') / 255.
```

#### 1.2 Pré-processamento das variáveis Y's:
    - Faço a verificação do formato dos labels, estão variando de 0 a 9 (apenas uma dimensão);
    - Faço a categorização para transformar os labels de uma dimensão para dez dimensões;

Exemplo:
    - O label do número '2' é o 2.
    - Após a categorização o label '2' vai passar a ser uma representação em formato de array com múltiplas dimensões:
    - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    


```python
# Verificando o formato dos labels
print("y_train.shape: {}".format(y_train.shape))

# Visualizando os 5 primeiros labels no y_train
print('Os 5 primeiros labels do y_train: {}'.format(y_train[:5]))
```

    y_train.shape: (60000,)
    Os 5 primeiros labels do y_train: [9 0 0 3 0]


- Faço a categorização dos labels:


```python
# Categorizando dos Y's
# Converter o array que possui apenas 1 dimensão, em um array com 10 dimensões.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```


```python
# Verificando as dimensões do array.
print('y_train.shape: {}'.format(y_train.shape))
print('y_test.shape: {}'.format(y_test.shape))

# Visualizando os 5 primeiros labels no y_train após categorização
print('Os 5 primeiros labels do y_train:\n {}'.format(y_train[:5]))
```

    y_train.shape: (60000, 10)
    y_test.shape: (10000, 10)
    Os 5 primeiros labels do y_train:
     [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]


#### 1.3 Colocando os nomes das classes em uma variável para apresentação posterior


```python
# Colocando os nomes das classes em uma variável
nomeClasses = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco", "Sandália", "Camisa", "Tênis", "Bolsa", "Tornozeleira"]
```

## 2. Criando modelo com MLP

- Flatten(): Transformar a imagem que está em formato de matriz para o formato de vetor;
- Dense(): Camada densa da Rede neural, a primeira camada Dense tem que ter o parâmetro de input_shape com a dimensão da imagem e profundidade. A quantidade de nodes para a primeira camada foi 28 x 28 (tamanho da imagem);
- Activation(): Função de ativação, optei por utilizar a ReLu pois tenho um bom resultado com ela;
- BatchNormalization(): Utilizado para normalizar as ativações da camada anterior, obtive uma melhora na acurácia e uma menor probabilidade de ocorrer overfitting;
- Dropout(): Aplicado para prevenir overfitting durante o treinamento, 'cancela' 20% (parâmetro selecionado) das ligações entre nodes.

Reduzi o número de nodes pela metade em cada camada Densa, até que na camada de saída tenho 10 nodes, pois temos 10 classes a serem classificadas.

Função utilizada na camada de saída foi a 'softmax', apresenta a probabilidade de classificação de cada classe, fazendo a selecão da classe que tem maior probabilidade.



```python
# Criando o modelo
model = Sequential()

# Flatten
model.add(Flatten())

# Camada de entrada
model.add(Dense(784, input_shape = (28, 28, 1)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Camada oculta
model.add(Dense(392))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Camada oculta
model.add(Dense(196))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Camada oculta
model.add(Dense(98))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Camada de saida
model.add(Dense(10))
model.add(Activation("softmax"))
```


```python
# Parametros
NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 32
EPOCHS = 40

# Optimizador
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
```

    WARNING:tensorflow:From /home/matheusjerico/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.


### 2.1 Compilando o Modelo
- loss: 'categorical_crossentropy' utilizado pois é uma classificação com mais de 2 classes;
- optimizer: SGD. Backpropagation.


```python
# Compilar modelo
model.compile(loss='categorical_crossentropy', optimizer= opt, metrics= ['accuracy'])
```

### 2.2 Treinando o modelo
- Passo os dados de treino, tamanho do lote, épocas e os dados de validação.


```python
# Fit no modelo
H1 = model.fit(X_train, y_train, batch_size=BS, epochs= EPOCHS, verbose=1, validation_data=(X_test, y_test))
```

    WARNING:tensorflow:From /home/matheusjerico/anaconda3/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    WARNING:tensorflow:From /home/matheusjerico/anaconda3/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/40
    60000/60000 [==============================] - 44s 741us/step - loss: 0.5765 - acc: 0.7930 - val_loss: 0.4325 - val_acc: 0.8397
    Epoch 2/40
    60000/60000 [==============================] - 43s 717us/step - loss: 0.4419 - acc: 0.8407 - val_loss: 0.3805 - val_acc: 0.8642
    Epoch 3/40
    60000/60000 [==============================] - 44s 734us/step - loss: 0.4033 - acc: 0.8547 - val_loss: 0.4059 - val_acc: 0.8552
    Epoch 4/40
    60000/60000 [==============================] - 49s 820us/step - loss: 0.3800 - acc: 0.8629 - val_loss: 0.3603 - val_acc: 0.8687
    Epoch 5/40
    60000/60000 [==============================] - 40s 671us/step - loss: 0.3650 - acc: 0.8669 - val_loss: 0.3419 - val_acc: 0.8749
    Epoch 6/40
    60000/60000 [==============================] - 43s 717us/step - loss: 0.3500 - acc: 0.8724 - val_loss: 0.3348 - val_acc: 0.8763
    Epoch 7/40
    60000/60000 [==============================] - 38s 626us/step - loss: 0.3389 - acc: 0.8766 - val_loss: 0.3299 - val_acc: 0.8816
    Epoch 8/40
    60000/60000 [==============================] - 41s 685us/step - loss: 0.3313 - acc: 0.8786 - val_loss: 0.3225 - val_acc: 0.8834
    Epoch 9/40
    60000/60000 [==============================] - 43s 723us/step - loss: 0.3223 - acc: 0.8824 - val_loss: 0.3212 - val_acc: 0.8803
    Epoch 10/40
    60000/60000 [==============================] - 38s 631us/step - loss: 0.3081 - acc: 0.8868 - val_loss: 0.3206 - val_acc: 0.8844
    Epoch 11/40
    60000/60000 [==============================] - 37s 625us/step - loss: 0.3075 - acc: 0.8876 - val_loss: 0.3154 - val_acc: 0.8863
    Epoch 12/40
    60000/60000 [==============================] - 38s 640us/step - loss: 0.2975 - acc: 0.8912 - val_loss: 0.3126 - val_acc: 0.8859
    Epoch 13/40
    60000/60000 [==============================] - 41s 678us/step - loss: 0.2957 - acc: 0.8925 - val_loss: 0.3132 - val_acc: 0.8867
    Epoch 14/40
    60000/60000 [==============================] - 46s 770us/step - loss: 0.2916 - acc: 0.8921 - val_loss: 0.3080 - val_acc: 0.8880
    Epoch 15/40
    60000/60000 [==============================] - 43s 718us/step - loss: 0.2820 - acc: 0.8956 - val_loss: 0.3076 - val_acc: 0.8891
    Epoch 16/40
    60000/60000 [==============================] - 45s 742us/step - loss: 0.2795 - acc: 0.8978 - val_loss: 0.3091 - val_acc: 0.8879
    Epoch 17/40
    60000/60000 [==============================] - 43s 724us/step - loss: 0.2767 - acc: 0.8977 - val_loss: 0.3032 - val_acc: 0.8910
    Epoch 18/40
    60000/60000 [==============================] - 45s 744us/step - loss: 0.2744 - acc: 0.8993 - val_loss: 0.3093 - val_acc: 0.8895
    Epoch 19/40
    60000/60000 [==============================] - 44s 737us/step - loss: 0.2682 - acc: 0.9007 - val_loss: 0.3033 - val_acc: 0.8900
    Epoch 20/40
    60000/60000 [==============================] - 45s 743us/step - loss: 0.2642 - acc: 0.9028 - val_loss: 0.3047 - val_acc: 0.8924
    Epoch 21/40
    60000/60000 [==============================] - 44s 733us/step - loss: 0.2627 - acc: 0.9037 - val_loss: 0.2996 - val_acc: 0.8927
    Epoch 22/40
    60000/60000 [==============================] - 44s 730us/step - loss: 0.2614 - acc: 0.9032 - val_loss: 0.3048 - val_acc: 0.8925
    Epoch 23/40
    60000/60000 [==============================] - 45s 754us/step - loss: 0.2568 - acc: 0.9064 - val_loss: 0.2995 - val_acc: 0.8929
    Epoch 24/40
    60000/60000 [==============================] - 44s 729us/step - loss: 0.2549 - acc: 0.9075 - val_loss: 0.3016 - val_acc: 0.8902
    Epoch 25/40
    60000/60000 [==============================] - 44s 730us/step - loss: 0.2537 - acc: 0.9064 - val_loss: 0.2975 - val_acc: 0.8947
    Epoch 26/40
    60000/60000 [==============================] - 47s 781us/step - loss: 0.2479 - acc: 0.9078 - val_loss: 0.2999 - val_acc: 0.8926
    Epoch 27/40
    60000/60000 [==============================] - 47s 782us/step - loss: 0.2450 - acc: 0.9093 - val_loss: 0.3037 - val_acc: 0.8930
    Epoch 28/40
    60000/60000 [==============================] - 47s 779us/step - loss: 0.2447 - acc: 0.9099 - val_loss: 0.3002 - val_acc: 0.8944
    Epoch 29/40
    60000/60000 [==============================] - 48s 796us/step - loss: 0.2428 - acc: 0.9103 - val_loss: 0.2975 - val_acc: 0.8943
    Epoch 30/40
    60000/60000 [==============================] - 45s 749us/step - loss: 0.2420 - acc: 0.9101 - val_loss: 0.2959 - val_acc: 0.8958
    Epoch 31/40
    60000/60000 [==============================] - 45s 746us/step - loss: 0.2390 - acc: 0.9122 - val_loss: 0.2983 - val_acc: 0.8944
    Epoch 32/40
    60000/60000 [==============================] - 46s 760us/step - loss: 0.2349 - acc: 0.9128 - val_loss: 0.2986 - val_acc: 0.8925
    Epoch 33/40
    60000/60000 [==============================] - 43s 715us/step - loss: 0.2308 - acc: 0.9159 - val_loss: 0.2953 - val_acc: 0.8970
    Epoch 34/40
    60000/60000 [==============================] - 43s 716us/step - loss: 0.2315 - acc: 0.9146 - val_loss: 0.2985 - val_acc: 0.8962
    Epoch 35/40
    60000/60000 [==============================] - 42s 707us/step - loss: 0.2289 - acc: 0.9150 - val_loss: 0.2929 - val_acc: 0.8988
    Epoch 36/40
    60000/60000 [==============================] - 43s 722us/step - loss: 0.2262 - acc: 0.9169 - val_loss: 0.2987 - val_acc: 0.8949
    Epoch 37/40
    60000/60000 [==============================] - 42s 707us/step - loss: 0.2295 - acc: 0.9157 - val_loss: 0.2963 - val_acc: 0.8967
    Epoch 38/40
    60000/60000 [==============================] - 43s 709us/step - loss: 0.2266 - acc: 0.9167 - val_loss: 0.2944 - val_acc: 0.8983
    Epoch 39/40
    60000/60000 [==============================] - 52s 873us/step - loss: 0.2246 - acc: 0.9176 - val_loss: 0.2964 - val_acc: 0.8973
    Epoch 40/40
    60000/60000 [==============================] - 62s 1ms/step - loss: 0.2209 - acc: 0.9193 - val_loss: 0.2951 - val_acc: 0.8994


### 3. Avaliando modelo MLP
- Utilizo a função evaluate() para obter a acurácia e o erro nos dados de teste


```python
# Mostrando resultados
scores = model.evaluate(X_test, y_test)
```

    10000/10000 [==============================] - 3s 292us/step



```python
# Mostrando o erro e acurácia do modelo MLP
print("Loss do Modelo MLP: {:.5}%".format(scores[0]*100))
print("Accuracy do Modelo MLP: {:.5}%".format(scores[1]*100))
```

    Loss do Modelo MLP: 29.513%
    Accuracy do Modelo MLP: 89.94%


#### 3.1 Predição do modelo MLP
- Utilizo a função predict_classes() para fazer a predição dos dados de teste.
- Categorizo a predição.


```python
preds = model.predict_classes(X_test)
print(preds[1])
preds = to_categorical(preds)
print(preds[1])
```

    2
    [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]


- Avaliando o modelo com a função classification_report(): obtenho as metricas de precisão, recall e f1-score, além da quantidade de imagens para cada classificação


```python
# Mostrando precisão, recall e f1-score do modelo MLP
print(classification_report(y_test, preds, target_names = nomeClasses))
```

                  precision    recall  f1-score   support
    
        Camiseta       0.86      0.84      0.85      1000
           Calça       0.99      0.97      0.98      1000
         Pulôver       0.83      0.82      0.82      1000
         Vestido       0.90      0.91      0.91      1000
          Casaco       0.82      0.84      0.83      1000
        Sandália       0.99      0.97      0.98      1000
          Camisa       0.73      0.75      0.74      1000
           Tênis       0.95      0.96      0.95      1000
           Bolsa       0.98      0.98      0.98      1000
    Tornozeleira       0.95      0.97      0.96      1000
    
       micro avg       0.90      0.90      0.90     10000
       macro avg       0.90      0.90      0.90     10000
    weighted avg       0.90      0.90      0.90     10000
     samples avg       0.90      0.90      0.90     10000
    


- Utilizo a função multilabel_confusion_matrix() para verificar a matriz de confusão de cada label


```python
# Matriz de confusão do Modelo MLP
matriz_confusão = multilabel_confusion_matrix(y_test, preds)
```


```python
# Visualizando matriz de confusão do Modelo MLP
for i, matriz in enumerate(matriz_confusão):
    print("\nMatriz de confusão da classificação do número: {}".format(i))
    print(matriz)
```

    
    Matriz de confusão da classificação do número: 0
    [[8867  133]
     [ 160  840]]
    
    Matriz de confusão da classificação do número: 1
    [[8994    6]
     [  27  973]]
    
    Matriz de confusão da classificação do número: 2
    [[8829  171]
     [ 184  816]]
    
    Matriz de confusão da classificação do número: 3
    [[8902   98]
     [  91  909]]
    
    Matriz de confusão da classificação do número: 4
    [[8812  188]
     [ 161  839]]
    
    Matriz de confusão da classificação do número: 5
    [[8987   13]
     [  33  967]]
    
    Matriz de confusão da classificação do número: 6
    [[8721  279]
     [ 254  746]]
    
    Matriz de confusão da classificação do número: 7
    [[8951   49]
     [  42  958]]
    
    Matriz de confusão da classificação do número: 8
    [[8978   22]
     [  24  976]]
    
    Matriz de confusão da classificação do número: 9
    [[8953   47]
     [  30  970]]


### 4. Gráfico de rendimento do Modelo MLP de acordo com as Épocas



```python
# Plotando gráfico de rendimento do Modelo MLP
N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(15,15))

plt.subplot(221)
plt.title("Loss do Modelo MLP")
plt.plot(np.arange(0, N), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H1.history["val_loss"], label="val_loss")
plt.xlabel("Epoch #")
plt.ylabel("Erro")
plt.legend()

plt.subplot(222)
plt.title("Acurácia e Erro do Modelo MLP")
plt.plot(np.arange(0, N), H1.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H1.history["val_acc"], label="val_acc")
plt.title("Acurácia e Erro do Modelo MLP")
plt.xlabel("Epoch #")
plt.ylabel("Acurácia")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f4be8158908>




![png](imagens/output_41_1.png)

