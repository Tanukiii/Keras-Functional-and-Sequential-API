#!/usr/bin/env python
# coding: utf-8

# # Keras Functional  and Sequential APIの使い方

# ## Preparcing Tensorflow2.x and Dataset MNIST

# In[1]:


import tensorflow as tf
import keras

# Mnist
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization 0-255の値が入っているので、0-1に収まるよう正規化します
x_train, x_test = x_train / 255.0, x_test / 255.0

# Check the data
# Now, each row has one data
print(x_train.shape, x_test.shape)


# # Sequential の場合の書き方

# ### ・keras.models.Sequential()にlistで与える
# 
# ### ・model.add()で1層ずつ足してくかしてモデルをつくる
# 
# ### 最期に学習条件を決めてcompileすれば完成です。
# 

# In[2]:


# Sequentialモデルを定義します

model = keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape =(28, 28)))
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation ='softmax'))

# モデルをcompileします
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
display(model.summary())

# 学習します
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=5)

# テストデータの予測精度を計算します
print(model.evaluate(x_test, y_test))


# # Functional の場合の書き方

# ## ①入力が１つの場合

# ### 上記のSequentialの場合とまったく同じモデルをfunctional APIで書くと次のようになります。

# Sequentialだと入力数と出力数がどちらも１つと決まってるのでSequentialでネットワーク構造を定義したら完成でしたが、functional APIだと入力と出力をどちらも複数設定できますので、ネットワーク構造をkeras.layersで定義する部分の２つを書いておいて、入力と出力がいくつあるのかkeras.Model()で定義して完成となります。

# In[3]:


# モデル構造を定義します
inputs = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# 入出力を定義します
model = keras.Model(inputs=inputs, outputs=predictions)

# モデルをcompileします
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
display(model.summary())

# 学習します
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=5)

# テストデータの予測精度を計算します
print(model.evaluate(x_test, y_test))


# (Note)
# 
# 学習データとテストデータのようにkerasの外からkerasモデルに渡すデータは必ず最初にkeras.layers.Input()で受け取り、そこから加える層の右にその層への入力を（）付きで与えるように書いて、1層ずつ増やしていくという書き方になります。
# 
# 下の例だとpredictionsに入力から出力までのInput => Flatten => Dense(128, relu) => Dropout => Dense(10, softmax)までのネットワークが全部入ってますので、Sequentialで書いたmodelと同じ内容になります

# ## ② 入力が2つある場合(出力は１つ)

# 入力が複数ある場合はinputが複数あるネットワークを書いて、keras.Model()にlistでinputを与えるようにします。下の例はmnistデータを2つに分けてkeras model内で結合してから同じネットワークに通すようにしたものです。

# In[4]:


# 複数入力のテストの為にxを分割してみます
# 全ての行、３９２
x_train2_1 = x_train.reshape(60000, 784)[:,:392]  # (60000, 392) 
x_train2_2 = x_train.reshape(60000, 784)[:,392:]  # (60000, 392)

x_test2_1 = x_test.reshape(10000, 784)[:,:392] # (10000, 392)
x_test2_2 = x_test.reshape(10000, 784)[:,392:] # (10000, 392)

# Functional APIでモデルを定義します
input1 = tf.keras.layers.Input(shape=(392,))
input2 = tf.keras.layers.Input(shape=(392,))
inputs = tf.keras.layers.concatenate([input1, input2])

x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# 入出力を定義します
model = tf.keras.Model(inputs=[input1, input2], outputs=predictions)

# モデルをcompileします
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
display(model.summary())

# 学習します
hist = model.fit([x_train2_1, x_train2_2], y_train, validation_split=0.1, epochs=5)

# テストデータの予測精度を計算します
print(model.evaluate([x_test2_1, x_test2_2], y_test))


# ## ③ 入力と出力が2つある場合（損失関数は１つ）

# 分岐を加えて出力が2つあるmodelに変えてみました。x1とx2の2つの経路に分岐していて、prediction1とprediction2がそれぞれの出力までのネットワーク情報をもっています。出力段が2つになったのでkeras.Model()に与える出力段も２つになります。

# In[5]:


# 複数入力のテストの為にxを分割してみます
x_train2_1 = x_train.reshape(60000, 784)[:,:392]
x_train2_2 = x_train.reshape(60000, 784)[:,392:]
x_test2_1 = x_test.reshape(10000, 784)[:,:392]
x_test2_2 = x_test.reshape(10000, 784)[:,392:]

# Functional APIでモデルを定義します
input1 = keras.layers.Input(shape=(392,))
input2 = keras.layers.Input(shape=(392,))

# Prediction 1
inputs1 = keras.layers.concatenate([input1, input2])
x1 = keras.layers.Dense(128, activation='relu')(inputs1)
x1 = keras.layers.Dropout(0.2)(x1)
prediction1 = keras.layers.Dense(10, activation='softmax')(x1)

# Prediction 2
inputs2 = keras.layers.concatenate([input1, input2])
x2 = keras.layers.Dense(128, activation='relu')(inputs2)
x2 = keras.layers.Dropout(0.2)(x2)
prediction2 = keras.layers.Dense(10, activation='softmax')(x2)

# 入出力を定義します
model = keras.Model(inputs=[input1, input2], outputs=[prediction1, prediction2])

# モデルをcompileします
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
display(model.summary())

# 学習します
hist = model.fit([x_train2_1, x_train2_2], [y_train, y_train], 
                 validation_split=0.1, epochs=5)

# テストデータの予測精度を計算します
print(model.evaluate([x_test2_1, x_test2_2], [y_test, y_test]))


# # ④ 入力、出力、損失関数が2つある場合

# せっかく出力を分けたので損失関数も別々に入れてみます。
# 
# modelを作るときにname=''で名付けておいて、compile()するときにlossを辞書型で渡せば出力ごとに異なる損失関数を使うことができます。下の例だと同じ損失関数を使ってますが、ぜんぜん違う損失関数を指定しても構いません。
# 
# 学習はトータルの損失関数を最小化するように進めますがデフォルトでは単純に合計するようです。加算比率をloss_weightsに辞書型で渡すことで指定することもできるので、以下では0.5ずつで加算するようにしています。

# In[6]:


# 複数入力のテストの為にxを分割してみます
x_train2_1 = x_train.reshape(60000, 784)[:,:392]
x_train2_2 = x_train.reshape(60000, 784)[:,392:]
x_test2_1 = x_test.reshape(10000, 784)[:,:392]
x_test2_2 = x_test.reshape(10000, 784)[:,392:]

# Functional APIでモデルを定義します
input1 = keras.layers.Input(shape=(392,))
input2 = keras.layers.Input(shape=(392,))

# Prediction 1
inputs1 = keras.layers.concatenate([input1, input2])
x1 = keras.layers.Dense(128, activation='relu')(inputs1)
x1 = keras.layers.Dropout(0.2)(x1)
prediction1 = keras.layers.Dense(10, activation='softmax', name='prediction1')(x1)

# Prediction 2
inputs2 = keras.layers.concatenate([input1, input2])
x2 = keras.layers.Dense(128, activation='relu')(inputs2)
x2 = keras.layers.Dropout(0.2)(x2)
prediction2 = keras.layers.Dense(10, activation='softmax', name='prediction2')(x2)

# 入出力を定義します
model = keras.Model(inputs=[input1, input2], outputs=[prediction1, prediction2])


# モデルをcompileします
model.compile(optimizer='adam',
              loss={'prediction1': 'sparse_categorical_crossentropy', 
                    'prediction2': 'sparse_categorical_crossentropy'},
              loss_weights={'prediction1': 0.5,
                            'prediction2': 0.5},
              metrics=['accuracy'])
display(model.summary())

# 学習します
hist = model.fit([x_train2_1, x_train2_2], [y_train, y_train], 
                 validation_split=0.1, epochs=5)

# テストデータの予測精度を計算します
print(model.evaluate([x_test2_1, x_test2_2], [y_test, y_test]))


# ## ⑤ 学習済みmodelを組み込む

# 学習済みmodelを部品として組み込むことも出来ます。
# 
# 使い方はkeras.layersの代わりに学習済みmodelを置くだけですし、組み込んだら1つのkeras modelとして使えますのでアンサンブルモデルも簡潔に書けて便利です。
# 下の例では上半分で学習しで作ったmodelを下半分で部品として組み込んだmodel2を作っています。
# 

# In[7]:


# Functional APIでモデルを定義します
inputs = keras.layers.Input(shape=(28, 28))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# 入出力を定義します
model = keras.Model(inputs=inputs, outputs=predictions)

# モデルをcompileします
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
display(model.summary())

# 学習します
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=5)

# テストデータの予測精度を計算します
print(model.evaluate(x_test, y_test))


# ######################################################

# In[8]:


# モデルを再利用するモデルを定義します
inputs = keras.layers.Input(shape=(28, 28))
predictions = model(inputs)

# モデルをcompileします
model2 = keras.Model(inputs=inputs, outputs=predictions)
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
display(model2.summary())

# テストデータの予測精度を計算します
print(model2.evaluate(x_test, y_test))


# ## Appendix: Slice

# In[9]:


import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90,])
data2=np.array(data).reshape(3,3) #3×3の配列を作成
data2


# In[10]:


data2[:2,] #0～1行目、すべての列


# In[11]:


data2[:,:1] #　全ての行、1列目以降


# In[12]:


data2[:,1:] #すべての行、1列目以降


# In[13]:


data2[1:,1:] #1行目以降、1列目以降

