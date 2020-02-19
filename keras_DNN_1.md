# 케라스로 시작하는 DNN

> 모든 딥러닝 구성은 아래의 순서로 진행된다



1. data 구성
2. model 구성
3. model complie
4. model train
5. model evaluate & predict



### 1. data

```python
import numpy as np 

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
```



### 2. model 구성

```python
from keras.models import Sequential
from keras.layers import Dense 

model = Sequential()

model.add(Dense(10, input_dim = 1)) # model.add(Dense(10, input_shape=(1,)))과 같다.
model.add(Dense(16))
model.add(Dense(1))

model.summary()
```



### 3.model compile

``` python
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
```



### 4.model train

```python
model.fit(x, y, epochs=10)
```



### 5. model evaluate

``` python
loss, mse = model.evaluate(x, y)

print('loss: ', loss)
print('mse: ', mse)
```



### 6.model predict

```python
x_predict = np.array([11,12,13])
y_predict = model.predict(x_predict)

print(y_predict)
```





[구글코랩에서 구동시켜보기](https://colab.research.google.com/github/elbicuderri/keras_tutorial/blob/master/keras_DNN_1.ipynb)