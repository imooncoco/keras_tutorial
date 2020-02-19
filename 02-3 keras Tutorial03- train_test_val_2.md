<h2> 02-3 keras Tutorial03- train_test_val</h2>

> 데이터를 train, test, validation으로 분할하는 기법에 대해 알아보겠습니다.



<h4>1. data 구성</h4>

```python import numpy as np

x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train = x[:60]  #1~60
y_train = y[:60] 
x_test = x[60:80] #61~80
y_test = y[60:80]
x_val = x[80:]
y_val = y[80:]    #81~100


```

- 먼저 x, y를 1부터 100까지 요소가 들어있는 리스트로 초기화 하였습니다.

- x_train은 첫번째요소부터 59번째 요소까지 자르겠다는 의미가 됩니다.

  즉, x[0]=1 부터 x[59]=60까지만 저장하게 됩니다.

  마찬가지로 x_val은 61~80, x_test는 81~100까지 저장합니다.(y도 마찬가지)

  이렇게 잘린 데이터의 비율은 train : val : test = 6 : 2 : 2



<h4>2. model 구성</h4>

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_shape = (1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))
```



<h4>3. model compile</h4>

```python
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
```



<h4>4. model train</h4>

```python
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))
```



<h4>5. model evaluate</h4>

```python
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mae : ', mse)
```



<h4>6. model predict</h4>

```python
x_prd=np.array([101, 102, 103])
aaa=model.predict(x_prd, batch_size=1)
print(aaa)
```

- 예시로 [101, 102, 103]을 입력하여 예측값을 확인합니다.



[Google Colab에서 구동시켜보기](https://colab.research.google.com/github/elbicuderri/keras_tutorial/blob/master/keras02-2_train_test_val.py)
