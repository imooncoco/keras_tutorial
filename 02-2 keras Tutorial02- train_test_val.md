<h2> 02-2 keras Tutorial02- train, test, validation data</h2>

- Train set

  학습에 사용되는 모델에 맞는 예제 세트

- Test set

  완전히 지정된 분류기의 성능을 평가하기 위해서만 사용되는 예제 세트

  교육 데이터 세트에 대한 최종 모델 적합에 대한 편견없는 평가를 제공하는데 사용되는 데이터 샘플입니다.

- Validation set

  신경망에서 히든 유닛의수를 선택하기 위해 분류기의 매개변수를 조정하는데 사용되는 예제세트

![](C:\Users\imoon\OneDrive\바탕 화면\keras02.PNG)



<h4>1. data 구성</h4>


```python
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test= np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([101, 102, 103, 104, 105])
```



<h4>2. model 구성</h4>

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_shape = (1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
```



<h4>3. model compile</h4>

```python
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
```



<h4>4. model train</h4>

```python
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)
```

- fit의 파라미터인 **validation_data=(x_val, y_val)** 부분을 살펴보면,

  validation set으로 일단 x_val, y_val을 주고 epoch 1번을 돌때마다 검사하여 w를 수정하게 됩니다.

  **fit(훈련)**은 **[x_train, y_train]** 와 **[x_val, y_val]** 사용합니다.

  **evaluate(평가)**는 **[x_test, y_test]** 사용합니다.

   

<h4>5. model evaluate</h4>

```python
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc: ', mse)
```



<h4>6. model predict</h4>

```python
x_prd=np.array([11, 12, 13])
aaa=model.predict(x_prd, batch_size=1)
print(aaa)
```

- 예시로 [11, 12, 13]을 입력하여 예측값을 확인합니다.



[Google Colab에서 구동시켜보기](https://colab.research.google.com/github/elbicuderri/keras_tutorial/blob/master/keras02-2_train_test_val.py)



