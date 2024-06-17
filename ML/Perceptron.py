# 2024.06.12~


import numpy as np

class Perceptron:

  def __init__(self,eta=0.01,n_iter=50,random_state=1):
    self.eta = eta
    self.n_iter = n_iter # 학습횟수
    self.random_state = random_state
    self.w_ = 0
    self.b_ = 0
    self.loss_ = []

  # 가중치 초기화
  def init_weight(self, X):
    rgen = np.random.RandomState(self.random_state)
    self.w_  = rgen.normal(loc=0.0,scale=0.01,size=X.shape[1])  # 정규분포를 이용해서 가중치 초기화 
    self.b_ = np.float_(0.)      # 초기편향을 0으로

  # 학습
  def fit(self, X, y):
    self.init_weight(X)
    self.errors_ = []

    for i in range(self.n_iter):
      errors = 0
      for xi, target in zip(X,y):
        update = self.eta*(target - self.predict(xi))
        self.w_ +=  update*xi # 가중치 업데이트
        self.b_ += update     # 편향 업데이트
        errors += int(update != 0.0)
      # print(f"{i} epoch : {errors}")
      self.errors_.append(errors)
    return self

  def net_input(self,X):
    # 학습되어 나온 가중치와 편향으로 구해진 선형방정식
    return np.dot(X,self.w_) + self.b_ 
  
  # 분류 예측
  def predict(self,X):
    return np.where(self.net_input(X) >=0.0, 1, -1)