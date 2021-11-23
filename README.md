# pytorch_fashionMNIST_NN

## NN Architecture
```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

## Run on Apple M1 chip
The following results of 5 training Epochs:
> Accuracy: 65.6%, Avg loss: 1.065081 
```
Epoch 1
-------------------------------
loss: 2.311282  [    0/60000]
loss: 2.292253  [ 6400/60000]
loss: 2.273526  [12800/60000]
loss: 2.263635  [19200/60000]
loss: 2.243481  [25600/60000]
loss: 2.215104  [32000/60000]
loss: 2.230299  [38400/60000]
loss: 2.193994  [44800/60000]
loss: 2.189474  [51200/60000]
loss: 2.150693  [57600/60000]
Test Error: 
 Accuracy: 39.7%, Avg loss: 2.145558 

Epoch 2
-------------------------------
loss: 2.166793  [    0/60000]
loss: 2.146727  [ 6400/60000]
loss: 2.090312  [12800/60000]
loss: 2.101774  [19200/60000]
loss: 2.049195  [25600/60000]
loss: 1.990566  [32000/60000]
loss: 2.021493  [38400/60000]
loss: 1.944836  [44800/60000]
loss: 1.947774  [51200/60000]
loss: 1.866263  [57600/60000]
Test Error: 
 Accuracy: 59.3%, Avg loss: 1.865350 

Epoch 3
-------------------------------
loss: 1.912574  [    0/60000]
loss: 1.867478  [ 6400/60000]
loss: 1.756603  [12800/60000]
loss: 1.789926  [19200/60000]
loss: 1.679304  [25600/60000]
loss: 1.637864  [32000/60000]
loss: 1.657759  [38400/60000]
loss: 1.569752  [44800/60000]
loss: 1.588839  [51200/60000]
loss: 1.477876  [57600/60000]
Test Error: 
 Accuracy: 62.7%, Avg loss: 1.496239 

Epoch 4
-------------------------------
loss: 1.575702  [    0/60000]
loss: 1.529956  [ 6400/60000]
loss: 1.387199  [12800/60000]
loss: 1.452385  [19200/60000]
loss: 1.331896  [25600/60000]
loss: 1.330868  [32000/60000]
loss: 1.348385  [38400/60000]
loss: 1.281373  [44800/60000]
loss: 1.307333  [51200/60000]
loss: 1.209459  [57600/60000]
Test Error: 
 Accuracy: 64.6%, Avg loss: 1.230349 

Epoch 5
-------------------------------
loss: 1.315388  [    0/60000]
loss: 1.292263  [ 6400/60000]
loss: 1.127821  [12800/60000]
loss: 1.232079  [19200/60000]
loss: 1.104612  [25600/60000]
loss: 1.127488  [32000/60000]
loss: 1.157326  [38400/60000]
loss: 1.100006  [44800/60000]
loss: 1.130908  [51200/60000]
loss: 1.049484  [57600/60000]
Test Error: 
 Accuracy: 65.6%, Avg loss: 1.065081 

Finished!
```

## Predictions of Model:
'''
# Prediction classes:...
Predicted: "Ankle boot", Actual: "Ankle boot"
'''