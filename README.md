LeNet-5
=======
![Architecture](./images/LeNet-5_Architecture.JPG)
## __Simple implementation of LeNet-5 by Pytorch__


* __train__

```
  python3 LeNet.py --mode train --epoch 20 --lr 0.1 --download True --optim SGD --momentum 0.0
```

mode : 'train' or 'test' mode.

epoch : total epoch of train.

lr : learning rate.

download : If True, train and test data are downloaded by torchvision.

optim : optimizer. 'SGD' or 'ADAM'

momentum : momentum parameter in SGD optimizer.

* __test__

```
  python3 LeNet.py --mode test --download True
```
when I trained this model by 20 epoch, I got __98.56%__ accuracy.
