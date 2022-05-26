# micrograd4j

利用java开发一个Autograd引擎。动态构建DAG，并实现反向传播。

A micro Autograd engine developed with java(The idea for this project came from [micrograd](https://github.com/karpathy/micrograd)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG. The DAG only operates over scalar values. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification.

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

#### test/TestEngine.java

``` java
Value a = new Value(-4.0);
Value b = new Value(2.0);
Value c = a.add(b);
Value d = a.mul(b).add(b.pow(3));
c = c.add(c.add(1));
c = c.add(c.add(1).add(a.neg()));
d = d.add(d.mul(2).add((b.add(a).relu())));
d = d.add(d.mul(3).add((b.add(a.neg())).relu()));
Value e = c.add(d.neg());
Value f = e.pow(2);
Value g = f.div(2.0);
g = g.add(f.rdiv(10.0));

g.backward();
System.out.println("a.data -> " + a.data + "; " + "a.grad -> " + a.grad); // a.data -> -4.0; a.grad -> 138.83381924198252
System.out.println("b.data -> " + b.data + "; " + "b.grad -> " + b.grad); // b.data -> 2.0; b.grad -> 645.5772594752186
System.out.println("g.data -> " + g.data + "; " + "g.grad -> " + g.grad); // g.data -> 24.70408163265306; g.grad -> 1.0
```

### Training a neural net

#### Demo.java

`Demo.java` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn.MLP.java` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the code, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset (resources/test_data/test_data.txt):
``` 
MLP of [Layer of [ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2),ReLuNeuron(2)],Layer of [ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16),ReLuNeuron(16)],Layer of [LinearNeuron(16)]]
number of parameters 337
step 0 loss 1.9638621133484333, accuracy 50.0%
step 1 loss 2.3490456150665735, accuracy 50.0%
step 2 loss 2.6464776189077397, accuracy 50.0%
step 3 loss 0.6152116902047455, accuracy 74.0%
step 4 loss 1.139655680191554, accuracy 52.0%
step 5 loss 0.5383808737647844, accuracy 78.0%
step 6 loss 0.6543561574610679, accuracy 75.0%
step 7 loss 0.5203942292803446, accuracy 78.0%
step 8 loss 0.46172360123857714, accuracy 77.0%
step 9 loss 0.5481515887605712, accuracy 77.0%
step 10 loss 0.43029113281491693, accuracy 78.0%
step 11 loss 0.3941153832954229, accuracy 79.0%
step 12 loss 0.3653660105656176, accuracy 81.0%
step 13 loss 0.36678183332025927, accuracy 81.0%
step 14 loss 0.33462358607763043, accuracy 82.0%
step 15 loss 0.33043574832504513, accuracy 83.0%
step 16 loss 0.20555882746098753, accuracy 89.0%
step 17 loss 0.17954274179722446, accuracy 92.0%
step 18 loss 0.1634517690997521, accuracy 90.0%
step 19 loss 0.15808448751828796, accuracy 93.0%
step 20 loss 0.17070764235037178, accuracy 90.0%
step 21 loss 0.15112896847332996, accuracy 93.0%
step 22 loss 0.14816380049355352, accuracy 93.0%
step 23 loss 0.12741924284543907, accuracy 93.0%
step 24 loss 0.1258592888164256, accuracy 96.0%
step 25 loss 0.14448993222619533, accuracy 94.0%
step 26 loss 0.11703575880664031, accuracy 95.0%
step 27 loss 0.11991399250076275, accuracy 95.0%
step 28 loss 0.11250644859559832, accuracy 96.0%
step 29 loss 0.1122712379123405, accuracy 97.0%
step 30 loss 0.10848166745964823, accuracy 97.0%
step 31 loss 0.11053301474073045, accuracy 96.0%
step 32 loss 0.11475675943130205, accuracy 96.0%
step 33 loss 0.1261635901826707, accuracy 93.0%
step 34 loss 0.15131709864479434, accuracy 94.0%
step 35 loss 0.10893801341199083, accuracy 95.0%
step 36 loss 0.09271950174394382, accuracy 97.0%
step 37 loss 0.09110418044688984, accuracy 97.0%
step 38 loss 0.09912837412748972, accuracy 97.0%
step 39 loss 0.11986141502645908, accuracy 96.0%
step 40 loss 0.16106703014875767, accuracy 93.0%
step 41 loss 0.09798468198520184, accuracy 97.0%
step 42 loss 0.08102368944867655, accuracy 98.0%
step 43 loss 0.07303947184840244, accuracy 99.0%
step 44 loss 0.0863052809487441, accuracy 97.0%
step 45 loss 0.07291825732593486, accuracy 100.0%
step 46 loss 0.1057557980145795, accuracy 96.0%
step 47 loss 0.08093449824345554, accuracy 97.0%
step 48 loss 0.06319761143918433, accuracy 100.0%
step 49 loss 0.06386736914872647, accuracy 98.0%
step 50 loss 0.06845829278120481, accuracy 100.0%
step 51 loss 0.09904393774556877, accuracy 96.0%
step 52 loss 0.07282111419678025, accuracy 97.0%
step 53 loss 0.05540132230996909, accuracy 100.0%
step 54 loss 0.06998143976127322, accuracy 97.0%
step 55 loss 0.05986002955127303, accuracy 100.0%
step 56 loss 0.09534546654833871, accuracy 96.0%
step 57 loss 0.06014013456733181, accuracy 98.0%
step 58 loss 0.047855074405145484, accuracy 100.0%
step 59 loss 0.054283928016275594, accuracy 98.0%
step 60 loss 0.04528611993382831, accuracy 100.0%
step 61 loss 0.05462375094558794, accuracy 99.0%
step 62 loss 0.042032793145952985, accuracy 100.0%
step 63 loss 0.04338790757350784, accuracy 100.0%
step 64 loss 0.051753586897849514, accuracy 99.0%
step 65 loss 0.03645154714588962, accuracy 100.0%
step 66 loss 0.035129307532627406, accuracy 100.0%
step 67 loss 0.040085759825092944, accuracy 100.0%
step 68 loss 0.05215369584037617, accuracy 99.0%
step 69 loss 0.03633940406301827, accuracy 100.0%
step 70 loss 0.03888015127347711, accuracy 100.0%
step 71 loss 0.04090424005630395, accuracy 100.0%
step 72 loss 0.031172216887366416, accuracy 100.0%
step 73 loss 0.04072426213271741, accuracy 100.0%
step 74 loss 0.059378521342605975, accuracy 98.0%
step 75 loss 0.041849846606535956, accuracy 100.0%
step 76 loss 0.03390850067201953, accuracy 100.0%
step 77 loss 0.02882639946719858, accuracy 100.0%
step 78 loss 0.040177016098820253, accuracy 100.0%
step 79 loss 0.031580874763228226, accuracy 100.0%
step 80 loss 0.02911959861027716, accuracy 100.0%
step 81 loss 0.03476876690968894, accuracy 100.0%
step 82 loss 0.026663940738996236, accuracy 100.0%
step 83 loss 0.025912574698691876, accuracy 100.0%
step 84 loss 0.02846805443278455, accuracy 100.0%
step 85 loss 0.02539113644948084, accuracy 100.0%
step 86 loss 0.026658747343023592, accuracy 100.0%
step 87 loss 0.024365215229248158, accuracy 100.0%
step 88 loss 0.02408029822395616, accuracy 100.0%
step 89 loss 0.023459113242738115, accuracy 100.0%
step 90 loss 0.02343612411952584, accuracy 100.0%
step 91 loss 0.022919043489410436, accuracy 100.0%
step 92 loss 0.022826550953514414, accuracy 100.0%
step 93 loss 0.02272174326823475, accuracy 100.0%
step 94 loss 0.022656645555664323, accuracy 100.0%
step 95 loss 0.021745650879217204, accuracy 100.0%
step 96 loss 0.022068520750511193, accuracy 100.0%
step 97 loss 0.021523017591105996, accuracy 100.0%
step 98 loss 0.021910340545673795, accuracy 100.0%
step 99 loss 0.02094203506234891, accuracy 100.0%
``` 

