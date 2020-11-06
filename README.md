# A-simple-ReID-baseline
A simple reid baseline for experiment

**Unfinished work**

# Reference
The codes are expanded on 
- [ReID-baseline](https://github.com/michuanhaohao/deep-person-reid) by Luo & Liao 
- [open-reid](https://github.com/Cysu/open-reid) by Cysu
- [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker) by KevinMusgrave


# Dependencies
- [Pytorch](https://pytorch.org/)(1.6.0)
- [pytorch-metric-learning](https://go.ctolib.com/KevinMusgrave-pytorch_metric_learning.html)

# Construction
The structure of the reid is shown below
```
- reid
	- Datasets
		- datamanager.py
		- dataloader.py
	- eval
		- ranking.py
		- rerank.py
	- log
		- baseline-checkpoint-log
		- ...
	- loss_func
		- softmax.py
		- trihard.py
		- arcface.py
		- cosface.py
		- normface.py
		- ...
	- metric_learning
		- euclidean_dist.py
		- cos_dist.py
		- jaccard_dist.py
	- models
		- activation
		- attention
		- conv
		- pool
		- transform
		- backbone
		- resnet50.py
		- resnet101.py
		- resnext.py
		- mobilenet.py
		- mudeep.py
		- ...
	- optim
		- SGD.py
		- adam.py
		- ...
	- utils
		 - dataset
		 	- datasets.py
		 	- optimizer.py
		 	- sampler.py
		 	- transform.py
		 	- preprocessor.py
		 - logger.py
		 - meter.py
		 - osutils.py
		 - serialization.py
	- Visualization
	- trainer.py
	- tester.py
	- main.py
