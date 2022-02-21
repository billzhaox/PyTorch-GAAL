# PyTorch-GAAL

This is the PyTorch implementation of this paper:

Jia-Jie Zhu and Jose Bento. Generative adversarial active learning. arXiv preprint arXiv:1702.07956, 2017.

## Directory Structure
```
———— PyTorch-GAAL
 |__ data 
 |__ gan			# train and save DCGAN model
 |__ oracle			# a pretrained CNN act as human oracle
 |__ plot			
 |__ main.py		# Generate commmand to run train.py
 |__ train.py		# main training loop
 |__ utils.py		# toolbox
 |__ requirements.txt       # store dependencies, usage: pip install -r requirements.txt
 |__ README.md
```

## Usage

```bash
python main.py
```

## My Environment (For your reference).
```
python 3.9.7 + pytorch 1.9.0 + torchvision 0.10.0 + cuda 10.1 + cuDNN 7.6.5
```

## To be updated
