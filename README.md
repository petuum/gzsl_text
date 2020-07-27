## Generalized Zero-Shot Text Classification for ICD Coding
Implementation for IJCAI 2020 paper: [Generalized Zero-Shot Text Classification for ICD Coding](https://www.ijcai.org/Proceedings/2020/0556.pdf)

### Dependencies
* Python 3
* torch==1.4.0
* numpy==1.18.2
* nltk==3.4.4
* scikit-learn==0.21.2
* matplotlib==3.1.1
* gensim==3.7.3
* tqdm==4.43.0
* pandas==0.24.2

### Dataset and preprocessing
Please modify ```ICD_DATA_DIR``` in ```constant.py```, which will be the main directory for saving processed data, models etc. To get started, download [```resources.tar.gz```](https://drive.google.com/file/d/1WNLxdmRclD2gHvNRhQ-IwptN_eFwrqU2/view?usp=sharing) and extract it to ```ICD_DATA_DIR/resources```. The ```resources``` directory contains all the necessary vocabulary, embeddings, data splits files which will be used for preprocessing and training.  

The preprocessing script is expecting raw [MIMIC-III database](https://mimic.physionet.org/) saved in ```/path/to/MIMIC-III/```. In particular, there should be two csv files: ```/path/to/MIMIC-III/NOTEEVENTS.csv``` and ```/path/to/MIMIC-III/DIAGNOSES_ICD.csv```

For preprocessing the raw MIMIC-III datasets, run: ```python3 preprocess.py --mimic_dir=/path/to/MIMIC-III/```,
which will extract and tokenizing patient notes accordingly and save to ```ICD_DATA_DIR/processed``` directory.

### Step 0 (optional): Training base feature extractor model
The GAN model is based on a pre-trained feature extractor. To train the feature extractor model, run ```python3 train_base.py```.
The default hyper-parameters is set in ```get_base_config``` function in ```config.py```. In particular, we use LDAM loss function by setting ```--class_margin C=2.0``` and GRNN for encoding ICD hierarcy by setting ```--graph_encoder=gate```.

We also provide our trained base model. Please download [```models.tar.gz```](https://drive.google.com/file/d/12K5_V693QN0ASbhlGhJWpUeG8BNqcKqy/view?usp=sharing) and extract it to ```ICD_DATA_DIR/models```. To evaluate the base model, run ```python3 train_base.py --evaluate```.

### Step 1: Training GAN model
To train the GAN model, run ```python3 train_gan.py```.  The default hyper-parameters is set in ```get_gan_config``` function in ```config.py```. 

* To include keywords prediction task when training GAN, add the arguments ```--top_k=20 --reg_ratio=0.01```, which will include a linear layer on top of generated GAN features for predicting the ```top_k=20``` ICD code related keywords in the patient notes. ```reg_ratio``` is the coefficient for keywords prediction loss. This task can be turned off by setting ```--reg_ratio=0.0```.
* To include zero-shot ICD codes when training GAN, add the arguments ```--add_zero```.

### Step 2: Fine-tuning ICD code classifiers
The final step is to fine-tune the classifiers in pre-trained model using GAN generated features for few-shot and zero-shot codes.
For fine-tuning, run ```python3 finetune.py```. The default hyper-parameters is set in ```get_finetune_config``` function in ```config.py```.
Please set the arguments related to base model the same as in Step 0 and arguments related to GAN model the same as in Step 1.
