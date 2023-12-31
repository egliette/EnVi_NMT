# English-to-Vietnamese Neural Machine Translation

![Sample Image](assets/main.png)
[![Python 3.10.7](https://img.shields.io/badge/python-3.10.7-blue)](https://www.python.org/downloads/release/python-3107/)
[![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pypi.org/project/torch/2.0.1/)

**Viblo tutorial:** https://viblo.asia/p/cung-tim-hieu-he-thong-dich-may-mang-no-ron-tu-dau-tu-bleu-score-den-beam-search-decoding-oK9VyxDXLQR

**Full pipeline notebook:**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-3j8lr99-aD2TDWYxaecIJ3VyHehc5Ct?usp=sharing)  

**Google Colab Streamlit link:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1grZfoaD9SCfwOJi-QDOdGRJMlaekDY3f?usp=sharing) 

English-Vietnamese Neural Machine Translation implementation from the scratch with PyTorch.

## 1. Setup 🧰
Create virtual environment then install required packages:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configuration 🛠️
If you would like to utilize your own dataset and hyperparameters, there are two methods to achieve this:
- **1. Modifying the default `config.yml` file**: Open the `config.yml` file and make the necessary adjustments to the variables.
- **2. Specifying a custom configuration file**: By default, the configuration file path is set to `config.yml`. However, if you wish to conduct different experiments simultaneously, you can create your own configuration file in YAML format and pass its path using the `--config` flag. **Please note that our code currently supports reading configuration files in the YAML format exclusively**. For instance:
  ```bash
  python train.py --config my_config.yml
  ```

## 3. Usage 👨‍💻
The complete pipeline encompasses three primary processes:

- **1. ⚙️ Preprocessing**: This involves reading English-Vietnamese parallel corpora, tokenizing the sentences, building vocabularies, and mapping the tokenized sentences to tensors. The resulting tensors are then saved into a DataLoader, along with the trained tokenizers containing the vocabulary for both languages. To execute the preprocessing step, run the following command:
  ```bash 
  python preprocess.py --config config.yml
  ```  

- **2. 🚄 Training and Validation**: In this step, the prepared tokenizers and DataLoaders are loaded. A Seq2Seq model is created, and if available, its checkpoint is loaded. The training process is initiated, and the results are recorded in a CSV file. To train and validate the model, use the following command:
  ```bash 
  python train.py --config config.yml
  ```  

- **3. 🧪 Testing**: This step involves testing the pretrained model on the testing DataLoader. For each pair of sentences, the source, target, predicted sentences, and their respective scores are printed. To perform the testing, execute the following command:
  ```bash 
  python test.py --config config.yml
  ```  

Alternatively, you can run the full pipeline with a single command using the following:
```bash
bash full_pipeline.sh --config config.yml
```

## 4. Inference with Streamlit 🚀

To facilitate an intuitive interaction with your trained model, follow the steps below to host a web server using Streamlit:

- Execute the command provided below to initiate the server:
  ```bash
  streamlit run inference_streamlit.py -- --config config.yml
  ```
  You have the option to replace `config.yml` with your customized configuration file.

- If you prefer to test the server on Google Colab or do not have access to a GPU device, you can conveniently host your server there. Simply access the demo **Google Colab Streamlit link**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1grZfoaD9SCfwOJi-QDOdGRJMlaekDY3f?usp=sharing)  
This link will allow you to utilize my trained model and tokenizers.

Ensure that you have the necessary dependencies and libraries installed before running the Streamlit server. This will enable you to interact seamlessly with your trained model and explore its capabilities through a user-friendly web interface.


## 5. References 📝
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/pbcquoc/transformer 
- https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/ 
- https://github.com/hyunwoongko/transformer 
- https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
- https://en.wikipedia.org/wiki/BLEU
