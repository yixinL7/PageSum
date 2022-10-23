# PageSum

## Quick Links

- [How to Install](#1-how-to-install)
- [Description of Codes](#2-description-of-codes)
- [Data](#3-data)
- [How to Run](#4-how-to-run)
- [Results, Outputs, Checkpoints](#5-results-outputs-checkpoints)

## 1. How to Install

- `python3.8`
- `conda create --name env --file spec-file.txt`
- Further steps
    - install additional libraries (after activating the conda env) `pip install -r requirements.txt`
    - `compare_mt` -> https://github.com/neulab/compare-mt
        ```console
        git clone https://github.com/neulab/compare-mt.git
        cd ./compare-mt
        pip install -r requirements.txt
        python setup.py install
        ```
Our code is based on Huggingface's [Transformers](https://github.com/huggingface/transformers) library. 

## 2. Description of Codes
- `data_utils.py` -> dataloader
- `main.py` -> training and evaluation procedure
- `modeling_bart_ours.py`, `modeling_utils.py`, `generation_utils.py` -> modefied from Transformers library for PageSum's locality-aware modeling
- `utils.py` -> utility functions
- `cal_rouge.py` -> calculate ROUGE scores

### Workspace
Following directories should be created for our experiments.
- `./cache` -> storing model checkpoints
- `./result` -> storing evaluation results

## 3. Data

We use the following datasets for our experiments.

- arXiv, PubMed -> https://github.com/armancohan/long-summarization
- MultiNews -> https://github.com/Alex-Fabbri/Multi-News
- GovReport -> https://github.com/luyang-huang96/LongDocSum

We provided the preprocessed data below. Each data example is stored as an individual json file.

| arXiv    | PubMed  | MultiNews | GovReport |
|----------|---------|---------|---------|
| [link](https://drive.google.com/file/d/1Z1ZLCg0ocnXtBX-REeE6xcVaD7TQRHIf/view?usp=sharing)     | [link](https://drive.google.com/file/d/1bdG4kCjte9C93zacBehp3YU_OLEPES_t/view?usp=sharing)   | [link](https://drive.google.com/file/d/1TYn5jyyRMLz16HPmzWpZB62fv0GDe020/view?usp=sharing)   | [link](https://drive.google.com/file/d/1JKv5CiquZMIy32xrN5Jeq1CxtdMXnN8z/view?usp=sharing)   |

## 4. How to Run

### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.
### Train
```
python main.py --cuda --gpuid [list of gpuid] -l
```
### Fine-tune
```
python main.py --cuda --gpuid [list of gpuid] -l --model_pt [model path]
```
### Inference
```
python main.py --cuda --gpuid [single gpu] -e --model_pt [model path]
```

### Evaluate

We provide the evaluation script in `cal_rouge.py`. If you are going to use Perl ROUGE package, please change line 13 into the path of your perl ROUGE package.
```python
_ROUGE_PATH = '/YOUR-ABSOLUTE-PATH/ROUGE-RELEASE-1.5.5/'
```

#### Example: evaluating PageSum on arXiv
```console
# write the system-generated files to a file: ./result/arxiv/test.out
python main.py --cuda --gpuid 0 -e --model_pt arxiv/model.bin

# calculate the ROUGE scores using ROUGE Perl Package
python cal_rouge.py --ref ./arxiv/test.target --hyp ./result/arxiv/test.out.tokenized -l

# calculate the ROUGE scores using ROUGE Python Implementation
python cal_rouge.py --ref ./arxiv/test.target.tokenized --hyp ./result/arxiv/test.out.tokenized -l -p
```

## 5. Results, Outputs, Checkpoints

### arXiv
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| LED     | 48.10   | 19.78   | 43.08   |
| PageSum     | 49.72   | 21.06   | 44.69   |

### PubMed
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| LED  | 46.93   | 19.88  | 42.73   |
| PageSum     | 48.24   | 21.06   | 44.26   |

### MultiNews
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| PRIMERA  | 50.29   | 21.20 | 46.23   |
| PageSum     | 51.17   | 21.39   | 46.88   |

### GovReport
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| LED  | 59.42   | 26.53  | 56.63   |
| PageSum     | 59.91   | 27.20  | 57.07   |



Our model outputs on these datasets can be found in `./output`.

We summarize the outputs and model checkpoints below.
You could load these checkpoints using `model.load_state_dict(torch.load(path_to_checkpoint))`.

|          | Checkpoints | Model Output | Reference Output |
|----------|---------|---------|---------|
| arXiv    | [model_spatial.bin](https://drive.google.com/file/d/1OqRl0h5viP_kvo-KvQrPuPjucyq2ihFy/view?usp=sharing) <br> [model_section.bin](https://drive.google.com/file/d/1MnUBK9qcs2ubsz2Qg06qzZT8wUwLMBJg/view?usp=sharing) | [arxiv.test.spatial.out](output/arxiv.test.spatial.out) <br> [arxiv.test.section.out](output/arxiv.test.section.out) | [arxiv.test.reference](output/arxiv.test.reference)  |
| PubMed    | [model.bin](https://drive.google.com/file/d/12T8VzXkdEL1oOG_PUXOor8G0m3slqDe5/view?usp=sharing) | [pubmed.test.ours.out](output/pubmed.test.ours.out) | [pubmed.test.reference](output/pubmed.test.reference)  |
| MultiNews    | [model.bin](https://drive.google.com/file/d/1-SrHwhvpcZOeK2VR9sM5n0SRoLrJ5hhP/view?usp=sharing) | [multinews.test.ours.out](output/multinews.test.ours.out) | [multinews.test.reference](output/multinews.test.reference)  |
| GovReport    | [model.bin](https://drive.google.com/file/d/1k0f7754r1pS2b78mmJnU-UPjN1UgVOfR/view?usp=sharing) | [govreport.test.ours.out](output/govreport.test.ours.out) | [govreport.test.reference](output/govreport.test.reference)  |
