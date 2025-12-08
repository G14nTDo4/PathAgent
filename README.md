# PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/G14nTDo4/PathAgent)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17052-b31b1b.svg)](https://arxiv.org/abs/2511.17052)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2511.17052.pdf)

<div align="center">

</div>

Official code implementation for paper "PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning"

>Jingyun Chen, Linghan Cai, Zhikang Wang, Yi Huang, Songhan Jiang, Shenjin Huang, Hongpeng Wang, Yongbing Zhang


## Overview

PathAgent is the first training-free interactive agent specifically designed for WSI analysis. By coordinating off-the-shelf pathology models through an agent, it yields traceable decisions and competitive accuracy, suggesting a pragmatic route of computational pathology.

The contributions of PathAgent can be summarized in three aspects:

1. Dynamic analytic Logic: We replace single-step reasoning with Multi-Step Reasoning in the Executor. This mechanism can construct analytic logic and dynamically provide guidelines to retrieve task-relevant information.
2. Adaptive Magnification: PathAgent can adaptively select an appropriate scale based on the analytic state, generating more refined visual evidence.
3. Enhanced Evidence Retrieval: We improve the accuracy of evidence capture by simplifying the query strategy of the Navigator.

![architecture](./assets/Overview.png)
<p align="center"><i>Overview of PathAgent</i></p>

![architecture](./assets/CaseStudy.png)
<p align="center"><i>Illustration of PathAgent's inference procedure</i></p>

## 1. Installation 

```bash
git clone https://github.com/G14nTDo4/PathAgent.git
cd PathAgent
conda create -n pathagent python=3.9 -y
conda activate pathagent

# Install PyTorch (specify CUDA version)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Data Preparation
The following steps take the WSI-VQA dataset as an example.

### 2.1 Patch Generation
Copy data_preparation_script/coordinate_generation.py to the [CLAM](https://github.com/mahmoodlab/CLAM) project directory, and use the following code to generate the patches.
```bash
cd CLAM_PROJECT_DIRECTORY
python coordinate_generation.py \
  --source DATA_DIRECTORY  \
  --save_dir RESULTS_DIRECTORY \
  --preset tcga.csv \
  --step_size 4096 \
  --patch_size 4096 \
  --patch \
  --seg \
```

The DATA_DIRECTORY is the storage directory for svs files.
```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

The above command will segment every slide in DATA_DIRECTORY and generate the following folder structure at the specified RESULTS_DIRECTORY:
```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```

Using the h5 file in RESULTS_DIRECTORY, we can extract the patch from WSI.
```bash
python data_preparation_script/patch_generation.py \
  --h5_dir RESULTS_DIRECTORY/patches \
  --slide_dir DATA_DIRECTORY \
  --output_root RESULTS_DIRECTORY/patches_output \
  --patch_size 4096
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/patches_output:
```bash
RESULTS_DIRECTORY/patches_output/
	├── slide_1
    		├── 0_23904.png
    		├── 4096_28000.png
    		└── ...
	├── slide_2
    		├── 4128_7296.png
    		├── 4128_11392.png
    		└── ...
	└── ...
```

### 2.2 Description Generation
Before generating the description, we first need to modify the Quilt-LLaVA system prompt, replacing quilt-llava/llava/conversation.py with data_preparation_script/conversation_pathology_v0.py. Please note that the final file name is still conversation.py. The different name is only to distinguish the file from the file in the [Quilt-LLaVA](https://github.com/aldraus/quilt-llava) project.

Multiple processes can be executed simultaneously to speed up the description generation process, which is also very user-friendly for users with limited VRAM. Only the description files for each block need to be concatenated at the end. Here, we divide all WSIs in the test set into four equal parts. Readers can adjust the number of parts as needed.
```bash
python data_preparation_script/split_files.py \
  --image_dir RESULTS_DIRECTORY/patches_output \
  --save_dir RESULTS_DIRECTORY/split_name \
  --num_splits 4
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/split_name:
```bash
RESULTS_DIRECTORY/split_name/
	├── slides_part1.txt
	├── slides_part2.txt
	└── ...
```

Copy data_preparation_script/description_generation.py and data_preparation_script/multi_description_generation.sh to the Quilt-LLaVA project directory, and use the following code to generate the description.
```bash
cd Quilt_LLaVA_DIRECTORY
bash multi_description_generation.sh
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/desc:
```bash
RESULTS_DIRECTORY/desc/
	├── patches_descriptions1.json
	├── patches_descriptions2.json
	├── patches_descriptions3.json
	└── patches_descriptions4.json
```

The following command is used to merge all JSON files in the desc folder into a single JSON file.
```bash
python data_preparation_script/merge_json_results.py \
  --input_dir RESULTS_DIRECTORY/desc \
  --save_dir RESULTS_DIRECTORY/desc/patches_descriptions.json \
```

### 2.3 Patch Embeddings Generation
Copy data_preparation_script/img_emb_generation.py and data_preparation_script/multi_emb_generation.sh to the [PLIP](https://github.com/PathologyFoundation/plip) project directory, and use the following code to generate the Embeddings.
```bash
cd PLIP_PROJECT_DIRECTORY
bash multi_emb_generation.sh
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/img_features:
```bash
RESULTS_DIRECTORY/img_features/
	├── slide_1
    		├── 0_23904.npy
    		├── 4096_28000.npy
    		└── ...
	├── slide_2
    		├── 4128_7296.npy
    		├── 4128_11392.npy
    		└── ...
	└── ...
```

## 3. Inference

```bash
python pathagent.py \
    --plip_lib_path PLIP_PROJECT_DIRECTORY \
    --qwen_ckpt QWEN_CHECKPOINT_PATH \
    --plip_ckpt PLIP_CHECKPOINT_PATH \
    --patho_r1_ckpt PATHOR1_7B_CHECKPOINT_PATH \
    --descriptions_file RESULTS_DIRECTORY/desc/patches_descriptions.json \
    --questions_file WSI_VQA_PROJECT_DIRECTORY/dataset/WSI_captions/WsiVQA_test.json \
    --feature_dir RESULTS_DIRECTORY/img_features \
    --patch_root RESULTS_DIRECTORY/patches_output \
    --save_dir RESULTS_DIRECTORY/results/wsi-vqa \
    --dataset_name "wsi_vqa"
```

## 4. Evaluation

```bash
python eval/metics.py \
    --results_dir RESULTS_DIRECTORY/results/wsi-vqa
```


## Citation
```bibtex
@article{chen2025pathagent,
      title={PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning}, 
      author={Jingyun Chen and Linghan Cai and Zhikang Wang and Yi Huang and Songhan Jiang and Shenjin Huang and Hongpeng Wang and Yongbing Zhang},
      journal={arXiv preprint arXiv:2511.17052},
      year={2025}
}
```
