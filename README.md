# Mitigating Spurious Correlations in Image Datasets via Generative Counterfactual Image Augmentation

<img width="1804" height="872" alt="ChatGPT Image Aug 24, 2026, 05_24_35 PM" src="https://github.com/user-attachments/assets/82cff478-cffa-4b74-84d9-37def211c5b8" />

## Abstract
Machine learning models often suffer from performance degradation and reduced generalizability when faced with out-of-distribution data and spurious correlations, as they tend to rely on biased shortcut features rather than learning the core characteristics. Although various solutions in the form of data manipulation, representation learning, and learning strategy have been proposed to address this challenge, a major limitation of many successful methods is their heavy reliance on group annotations for data balancing, a requirement that is costly and sometimes infeasible in real-world applications. To overcome this limitation, the present study proposes a novel method, termed Generative Counterfactual Image Augmentation (GCIA), which improves upon the DFR framework. By leveraging the capabilities of pre-trained image editing models, the proposed method generates counterfactual image data to implicitly neutralize spurious features at the dataset level. The key advantage of this approach is a significant reduction in the need for group annotations; reliance on these labels is eliminated during the balancing and training phases, and they are required solely for hyperparameter tuning. The performance of the proposed method was evaluated on standard datasets, including Waterbirds, CelebA, and UrbanCars. Experimental results demonstrate that, despite a severe reduction in access to group annotations, this method achieves highly competitive performance. It exhibits substantial superiority in improving Worst-Group Accuracy compared to baseline models like ERM (for instance, increasing this accuracy from 21.4% to 79.5% in the UrbanCars dataset and from 46.6% to 85.6% in CelebA). Ultimately, this research demonstrates that employing generative models for semantic data balancing is an efficient and cost-effective strategy to mitigate model bias, effectively enhancing the robustness of computer vision systems against spurious correlations.

## Training

1. **ERM Training**:
```bash
dataset="urbancars"
seed=1
uv run main.py \
    --output_path logs \
    --experiment ERM \
    --dataset "$dataset" \
    --dataset_path "datasets/${dataset}" \
    --optimizer SGD \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --epochs 300 \
    --pretrained_path imagenet \
    --batch_size 128 \
    --seed "$seed"
```

2. **Edit images**:
```bash
dataset="urbancars"
editing_model="flux2-klein-4b"
uv run edit_images.py \
    --dataset "${dataset}" \
    --dataset_path datasets/${dataset} \
    --output_dir "datasets/${dataset}_edited_${editing_model}" \
    --batch_size 16 \
    --edit_model "${editing_model}"
```

3. **GCIA**:
```bash
DATASET="celeba"
EDITING_MODEL="flux2_klein_4b"
SAMPLE_SIZE=128
BATCH_SIZE=128
LR=0.005
OPTIMIZER="adamW"
L1=1e-2
SEED=1
uv run main.py \
    --output_path logs \
    --dataset "$DATASET" \
    --dataset_path "datasets/${DATASET}" \
    --experiment GCIA \
    --sample_size "$SAMPLE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --pretrained_path "checkpoints/${DATASET}/erm_seed${SEED}/final_checkpoint.pt" \
    --l1 "$L1" \
    --epochs 100 \
    --optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --balanced_dataset_path "datasets/${DATASET}_edited_${EDITING_MODEL}"
```


**Note:** If the `--feature_only` flag is used, you should provide the pre-computed features of the specified dataset, which can be saved using the `save_features.py` file in the repository. If the flag is not specified, the raw image of the dataset should be provided. Here is an example script:

- For main datasets:
```bash
dataset="urbancars"
seed=1
uv run save_features.py \
    --dataset "$dataset" \
    --dataset_path "datasets/${dataset}" \
    --save_path "datasets_features/${dataset}/noaug_features_seed${seed}" \
    --pretrained_path "checkpoints/${dataset}/erm_seed${seed}/final_checkpoint.pt" \
    --batch_size 512 
```
- For Counterfactual Image Augmented Datasets:
```bash
dataset="urbancars"
sample_size=128
editing_model="flux2_klein_4b"
seed=1
uv run save_features.py \
    --dataset "$dataset" \
    --dataset_path "datasets/${dataset}_edited_${editing_model}" \
    --save_path "datasets_features/${dataset}_edited_${editing_model}_samples${sample_size}/noaug_features_seed${seed}" \
    --pretrained_path "checkpoints/${dataset}/erm_seed${seed}/final_checkpoint.pt" \
    --batch_size 512 \
    --gcia True \
    --sample_size "$sample_size"
```

- GCIA with `--feature_only` used:
```bash
DATASET="celeba"
EDITING_MODEL="flux2_klein_4b"
SAMPLE_SIZE=128
BATCH_SIZE=128
LR=0.005
OPTIMIZER="adamW"
L1=1e-2
SEED=1
uv run main.py \
    --output_path logs \
    --dataset "$DATASET" \
    --dataset_path "datasets_features/${DATASET}/noaug_features_seed${SEED}" \
    --experiment GCIA \
    --sample_size "$SAMPLE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --pretrained_path "checkpoints/${DATASET}/erm_seed${SEED}/final_checkpoint.pt" \
    --l1 "$L1" \
    --epochs 100 \
    --optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --balanced_dataset_path "datasets_features/${DATASET}_edited_${EDITING_MODEL}_samples${SAMPLE_SIZE}/noaug_features_seed${SEED}" \
    --feature_only True
```

## Results

| **Method** | **Group Info**<br>**Train/Val** | **WaterBirds**<br>**Worst (%)** | **WaterBirds**<br>**Mean (%)** | **CelebA**<br>**Worst (%)** | **CelebA**<br>**Mean (%)** | **UrbanCars**<br>**Worst (%)** | **UrbanCars**<br>**Mean (%)** |
|:---:|:---:|---:|---:|---:|---:|---:|---:|
| **ERM** | **✗/✗✗** | 66.2±1.8 | 90.2±0.6 | 46.6±3.0 | 95.5±0.0 | 21.4±2.2 | 74.1±0.5 |
| **SUBG** | **✓/✓✓** | 89.1±1.1 | - | 85.6±2.3 | - | - | - |
| **GDRO** | **✓/✓✓** | 91.4 | 93.5 | 88.9 | 92.9 | 73.1 | 84.2 |
| **JTT** | **✗/✗✓** | 86.7 | 93.3 | 81.1 | 88 | 79.5 | 86.3 |
| **GDRO + EIIL** | **✗/✗✓** | 77.2±1.0 | 96.5±0.2 | 81.7±0.8 | 85.7±0.1 | 76.5±2.6 | 85.4±2.1 |
| **SELF** | **✗/✗✓** | 91.6±1.4 | 93.6±1.1 | 83.9±0.9 | 91.7±0.4 | 83.2±0.8 | 90.0±0.5 |
| **AFR** | **✗/✗✓** | 90.4±1.1 | 94.2±1.2 | 82.0±0.5 | 91.3±0.3 | 80.2±2.0 | 87.1±1.2 |
| **EVaLS-GL** | **✗/✗✓** | 89.4±0.3 | 95.1±0.3 | 84.6±1.6 | 91.1±0.6 | 83.5±1.7 | 88.3±0.9 |
| **GCIA (Ours)** | **✗/✗✓** | 90.3±0.4 | 95.1±0.3 | 85.6±2.1 | 90.8±0.7 | 79.5±1.9 | 87.5±0.4 |
| **DFR** | **✗/✓✓** | 92.9±0.2 | 94.2±0.4 | 88.3±1.1 | 91.3±0.3 | 79.6±2.2 | 87.5±0.6 |
| **SSA** | **✗/✓✓** | 89.0±0.6 | 92.2±0.9 | 89.8±1.3 | 92.8±0.1 | - | - |

### XGrad-CAM
#### Waterbirds
<img width="755" height="1280" alt="photo_2026-09-13_16-23-52" src="https://github.com/user-attachments/assets/2bb42bd3-2e60-40af-8922-b6dc5c0f539b" />

#### CelebA
<img width="782" height="1280" alt="photo_2026-09-13_16-23-52" src="https://github.com/user-attachments/assets/d6cf08e1-315b-47ad-9c0f-8cbd5368c002" />

#### UrbanCars
<img width="753" height="1280" alt="photo_2026-09-13_16-23-52" src="https://github.com/user-attachments/assets/a913821e-73fd-4ca1-b4a8-a195a901dbbc" />


## Datasets
- [CelebA](https://drive.google.com/file/d/1kMs0KmmdqxXvEXHRA6YFTlHrKGTVZdV9/view?usp=drive_link)
- [Waterbirds](https://drive.google.com/file/d/1UxSEZ1W0A4530ekGT8SsUCveqn2AMlL3/view?usp=sharing)
- [UrbanCars](https://drive.google.com/file/d/19QYUPRetPrgGzdrIoUKK6cd2zrvSyx6t/view?usp=drive_link)

## ERM Checkpoints

The ERM checkpoints for CelebA, Waterbirds, and UrbanCars are available [here](https://drive.google.com/file/d/1jGc9J4C_Ccy4P1WFfBG1-f4uuOfLzzk7/view?usp=drive_link).

## Counterfactual Image Augmented Datasets
- CelebA
  - [FLUX.2 klein 4B](https://drive.google.com/file/d/1jufYL1D5IQMATLN8uDb47bMZt6U8SauH/view?usp=drive_link)
  - [FLUX.2 klein 9B](https://drive.google.com/file/d/1rIto-SDMzIuDLTJHjLOPNumXue2_Nbz_/view?usp=drive_link)
- Waterbirds
  - [FLUX.2 klein 4B](https://drive.google.com/file/d/1BRV9gOMTdbSFS1KkCa-5NyyErUOCK0pq/view?usp=drive_link)
  - [FLUX.2 klein 9B](https://drive.google.com/file/d/1DEL477vcfJaiW9nc6QRiAbQ80i4SCweA/view?usp=drive_link)
- UrbanCars
  - [FLUX.2 klein 4B](https://drive.google.com/file/d/1KZy0s9v1fXSWXPsqH9SvAUC3h1rUbnrE/view?usp=drive_link)
  - [FLUX.2 klein 9B](https://drive.google.com/file/d/1nMIqXsl8CX0Y4BYmpJGDJM814qe2AARE/view?usp=drive_link)


## Acknowledgments

We would like to thank the authors of the following papers and repositories for their valuable contributions:

- [EVaLS](https://github.com/sharif-ml-lab/EVaLS)
- [Deep Feature Reweighting](https://github.com/PolinaKirichenko/deep_feature_reweighting)
- [Spurious Feature Learning](https://github.com/izmailovpavel/spurious_feature_learning)
- [WILDS](https://github.com/p-lambda/wilds)
- [DRO](https://github.com/kohpangwei/group_DRO/)
