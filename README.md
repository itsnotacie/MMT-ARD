# README






## Installation
The code is tested with Python 3.11. To install the required packages, run:
```shell
pip install -r requirements.txt
```


## Loading pretrained models
The provided checkpoints correspond to the vision encoder of CLIP. To load the full CLIP model (including the text encoder), you can use the following code:
```python
import torch
from open_clip import create_model_and_transforms
model, _, image_processor = create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device='cpu'
        )
checkpoint = torch.load('/path/to/fare_eps_2.pt', map_location=torch.device('cpu'))
model.visual.load_state_dict(checkpoint)
```
Alternatively load directly from HuggingFace:
```python
from open_clip import create_model_and_transforms
model, _, image_processor = open_clip.create_model_and_transforms('hf-hub:chs20/fare2-clip')



## Training


### Train teachers
The trained two teachers' weights will be release later.
#### clean teacher
Comment the adv training part in code, then run:
`python -m train.adversarial_training_clip --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /PATH/TO/ImageNet/ --template std --output_normalize True --steps 20000 --warmup 1400 --batch_size 32 --loss ce --opt adamw --lr 1e-5 --wd 1e-4 --attack none --inner_loss ce --norm linf --eps 4 --iterations_adv 10 --stepsize_adv 1 --wandb False --output_dir ./output --experiment_name NORMAL_TEACHER --log_freq 10 --eval_freq 10 --device '0'`

#### adv teacher
Refer to [Pre-trained-Model-Guided-Fine-Tuning-for-Zero-Shot-Adversarial-Robustness](https://github.com/serendipity1122/Pre-trained-Model-Guided-Fine-Tuning-for-Zero-Shot-Adversarial-Robustness/tree/main)


### Train students

```
torchrun --nproc_per_node=1 --master_port=29505 -m train.adversarial_distill_clip_ddp --eps 2 --experiment_name Adv_Distill_student --imagenet_root /PATH/TO/ImageNet/ --device '0' --batch_size 32 --student_arch {RN50|RN101|ViT-B-32-lora}
```



## Evaluation
Make sure files in `bash` directory are executable: `chmod +x bash/*`
### CLIP ImageNet

```shell
python -m CLIP_eval.clip_robustbench --clip_model_name {RN50|RN101|ViT-B-32-lora} --pretrained ../save_ckpts/Adv_Distill_student/ckpts/final.pt --dataset imagenet --imagenet_root /PATH/TO/ImageNet/  --wandb False --norm linf --eps 2
```

### CLIP Zero-Shot
Set models to be evaluated in `CLIP_benchmark/benchmark/models.txt` and datasets in `CLIP_benchmark/benchmark/datasets.txt`
(the datasets are downloaded from HuggingFace). Then run

```shell
cd CLIP_benchmark
./bash/run_benchmark_adv.sh
```














## Acknowledgements
This repository gratefully forks from

- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- [AutoAttack](https://github.com/fra31/auto-attack)
- [RobustVLM](https://github.com/chs20/RobustVLM)
- [MakeMultiHeadNaive](https://github.com/KyanChen/MakeMultiHeadNaive)
- [peft](https://github.com/huggingface/peft)



