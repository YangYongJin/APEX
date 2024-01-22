# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each prompting variant including APEX.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `Apex/`.


## Apex

#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/APEX/vit_b16_c2_ep15_batch16_2+2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file. You can also experiment on different pretrained models such as [EVA-CLIP](https://arxiv.org/abs/2303.15389) or [CoCa](https://arxiv.org/abs/2205.01917) by using `eva_clip.yaml` or `coca.yaml`.

Below, we provide instructions to train Apex on caltech101 with 20 seeds. After trained on 20 seeds, you can see the average accuracy of 20 seeds.


```bash
# Other possible dataset values includes [food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# trains and evaluates on base classes
bash scripts/apex/base_train_avg.sh caltech101
# evaluates on novel classes
bash scripts/apex/base_test_avg.sh caltech101
```

For the imagenet, you can reproduce the results by following codes:

```bash
# trains and evaluates on base classes
bash scripts/apex/base_train_avg_imagenet.sh imagenet
# evaluates on novel classes
bash scripts/apex/base_test_imagenet.sh imagenet
```


#### (2) Cross-Dataset Transfer and Domain Generalization 
We provide instructions to train Apex on ImageNet using all 1000 classes and then evaluating it directory on new downstream datasets.
We provide cross-dataset and domain generalization config for Apex: `configs/trainers/APEX/cross_eval.yaml`.
* Firstly, train Apex on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/apex/xd_train_apex.sh imagenet 1
```

* Now evaluate ImageNet model on downstream datasets.

```bash
bash scripts/apex/xd_test_all.sh 1
```

