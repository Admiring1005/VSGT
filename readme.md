# Overview of VSG-Transformer

**Main idea:** Migrate SG-Former from 2D images to 3D video to achieve saliency-guided spatiotemporal action positioning.

<img src="https://cdn.nlark.com/yuque/0/2024/png/35767401/1715609614356-9e647fdf-26ef-47f4-bd80-f50774004b2b.png" alt="img" style="zoom:80%;" />

**Model architecture:** This architecture is fundamentally underpinned by two integral modules: the person Attention Enhancement(PAE) and the VSG-Transformer itself.

<img src="https://cdn.nlark.com/yuque/0/2024/png/35767401/1715517113374-668ce078-5649-489a-b53b-b1a702adef63.png" alt="img" style="zoom:80%;" />

**attention map:**

![img](https://cdn.nlark.com/yuque/0/2024/png/35767401/1715693968630-e175ddee-17e3-4c5b-9152-23fbd9d0dd72.png)

**Main data table:**




# Datasets

Three data sets are mainly used, all related to spatiotemporal action positioning: HIA, UCF101-24, and JHMDB.

**HIA:** Mainly intensively annotates the location and behavior of people in indoor surveillance. Multiple people may appear in the video. The HIA data set labels each person's behavior.

**UCF101-24:** The data source is the UCF101 data set (behavior recognition), and each video corresponds to an action. When multiple people appear, UCF101-24 only marks the position and action label of the action subject person.

**JHMDB:** The data source is the HMDB data set (behavior recognition), and each video corresponds to an action. When multiple people appear, only the position and action label of the action subject person are marked.

**Dataset annotation format:** The spatiotemporal data set annotation format used in this article is:  ava and multisports. **HIA, UCF101-24, and JHMDB all implement the generation of ava and multisports format annotations, which can be used directly in mmaction2.**



# Key directory structure

```plain
├── checkpoints #Save pre-trained weights
├── configs #Configuration file
├── data #datasets
├── demo #Demo program provided by mmaction2
├── mmaction #Core implementation code
├── projects
├── requirements
├── requirements.txt
├── tests
├── tools #Testing code, and also provides some calculation codes for indicators such as flops
└── work_dirs #The storage location of model weights and indicator data during training
```

[checkpoints](https://drive.google.com/drive/folders/1i9wMk-wqj7E4Jiv7h8omrSvDmu-51Xbu?usp=drive_link) can be downloaded from this link.

**Core model code file:** mmaction/models/backbones/sgformer_swin_based.py



# Run code

Before running, you need to use the ln -s command to soft link the hia, ucf, and jhmdb data set directories to mmaction2/data.

```python
# Enter the mmaction2 directory

# test
# HIA
python tools/test.py configs/detection/vsgt/VSGT/hia_vgst_small.py checkpoints/vsgt_hia.pth
# UCF101-24
python tools/test.py configs/detection/vsgt/VSGT/ucf_vgst_small.py checkpoints/vsgt_ucf.pth
# JHMDB
python tools/test.py configs/detection/vsgt/VSGT/jhmdb_vgst_small.py checkpoints/vsgt_jhmdb.pth
```
