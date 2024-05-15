# Overview of VSG-Transformer

**Main idea:** Migrate SG-Former from 2D images to 3D video to achieve saliency-guided spatiotemporal action positioning.

<img src="https://cdn.nlark.com/yuque/0/2024/png/35767401/1715609614356-9e647fdf-26ef-47f4-bd80-f50774004b2b.png" alt="img" style="zoom:80%;" />

**Model architecture:** This architecture is fundamentally underpinned by two integral modules: the person Attention Enhancement(PAE) and the VSG-Transformer itself.

<img src="https://cdn.nlark.com/yuque/0/2024/png/35767401/1715517113374-668ce078-5649-489a-b53b-b1a702adef63.png" alt="img" style="zoom:80%;" />

**attention map:**

<img src="https://cdn.nlark.com/yuque/0/2024/png/35767401/1715693968630-e175ddee-17e3-4c5b-9152-23fbd9d0dd72.png" alt="img" style="zoom:80%;" />

**Main data table:**
<img src="https://cdn.nlark.com/yuque/0/2024/png/35767401/1715693947669-6bc3ae34-9e86-4292-a47f-6f942b2afe46.png#averageHue=%23eceae8&clientId=uf4e76fed-bd5b-4&from=paste&height=227&id=u0b47cf7c&originHeight=284&originWidth=1145&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=101369&status=done&style=none&taskId=uf699bd2d-c739-4c7e-83b0-493c5e04fa4&title=&width=916" alt="image.png"  />



# Datasets

Two data sets are mainly used, all related to spatiotemporal action positioning: HIA and UCF101-24.

**HIA:** Mainly intensively annotates the location and behavior of people in indoor surveillance. Multiple people may appear in the video. The HIA data set labels each person's behavior.

**UCF101-24:** The data source is the UCF101 data set (behavior recognition), and each video corresponds to an action. When multiple people appear, UCF101-24 only marks the position and action label of the action subject person.


**Dataset annotation format:** The spatiotemporal data set annotation format used in this article is:  ava and multisports. **Both HIA and UCF101-24 implement the generation of ava and multisports format annotations, which can be used directly in mmaction2.**



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

Before running, you need to use the ln -s command to soft link the hia and ucf data set directories to mmaction2/data.

```python
# Enter the mmaction2 directory

# test
# HIA
python tools/test.py configs/detection/vsgt/VSGT/hia_vgst_small.py checkpoints/vsgt_hia.pth
# UCF101-24
python tools/test.py configs/detection/vsgt/VSGT/ucf_vgst_small.py checkpoints/vsgt_ucf.pth
```
