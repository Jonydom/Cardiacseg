# CardiacSeg

The official implementation of paper &#34;Customizing Pre-Training Volumetric Transformers with Scaling Pyramid for 3D Cardiac Segmentation&#34;.

## Getting Started

### 1. Clone the Project & Install Requirements
```
git clone git@openi.pcl.ac.cn:OpenMedIA/CardiacSeg.git
pip install -r requirements.txt
```

### 2. Training
```
cd CardiacSeg
sh scripts/train_imagechd.sh
```

### 3. Inference
```
python inference.py --ckpt_url {model_path} --in_file {data_path} --out_file {output_path}
```
Our CardiacSeg model trained on the ImageCHD dataset can be downloaded [here](https://openi.pcl.ac.cn/OpenMedIA/CardiacSeg/modelmanage/model_filelist_tmpl?name=CardiacSeg_model_o5l2).