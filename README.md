# TRACE: Transform Aggregate and Compose visiolinguistic representations for image retrieval with text feedback
- Authors:  Surgan Jandial*, Ayush Chopra*, Pinkesh Badjatiya*, Pranit Chawla, Mausoom Sarkar, Balaji Krishnamurthy.  
_* denotes equal authors_


We use the following for training and testing of our model, reviewers are requested to follow the same if possible. To get started with the framework, install the following dependencies:
- Python 3.6
- [PyTorch 1.1.0]
- CUDA Version: 10.1
- Install the other dependencies using the pip-freezed package list as `pip install --user -r requirements_freeze.txt`


## Download the models
- Download the models inside the folder `models/` from <https://drive.google.com/file/d/1GpiY3YRGELpt8TXLGvolmzct44XNXujD/view?usp=sharing>

## Downloading the Dataset 
Follow the following steps to download the images:
1. Go to https://github.com/hongwang600/fashion-iq-metadata and into the image_url folder
2. Remove the broken links from asin2url.dress.txt, asin2url.shirt.txt and asin2url.toptee.txt
3. Download the the images into three folders, `data/resized_images/dress`, `data/resized_images/toptee` and `data/resized_images/shirt`
4. Resize the images for all three sub datasets and put them inside the folder `data/resized_images`

```
python resize_images.py --image_dir data/dress --output_dir data/resized_images/dress --image_size 224 # for dress
python resize_images.py --image_dir data/shirt --output_dir data/resized_images/shirt --image_size 224 # for shirt
python resize_images.py --image_dir data/toptee --output_dir data/resized_images/toptee --image_size 224 # for toptee
```



## Evaluating on individual models
1. Run the below command to evaluate on particular dataset

```
python evaluate_model.py --data_set dress
python evaluate_model.py --data_set shirt
python evaluate_model.py --data_set toptee
```

## Expected Results

|        |     R@10     |     R@50     |
|:------:|:------------:|:------------:|
|  Dress | 22.20 +- 0.2 | 46.26 +- 0.2 |
|  Shirt | 19.14 +- 0.2 | 39.14 +- 0.6 |
| TopTee | 23.55 +- 0.2 | 48.10 +- 1.0 |


## License

Attribution-NonCommercial-ShareAlike 4.0 International
Copyright: (c) 2020 Adobe Inc.
(Refer to LICENSE.txt for more details)
