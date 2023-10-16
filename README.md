# OpenTable Reviews Classification with BERT and Transfer Learning
This repo provides end-to-end pipeline for OpenTable Reviews Classification with BERT and Transfer Learning, as well as data collection with Web Scraping.  

## Data Collection
The detailed explanation of data collection is in [notebook](review_classification.ipynb).  
The data collection script is provided in [web_scraping.py](web_scraping.py), it will output one training df and one validation df.  
1. Find the target restaurant website in OpenTable, go to page 2 of review page.
2. Copy the url of review page 2, e.g. https://www.opentable.ca/r/chez-mal-manchester?page=2&sortBy=newestReview
3. Place the urls in url_list for training dataset and eval_url for validation dataset.
4. Run the main script, it will save the df as csv with two columns: reviews and overall rating of the restaurants.

## Data Structure 
The model takes csv file as input and transform into dataset, while takes review text as input and output ID from 0 to 4, where the id to label dictionary is:  
{0:'1 stars', 1:'2 stars', 2:'3 stars', 3:'4 stars', 4:'5 stars',}  

Sample csv to dataframe format (overall rating is from 0 to 4):

```
                                                  review  overall rating
0      Great ambiance and service. Lots of menu choic...               3
1      Exceptional service, cuisine, ambience.  Windo...               4
2      Our server Darcy was wonderful!  She accommoda...               2
3      Great food choices for lunch and excellent ser...               3
4      Always reliable and great place to go for lunc...               4
...                                                  ...             ...
13438  Our first visit to Chophouse. We will not go b...               3
13439  Friendly and attentive service and the food an...               4
13440  My family and I had an amazing time! Not only ...               4
13441                                         Great food               4
13442  Great food and excellent service. We’ll be back!!               4
```

## Usage
Please change the file/folder path in [train.py](train.py) as below:  

```python
    train_path = '/path to/data/train.csv'
    eval_path = '/path to/data/eval.csv'
    checkpoints_path = '/path to/checkpoints'
```

## Training Hyperparameters
The default training_args is listed below:  
```python
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )
```

## Transfer Learning
This repo is using [LiYuan/amazon-review-sentiment-analysis](https://huggingface.co/LiYuan/amazon-review-sentiment-analysis) as our pretrained model, then we applied our own dataset for fine tuning. The pretrained model is trained on [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) with bert-base-multilingual-uncased model.  
We chose this pretrained model because it provides similar data from customer. They trained on product reviews while we focus on restaurant reviews, they also trained on the label from 5 stars to 1 stars, which matches our need.  

## Accuracy Improvement
The original [train.py](train.py) provides classification for 5 classes {0,1,2,3,4}, which representing 1 star to 5 star.  
The new [train_3class.py](train_3class.py) reduce the class to {0,1,2} that represents {negative, neutral, positive}. Validation accuracy improved from ~0.7 to 0.87 with 2 epoches of training.
