# Deep Causal Metric Learning

## Requirements

The code was tested on pytorch==1.2.0 torchvision==0.4.0.


## Data split
We follows [A Metric Learning Reality Check]. The first half of classes are used for cross validation (i.e., training and validation) and the second half of classes are used as the test set. The first half of classes are deterministically split into 4 partitions. The first 0-12.5% of classes make up the first partition, the next 12.5-25% of classes make up the second partition, and so on. The training set comprises 3 of the 4 partitions (the remaining one as the validate set), and cycles through all leave-one-out possibilities

1. Prepare datasets.

   Download datasets: [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [SOP](https://cvgl.stanford.edu/projects/lifted_struct/), [In-Shop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), unzip and organize them as follows.

```
└───datasets
    └───split_train_test.py
    └───CUB_200_2011
    |   └───images.txt
    |   └───images
    |       └───001.Black_footed_Albatross
    |       └───...
    └───CARS196
    |   └───cars_annos.mat
    |   └───car_ims
    |       └───000001.jpg
    |       └───...
    └───SOP
    |   └───Stanford_Online_Products
    |       └───Ebay_train.txt
    |       └───Ebay_test.txt
    |       └───bicycle_final
    |       └───...
    └───Inshop
    |   └───list_eval_partition.txt
    |   └───img
    |       └───MEN
    |       └───WOMEN
    |       └───...
```

2. Run ```split_train_test.py``` to split the data into the cross-validation set (the first half of classes) and the test set (the second half of classes).
3. Run ```cross_valid_split.py``` to split the cross-validation set into training and validation sets (four fold cross-validation).


## Training
Training and test strategy: we follows [A Metric Learning Reality Check]. Specifically, the four floder cross-validation generates four training-validation sets. The model is trained on each training-validation set with embedding dimension 128, so that we finally obtain 4 embedding models, resulting in 4 different accuracies, one for each model’s embedding. We then report the average of these accuracies as ***the accuracy of dim-128***. We concatenate the 128-dim embeddings of the 4 models to get 512-dim embeddings, and then L2 normalize. We then report the accuracy of these embeddings as **the accuracy of dim-512**.   

The backbone: download the imagenet pertained [BN-Inception](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and put it into ```./pretrained_models```.

To train the model, run the command:<br />```sh train.sh```


## Testing

We adopt the metrics in [A Metric Learning Reality Check].

To test the model in dim-128 and dim-512 embedding spaces, run the command:<br />```sh test.sh```

[A Metric Learning Reality Check]: https://arxiv.org/pdf/2003.08505.pdf

## Citation

If you find the code useful, please consider citing this paper:
    
@inproceedings{deng2022deep,<br />
  title={Deep Causal Metric Learning},<br />
  author={Deng, Xiang and Zhang, Zhongfei},<br />
  booktitle = {Proceedings of the 39th international Conference on Machine Learning },<br />
  year={2022}<br />
}
