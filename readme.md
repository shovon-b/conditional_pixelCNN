# Description

This project was part of UBC's *CPEN 455: Deep Learning* course. The code for unconditional PixelCNN++ was provided. The task was to modify this code for conditional generation, train the model on a custom dataset, then use the trained model as an image classifier.

Here, we investigate two types of conditioning referred as 'mild' and 'strong' that determines how strongly the model is conditioned on training data. It was found that the strong conditioning does not perform well. The 'mild' conditioned model, when used as a classifier, was able to achieve 85% accuracy with moderate training. 

The project report can be found [here](report.pdf).

# Running the model

To train the model use the following prompt.

```bash python pcnn_train.py --dataset cpen455 --save_interval 10 --nr_filters 60 --nr_logistic_mix 5 --batch_size 64 --max_epochs 2 --conditional True --condition_strength strong```

Adjust the parameters as required. A detailed description for some other arguments not shown in the cmd prompt (such as enabling wandb) can be found in `pcnn_train.py` 
