
## Supervised Autoencoder Variants for End to End Anomaly Detection

*Authors: Max Lübbering, Michael Gebauer, Rajkumar Ramamurthy, Rafet Sifa, Christian Bauckhage*

Despite the success of deep learning in various domains such as natural language processing, speech recognition, and computer vision, learning from a limited amount of samples and generalizing to unseen data still pose challenges. Notably, in the tasks of outlier detection and imbalanced dataset classification, the label of interest is either scarce or its distribution is skewed, causing aggravated generalization problems.
In this work, we pursue the direction of multi-task learning, specifically the idea of using supervised autoencoders (SAE), which allows us to combine unsupervised and supervised objectives in an end to end fashion. We extend this approach by introducing an adversarial supervised objective to enrich the representations which are learned for the classification task. We conduct thorough experiments on a broad range of tasks, including outlier detection, novelty detection, and imbalanced classification, and study the efficacy of our method against standard baselines using autoencoders. Our work empirically shows that the SAE methods outperform one class autoencoders, adversarially trained autoencoders and multi layer perceptrons in terms of AUPR score comparison. Additionally, our analysis of the obtained representations suggests that the adversarial reconstruction loss functions enforce the encodings to separate into class-specific clusters, which was not observed for non-adversarial reconstruction loss functions.  


### Reproduction
To reproduce the results, execute `train_model.sh` with the respective dataset (ARR, REU, KDD, ATIS), as shown below. 

The script itself can be adapted to train for a different number of epochs (`num_epochs`). Also, the level of parallelization can be set to a different number of processes (`process_count`). Finally, feel free to change the logging directories as you please. By default we log to the root directory of the repository.

Please, take into account that we ran the full gridsearch on 16 Tesla GPUs for multiple days. So instead of running the experiments yourself, we are planning to publish the raw [DashifyML](https://github.com/dashifyML/dashifyML) results in the near future. 

```bash
cd starters
sh train_models.sh <dataset>
```

### Citation

TBD
