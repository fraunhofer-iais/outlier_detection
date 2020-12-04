
## From Imbalanced Classification to Supervised Outlier Detection Problems: Adversarially Trained Auto Encoders
*Authors: Max Lübbering, Rajkumar Ramamurthy, Michael Gebauer, Thiago Bell, Rafet Sifa, Christian Bauckhage*

Imbalanced datasets pose severe challenges in training well performing classifiers. This problem is also prevalent in the domain of outlier detection since outliers occur infrequently and are generally treated as minorities. One simple yet powerful approach is to use autoencoders which are trained on majority samples and then to classify samples based on the reconstruction loss. However, this approach fails to classify samples whenever reconstruction errors of minorities overlap with that of majorities. To overcome this limitation, we propose an adversarial loss function that maximizes the loss of minorities while minimizing the loss for majorities. This way, we obtain a well-separated reconstruction error distribution that facilitates classification. We show that this approach is robust in a wide variety of settings, such as imbalanced data classification or outlier- and novelty detection.


### Reproduction
To reproduce the results, execute `train_model.sh` with the respecive model type (ATA, MLP) and respective dataset (ARR, REU, KDD, ATIS), as shown below. 

The script itself can be adapted to train for a different number of epochs (`num_epochs`). Also, the level of parallelization can be set to a different number of processes (`process_count`). Finally, feel free to change the logging directories as you please. By default we log to the root directory of the repository.

Please, take into account that we ran the full gridsearch on 16 Tesla GPUs for multiple days. So instead of running the experiments yourself, we are planning to publish the raw [DashifyML](https://github.com/dashifyML/dashifyML) results in the near future. 

```bash
cd src 
sh train_models.sh <model_type> <dataset>
```

### Citation

```
@inproceedings{lubbering2020imbalanced,
  title={From Imbalanced Classification to Supervised Outlier Detection Problems: Adversarially Trained Auto Encoders},
  author={L{\"u}bbering, Max and Ramamurthy, Rajkumar and Gebauer, Michael and Bell, Thiago and Sifa, Rafet and Bauckhage, Christian},
  booktitle={International Conference on Artificial Neural Networks},
  pages={27--38},
  year={2020},
  organization={Springer}
}
```