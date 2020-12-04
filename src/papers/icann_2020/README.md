
## From Imbalanced Classification to Supervised Outlier Detection Problems: Adversarially Trained Auto Encoders

Imbalanced datasets pose severe challenges in training well performing classifiers. This problem is also prevalent in the domain of outlier detection since outliers occur infrequently and are generally treated as minorities. One simple yet powerful approach is to use autoencoders which are trained on majority samples and then to classify samples based on the reconstruction loss. However, this approach fails to classify samples whenever reconstruction errors of minorities overlap with that of majorities. To overcome this limitation, we propose an adversarial loss function that maximizes the loss of minorities while minimizing the loss for majorities. This way, we obtain a well-separated reconstruction error distribution that facilitates classification. We show that this approach is robust in a wide variety of settings, such as imbalanced data classification or outlier- and novelty detection.


### Reproduction
To reproduce the results, execute `run.py` as shown below. 
Note that `mtype` is to be replaced with one of MLP and ATA. By how many processes the grid search parallelizes. Furthermore, `nepochs` has to be replaced, by the respective number from the paper. Please, also take into account that we ran the full gridsearch on 16 Tesla GPUs for multiple days. So instead of running the experiments yourself, we are planning to publish the raw [DashifyML](https://github.com/dashifyML/dashifyML) results in the near future. 

```bash
python src/outlier_detection/run.py --model_type <mtype> \
                                    --run_mode TRAIN \
                                    --process_count <pcount> \
                                    --dashify_logging_path ./logs/ \
                                    --gs_config_path ./gs_configs/<mtype>/ARR.yml \
                                    --num_epochs <nepochs>
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