# How to use the source code? 
Here I only take USCHAD dataset as example, the evaluation on other clean labeled datasets are the same.
1. Pre-process the dataset and store data per subject, per trial.

2. Run src/add_noise_per_trial.py to add artificial noise to data labels.
'sym' for symmetric noise and 'asym' for asymmetric noise
noise levels are in ratio, from 0 to 1

3. Run src/self_supervision.py for self-supervised pretrain, to obtain a pre-trained feature extractor.
src/self_supervision_feature_extractor is a trained model

4. Modify and run src/run_bmtl.py to train VALERIAN models.
Due to the long training time, try not to train all models at once. Here I trained 28 models for one run and it tooks some time.

After the models are trained, there're two ways to use VALERIAN.
5. Use VALERIAN to clean the data labels from source domain.
Run src/run_clean_source_label_uschad.py

6. Adapt VALERIAN with small amount of clean labeled data from a new unseen subject, then evaluate its performance on this new subject.
Run src/run_eval_bmtl.py
