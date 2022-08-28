# Visual Transformers for 3D Medical Images Classification: Use-Case Neurodegenerative Disorders
This project aims to develop Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)) models using Alzheimer’s Disease Neuroimaging Initiative ([ADNI](https://adni.loni.usc.edu/)) dataset to classify Neurodegenerative Disorders. More specifically, the models can classify three categories (Cognitively Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer’s Disease (AD)) using brain Fluorodeoxyglucose (18F-FDG) Positron Emission Tomography (PET) scans. Also, we take advantage of Automated Anatomical Labeling ([AAL](https://www.sciencedirect.com/science/article/abs/pii/S1053811901909784)) brain atlas and attention maps to develop explainable models.

We propose three ViTs, the best of which obtains an accuracy of 82% on the test dataset with the help of transfer learning. Also, we encode the AAL brain atlas information into the best performing ViT, so the model outputs the predicted label, the most critical region in its prediction, and overlaid attention map on the input scan with the crucial areas highlighted. Furthermore, we develop two CNN models with 2D and 3D convolutional kernels as baselines to classify NDs, which achieve accuracy of 77% and 73%, respectively, on the test dataset. We also conduct a study to find out the importance of brain regions and their combinations in classifying NDs using ViTs and the AAL brain atlas.

Please refer to this [link](http://urn.kb.se/resolve?urn=urn:nbn:se:hh:diva-47250) for the written report.

# Notes
* We used a third-party module for CNN Grad-CAM implementation. The Grad-CAM codes will be uploaded once we ensure there are no copyright issues.
* Please refer to [ADNI](https://adni.loni.usc.edu/)'s website to download the dataset.
* Please contact me by [email](mailto:pooriya.khyr@gmail.com) if you have any questions.

# To Do:
* Improve generalization: try to find a training recipe that is less prone to overfitting.
