# ARPAM-DenseNet-classification
Yixiao Lin (https://opticalultrasoundimaging.wustl.edu/)


We use a deep learning DenseNet to classify the presence of residual rectal cancer after neoadjuvant therapy using co-registered ultrasound and photoacoustic images.
<figure>
  <img src="https://github.com/OpticalUltrasoundImaging/ARPAM-DenseNet-classification/blob/main/model%20diagram.PNG" alt="Model diagram">
  <figcaption>Fig 1. US-PAM DenseNet model diagram</figcaption>
</figure>

<figure>
  <img src="https://github.com/OpticalUltrasoundImaging/ARPAM-DenseNet-classification/blob/main/model%20prediction%20pipeline.PNG" alt="Model prediction">
  <figcaption>Fig 2. Model's prediction and interpretation on a co-registered US-PAM B scan</figcaption>
</figure>

# Paper abstract
Identifying complete response (CR) after rectal cancer preoperative treatment is critical to deciding subsequent management. Imaging techniques, including endorectal ultrasound and MRI, have been investigated but have low negative predictive values. By imaging post-treatment vascular normalization using photoacoustic microscopy, we hypothesize that co-registered ultrasound and photoacoustic imaging will better identify complete responders. In this study, we used in vivo data from 21 patients to develop a robust deep learning model (US-PAM DenseNet) based on co-registered dual-modality ultrasound (US) and photoacoustic microscopy (PAM) images and individualized normal reference images.  We tested the model’s accuracy in differentiating malignant from non-cancer tissue. Compared to models based on US alone (classification accuracy 82.9±1.3%, AUC 0.917(95%CI: 0.897-0.937)), the addition of PAM and normal reference images improved the model performance significantly (accuracy 92.4±0.6%, AUC 0.968(95%CI: 0.960-0.976)) without increasing model complexity. Additionally, while US models could not reliably differentiate images of cancer from those of normalized tissue with complete treatment response, US-PAM DenseNet made accurate predictions from these images. For use in the clinical settings, US-PAM DenseNet was extended to classify entire US-PAM B-scans through sequential ROI classification. Finally, to help focus surgical evaluation in real time, we computed attention heat maps from the model predictions to highlight suspicious cancer regions. We conclude that US-PAM DenseNet could improve the clinical care of rectal cancer patients by identifying complete responders with higher accuracy than current imaging techniques.

Citation: Lin, Y., Kou, S., Nie, H.,... & Zhu, Q. (2023). Deep learning based on co-registered ultrasound and photoacoustic imaging improves the assessment of rectal cancer treatment response. *Biomedical Optics Express*, DOI: 10.1364/BOE.487647

# Usage
model_evaluate.py loads the trained model and the example B scan shown below.
<figure>
  <img src="https://github.com/OpticalUltrasoundImaging/ARPAM-DenseNet-classification/blob/main/example_USPAM_Bscan.PNG" alt="Example scan">
  <figcaption>Fig 3. Example co-registered B scan for illustration</figcaption>
</figure>


The model will then classify the B scan and generate an attention heat map highlighting normal (green) and cancer (red) regions.
<figure>
  <img src="https://github.com/OpticalUltrasoundImaging/ARPAM-DenseNet-classification/blob/main/example_model_prediction.PNG" alt="Example prediction">
  <figcaption>Fig 4. Model predictions on the example B scan</figcaption>
</figure>

# Contact
For any questions, please contact Yixiao Lin at lin.yixiao@wustl.edu.
