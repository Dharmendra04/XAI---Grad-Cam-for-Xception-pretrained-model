# XAI---Grad-Cam-for-Xception-pretrained-model
Grad-CAM is a technique that generates heatmaps to highlight the regions of an input image that are important for a model's prediction, providing insight into the model's decision-making process.
# Explainable Artificial Intelligence (XAI) in Transfer learning-based Brain Tumor Image Analysis 

Group Members:
* 10808972 : H.M.L.N.K Herath
* 10790776 : Dharmendra S


## 1.What this Repository about
This Repository is about classifying brain tumor images by applying transfer learning and analysis using different kinds of XAI methods. Pre-trained models were fine-tuned using the ImageNet dataset and choose the best model by comparing the accuracy curve and training loss. Then applied the Explainable AI methods for the best and low accuracy models to compare the prediction performance. For this three different kinds of Explainable AI methods such as Grad CAM (Gradient-weighted Class Activation Mapping), LIME (Local Interpretable Model Agnostic Explanations), and SHAP ((SHapley Additive exPlanations) were chosen. According to our knowledge, this is the first project which applied these three commonly used XAI methods for a dataset and compared those XAI methods.

You can find the links for the pretrained models here,

* Best_model : https://drive.google.com/file/d/1byX3XP8bAe95NtMFlYX8cuRqmT2n3Sm5/view?usp=share_link
* Low_accuracy_model : https://drive.google.com/file/d/1n9BCLThjO5QWr3HHjkUrs8p0GA9QOV3g/view?usp=share_link

You can find the implementation of the Different XAI methods for two models, one is the best model(Xception) wiht 97% accuracy and other with a low accuracy model of around 80%. Please find the code links below

* 97% accuracy model : https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Best_Model_Xception_XAI_Analysis.ipynb
* 80% accuracy model: https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Low_Accuracy_Model_xception_XAI_Analysis.ipynb

You can find the video link for more explantation here,
* Video_link : https://drive.google.com/file/d/1-pR1v0cQhg2NHuHnWgvTp4FHfDxONXQy/view?usp=share_link

You can find the dataset we use for training the Xception Model here,
* MRI_Dataset : https://drive.google.com/file/d/1uYx2MgwD75rWNO3ul8-YjDzpiYZSkKoK/view?usp=share_link


## 2.Introduction
Our team consists of two members Herath Herath and Dharmendra Selvaratnam. As Herath has prior experience in working with pre-trained models on Autism spectrum disorder analysis,, we decided to do a project called ‘Explainable artificial intelligence (XAI) in Transfer learning-based Brain Tumor image analysis’. This is the Third Topic of the suggested topics for this module. Brain Tumor detection is a compulsory and sensitive practice in the medical field and Commonly used Deep learning models have a lot of black boxes which makes the model less interpretable, unlike other machine learning models which are comparatively transparent. Thus, Applying XAI techniques to explain these black boxes are precise for the medical communities and there is a lot of research piled up in this area. (Jin et al., 2023, Abeyagunasekera et al., 2022, and Van der Velden et al., 2022). 

#### Aim 

Identify brain tumors using artificial intelligence

#### Objectives  

* Find a pre-train model with high accuracy to classify brain tumors of the selected datasets  

* Explore various XAI methods to explain the prediction.  

* Apply various XAI methods to explain the model predictions. 


### 2.1 Transfer Learning with Pre train models 

In Deep Learning (DL), the Convolutional Neural Network(CNN) is a principal method applied to study visual images. CNN has the ability to extract dominant features from a set of images and learn to differentiate classes in various application domains. As an example, CNN is widely used in the healthcare sector to diagnose and classify health problems. This study tries to classify three types of brain tumors namely glioma tumor, meningioma tumor, and pituitary tumor. Due to the small number of data available we are using transfer learning to train our classification model. Transfer learning (TL) is an approach that acquires and stores knowledge from resolving a problem of one task and applies it to a different but related task. This can be achieved through pre-trained models which are already designed and trained using a large dataset ImageNet. It consists of a large number of categories representing many different areas. Kerase library( ) provide different type of pre-train models with various DCNN architectures.  All these pre-rain models were trained using a dataset called ImageNet which comprises 1.28 million training images, 100k testing images, and 50k validation images to classify 1000 classes. This natural image dataset is large enough to create a more generalized model. For our study, we have selected four different models based on their performance, the depth of the CNN, the number of parameters that we need to train and the applicability of the domain. ResNet(Huang et.al, 2017), and Inception (He et. al,2016, Szegedy et. al,2015) are the most popular pre-trained DL models, that were extensively applied in medical image classification. Xception, and DenseNet 201 are the other two models that were selected for this study. 

### 2.2 Explainability  

Machine Learning (ML) models and Neural Network architectures can be easily interpretable and understood when they’re simple. Nevertheless, when it is a complex model like a DCNN, it would be a difficult task to interpret since it is composed of millions of parameters and a huge number of layers with various designs. The lack of transparency is a significant obstacle to the implementation of AI in various sectors, particularly in domains like healthcare. In the absence of a comprehensive comprehension of how a model can forecast ailments like pneumonia or cancer in a patient, medical organizations will have no choice but to remain skeptical of ML and DL. Explainable AI is a way of interpreting the predictions of a DL model in a logical manner. We have selected three different XAI methods that were mostly used in medical image classification, Grad CAM, LIME and SHAP.   

#### Gradient-weighted Class Activation Mapping (Grad CAM) 

The Gradient-weighted Class Activation Mapping (Grad CAM) approach utilizes the gradients of a target concept, such as the predicted category of an image or a textual description, which flow into the last convolutional layer of a neural network model. This method generates a rough map of key areas in the image that are significant in predicting the target concept (Selvaraju et al., 2016).   

Additionally, the pixels that are considered by each of the convolution layers also can be visualized, hence a visualization of how each layer extracts unique features of an image can be clearly visualized. This kind of visualization from each layer cannot be visualized in other XAI methods, like SHAP, and LIME.  

Gradient-based methods XAI methods like Grad CAM are transparent and do not require any additional model training or modification (Cian et al., 2020). They work by analyzing the gradients of a neural network during its prediction, making it easy to interpret and understand the model's decision-making process. Thus, this model can be used to produce real-time visualization.   

As a technical overview, In the Grad-CAM technique, the gradients of the classification score are used in conjunction with the final convolutional feature map to identify the regions of an input image that have the most significant influence on the classification score. Specifically, the technique examines the magnitude of the gradient with respect to the final convolutional feature map to determine where the classification score depends most on the image data. Regions, where the gradient is larger, are considered to be more critical to the final score, as changes in these regions will have a more significant impact on the model's prediction. This method enables us to gain insights into the model's decision-making process and provides us with an explanation of why the model makes a particular prediction for a given input.  

#### Local Interpretable Model-Agnostic Explanations (LIME)   
Local Interpretable Model-Agnostic Explanations (LIME) is one of the robust model-agnostic techniques widely used in literature. Thus, this technique is not dependent on the specific architecture, parameters, or training methods used in each machine-learning model.  

The algorithm aims to identify the specific superpixels in an image that have a positive or negative impact on the model's decision-making process. By analyzing the relationship between each superpixel and the model's prediction, the algorithm can determine which regions of the image are most significant in influencing the model's output. The algorithm attempts to highlight the specific superpixels that contribute the most to the model's decision, whether positively or negatively, providing valuable insights into the model's inner workings.   

#### SHapley Additive Explanations (SHAP)

Shap(Lundberg, and Lee, 2017) is a game theoretic approach that explains the individual prediction of an ML model by calculating the contribution of each feature to the model prediction. for that, the SHAP value was computed by analyzing the prediction of a model and adding up the effects of every individual feature on the outcome. These figures can aid in comprehending the significance of each feature and the extent to which it played a role in the ultimate prediction outcome. In this manner, the team can gain insight into the decision-making process of the model and pinpoint the features that hold the most significance. As an example pixels may be clustered into superpixels, and the forecast can be allocated amongst them to elucidate an image.   

In SHAP, the effect of a feature is not solely dependent on that individual feature, but rather on the entire collection of features present in the dataset. Thus, the mean absolute value of a feature's impact on a target variable can serve as a metric to gauge its significance. SHAP is a model-agnostic approach that is suitable to interpret the models like neural networks that do not provide their own interpretation of the significance of features. The SHAP model is reliable, and hence, the explanations generated can be trusted, irrespective of the model being analyzed.  

SHAP has two model-agnostic approximation methods: general SHAP and Kernal SHAP and model-type-specific approximation methods: Linear SHAP, Tree SHAP, Deep SHAP, etc. We have applied general SHAP and Kernal SHAP for our study. Shap. Explainer is the basic explainer class in SHAP that works with different kinds of data. Kernel SHAP is an approach that employs a distinct weighted linear regression technique to calculate the significance of individual features. Rather than training models again with subsets of features, Kernal SHAP utilizes the already trained full model by substituting "missing features" with "marginalized features" which are approximated using data samples.  


## 3 Methodology 

### 3.1 Dataset 
we used a dataset from kaggle which contains enough training and test images for training a deep neural network. From the following link we have download the dataset : https://www.kaggle.com/datasets/mahdinavaei/brain-tumor-mri-images-huge, but now the dataset has been removed and alternate daasets can be found from the given link : https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri. The dataset consists of 19281 MRI images belonging to three different brain tumors: glioma tumor, meningioma tumor, and pituitary tumor. The dataset was divided into the ratio of 70:15:15 for training, validation, and testing. The dataset consists of 19281 MRI images belonging to three different brain tumors: glioma tumor, meningioma tumor, and pituitary tumor. The dataset was divided into the ratio of 70:15:15 for training, validation, and testing. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%201.png">
  <br />
  <em>Figure 1: Sample Testing Images</em>
</p>

### 3.2 Data augmentation 

Data Augmentation was used to increase the size of the training dataset using Keras-based ImageDataGenerator by changing the image augmentation parameters: rescale (1. /255), transformations (shear, zoom, rotation. Flipping etc). The Deep Learning model treats these newly generated images as different from the original image. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%202.png">
  <br />
  <em>Figure 2: Augmented Images</em>
</p>

### 3.3 Design  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Picture_work_flow.png">
  <br />
  <em>Figure 3: Flow chart of the work </em>
</p>

The overall workflow of the proposed study was illustrated in Figure 3. The process includes two main parts: pre-train model training and XAI analysis.  The brain tumor dataset was pre-processed by performing data augmentation. Then four different types of pre-trained models were trained by changing the hyperparameters to identify the best pre-training model. Features were extracted, using the highest accuracy pre-trained DNN model. Then the brain tumor classification model was evaluated by applying various XAI methods. 

## 4 Results 
### 4.1 Performance of the pretrained   

The pre-trained models were modified to classify brain tumors in higher accuracy. The selected pre-train models were Initialized by ImageNet weights and overlayer with a stack of new top layers. The stack of the top layer block is comprised of global average pooling layers, three dropout layers, three dense layers and one flatten layer. Further, L2 regularization was applied to reduce the overfitting of the model. As this is a multi-class classification problem softmax function was used as the activation function. First, the model was trained by freezing all layers except the top layers.  The training and validation accuracies were around 0.85 and 0.87 which didn’t improve further. As a result, the pertained models were trained from scratch by adjusting the hyperparameters. The highest performance was achieved by the ADAM optimizer with a 0.0001 learning rate. 

| Pretrained models| Accuracy      |
| -----------------|:-------------:|
| Xception         | 97.05 | 
| Inception V3     | 96.57       |  
| ResNet50         |81.50       |
|DenseNet201       |96.43          |

The performance of the pre-train models was tabulated in the Table above. It clearly shows that the accuracy value obtained from the pre-trained model, ResNet50 was lower than the rest of the pre-train models. All other pre-trained models except ReNet50 have gained accuracy values greater than 95%. The highest accuracy value was obtained from the Xception model. The accuracy curve and training loss curve for the best model is given in below Figure. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%204.png">
  <br />
  <em>Figure 4: Training and Validation accuracy curve &  Training and validation loss curve </em>
</p>

Figure 4 (a) shows the changes in the training and validation accuracy with the number of epochs. The gradual decrease of the training loss and validation loss during the training process is shown in Figure 4(b). The confusion matrix of the Xception model shown in Figure 5,  indicates that out of 989, 963 Glioma and Pituitary tumor images 977, 956  images were classified correctly. But the identification of Meningioma images much less compare to Glioma and Pituitary tumor images.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%205.png">
  <br />
  <em>Figure 5: Confusion matrix for all three-class obtained for the training dataset </em>
</p>

### 4.2 Performance of XAI methods

In this work, a Grad-CAM model was built, and visualization maps were produced for both accurate and inaccurate predictions. The contributions of each layer to the model architecture were also visualized. Our findings show how well Grad-CAM highlights the areas of the input image that had the most impact on the model's prediction. We were able to understand the model's decision-making process and pinpoint the areas of the input image that were most important for producing precise predictions by looking at the visualization maps. These results show the potential of Grad-CAM as a tool for deciphering and evaluating deep learning model internals.  

After finetuning the pre-trained model, Grad Cam, LIME and SHAP explainers were applied on a random test set, and various visualizations were made using the code in this Github Repository. In addition to the highest accuracy Xceptional model, a low accuracy Xceptional model was used to compare performance. 

#### 4.2.1 GradCAM 

In this work, a Grad-CAM model was built, and visualization maps were produced for both accurate and inaccurate predictions. The contributions of each layer to the model architecture were also visualized. Our findings show how well Grad-CAM highlights the areas of the input image that had the most impact on the model's prediction. We were able to understand the model's decision-making process and pinpoint the areas of the input image that were most important for producing precise predictions by looking at the visualization maps. These results show the potential of Grad-CAM as a tool for deciphering and evaluating deep learning model internals. Figure 6 shows the output of the Grad-CAM for different classes.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%206.png">
  <br />
  <em>Figure 6: Grad-CAM Images for Giloma, Meningioma, and pituitary tumor </em>
</p>

Grad-CAM has the distinctive feature of visualizing the effect of each layer on the prediction, which is an especially useful visualization technique to analyze how each convolution layer has contributed to the final prediction. This is useful for designing the model and seeing how different models’ convolutions can be changed when creating our own neural networks for any kind of image classification. Figure 7  shows the visualization of different layers the heat map will look like, and this image is from a correct prediction: Giloma obtained for the ground truth image of Giloma for a model of Xception with an accuracy of around 80%. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%207.png">
</p>

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%208.png">
  <br />
  <em>Figure 7: Grad-CAM visualization for each layer in the Xception model (80% Accuracy) for sample GIloma Image </em>
</p>

#### 4.2.2 LIME 

We initially choose an image to be described before using LIME to do an image classification task. Then, by making minor adjustments to the pixels, such as adding or removing a small bit of noise, we create a series of image perturbations. We run the black box image classifier on each modified image and log the resulting class probabilities.  

Then, we train a more straightforward, understandable model that can account for the black box model's predictions using the perturbed images and their corresponding class probabilities. A linear regression model that learns the correlation between the altered pictures and the accompanying class probabilities often serves as the interpretable model.

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%209.png">
  <br />
  <em>Figure 8: Lime Heat Maps for Three Different Classes [True class - Glioma   Predict class –Giloma] </em>
</p>

Figure 8 shows the heat map visualization of LIME for all the classes for a particular image of Giloma and from the figure it can be seen each pixel of the image was given a value between –0.15 to 0.15 where each of these values indicates how well the regions of the images were contributed positively or negatively to the final image. As for Meningioma, you can see all the values are positive paving the way for the prediction of the model to Meningioma. There are many negative values on GIloma and Pituitary tumor predictors. More importantly, even though the regions of the tumor get a positive value through LIME prediction, it is not highly positive (no dark blue color near tumor areas in Meningioma).   

The below Figure 9 shows a visualization of the negative and positive impact of the model in a clearer way. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2010.png">
  <br />
  <em>Figure 9: Positive (Green) and Negative (Red) Values Visualization in LIME Images [True class - Glioma   Predict class –Giloma] </em>
</p>

#### 4.2.3 SHAP 
Before applying the SHAP explainer, a mask was created for the testing image which will help the SHAP to highlight the areas on the testing image. the explainer applied to the image in larger number of times output images will be more refined. Figure 4 shows images belonging to three classes after applying the explainer 4000 times.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Picture_Shap.png">
   <br />
  <em>Figure 10: SHAP images for Negative  (Blue) and Positive (Red) values </em>
</p>


Each image in Figure 10 was classified correctly from the Xception model with higher accuracy. The line below shows the intensity of the SHAP value varies from negative to positive. Blue pixels indicate the negative intensities which decrease the probability of that image belonging to a specific class. In contrast, bright red pixels have more positive intensity, and it increases the probability of that class. The first glioma image has bright red pixels around the tumor it shows 99.9 % accuracy of being in the Glioma class. The probability of belonging to a Meningioma and Pituitary tumor is 2% and 1% respectively and both have more pixels than red. Similar results were obtained for other images also.   

## 5 Analysis 

In this study, we conduct an experimental analysis on pre-trained models using TL as the first phase of identifying brain tumor classification using MRI images. InceptionV3, ResNet50, Xception, and DensNet were trained from starched by initializing the weights from ImageNet weights. No significant differences in accuracy values were found in all four models Inception V3, ResNet50, Xception, and DensNet. All the observed accuracy values were higher than 95% except ResNet50 for this dataset no matter the depth of the architecture.  Then Grad CAM, LIM, and SHAP techniques were applied to describe the behaviour of the pre-trained model.  

### 5.1 Grad Cam Analysis 

In Figure 4 block2_sepconv2 did not give any contribution to the final prediction as it shows a dark heatmap. A trend can be seen in initial layers like block1_conv1, block1_conv2, and block2_sepconv1 have extracted fewer features for the final prediction when compared to the last few layers which are shown in Figure 3. The final layer shows a clear heat map when compared to other layers and that's why Grad Cam usually uses the final layer for plotting the final heatmap when explaining the black boxes of the neural networks.  

A similar visualization was plotted for the correct prediction of Giloma Image is shown below in Figures 5 and 6, from both figures we couldn’t identify any distinguishable pattern for the correct and wrong prediction. The heat maps created by the other layers seem to be changing from blank to bright spots from image to image. Thus, it indicates for different images the contribution of each layer is differing rapidly and any relationships related to why it's changing cannot be predicted and it can be a gap for future researchers to explore more about how and why these pattern changes according to different images of the same classes.

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2014.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2015.png">
  <br />
  <em>Figure 11: Grad Cam Visualization for each layer for Giloma Images on 80% accuracy model </em>
</p>

In Figure 11 a trend can be seen in initial layers like block1_conv1, block1_conv2, and block2_sepconv1 have extracted fewer features for the final prediction when compared to the last few layers which are shown in Figure 3. The final layer shows a clear heat map when compared to other layers and that's why Grad Cam usually uses the final layer for plotting the final heatmap when explaining the black boxes of the neural networks.  

A similar visualization was plotted for the correct prediction of the Meningioma Image is shown below in Figure 12. Here interestingly all the layers apart from the last layer gave a dark heat map and a similar way of pattern was seen in most images of Giloma as well. The reason why this is happening can be a future exploration in this field. All Giloma images show a clear pattern of all layers producing a non-empty heat map, but in the cases of Meningioma and Pituitary tumor the layers’ heat maps were empty apart from the last layer.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2016.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2017.png">
  <br />
  <em>Figure 12: Grad CAM Visualization for each layer for Meningioma Images </em>
</p>

Additionally, when we checked the heat maps of each layer for the best-trained Xception model with 97% accuracy, even for Giloma images the pattern repeats as only the last layers were producing the heat map and other layers gave a blank heat map. This can be an interesting aspect to explore and the reason for this pattern is unexplored in our experiment. Figure 13 shows the layer-wise visualization for a Giloma image trained on the best model of Xception (97% accuracy).   

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2018.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2019.png">
  <br />
  <em>Figure 13: Grad CAM Visualization for each layer for Giloma Images on 97% accuracy model </em>
</p>

When we compare our model performance, these Grad Cam images for the final layer can be used as shown in Figure 14, there is a clear indication of Grad cam Image on the best model having high-intensity pixels on the tumor and on the other hand the Grad Cam image on the 80 percent model failed to highlight the tumor areas compared to the best model Grad Cam image.  

<figure>
  <table>
    <tr>
      <td><img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2020.png" alt="image1"></td>
      <td><img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2021.png" alt="image2"></td>
    </tr>
  </table>
  <figcaption>Figure 14: Comparision of Same Grad Cam Images on different accuracy Models </figcaption>
</figure>


### 5.2 LIME Analysis

When comparing the same image results on two different accuracy models for Lime Images we can compare how the two models identified the positive and negative features by seeing the corresponding Lime images of both models. The second corresponding image in Figure 15 illustrates some of the parts identified as negative features in the low-accuracy model were identified as positive features in the best model. Also vice versa, the features identified as positive features in images three and four on the low accuracy model are identified as negative in the best accuracy model. This indicates how these models are learning their features with increased accuracy.    

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2022.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2023.png">
  <br />
  <em>Figure 15: Lime Heat Map for a Giloma Image on 97% accuracy model(top) - [True class - Glioma   Predict class – Glioma] and 80% accuracy model(down) - [True class - Glioma   Predict class –Meningioma] </em>
</p>

### 5.3 SHAP Analysis

By comparing the SHAP images generated by two different models trained on the same input data, we can assess how each model identifies and weighs positive and negative features. In Figure 16, the second corresponding SHAP images demonstrate that some parts of the image that were identified as negative features in the low-accuracy model were instead identified as positive features in the best model, and vice versa. This shows how the models are learning to weigh different features as they increase in accuracy. Very similar to the LIME model but the LIME model will give more intuition and is clearer than the SHAP images, as it is difficult to get intuition.  

Comparing SHAP images in this way can be a useful tool for understanding how different ML/DL models approach a particular problem and can provide insight into the strengths and weaknesses of each model. Note that the specific features identified as positive or negative will depend on the problem and the features used as input to the models.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2024.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2025.png">
  <br />
  <em>Figure 16: SHAP Heat Map for a Giloma Image on 97% accuracy model(top) - [True class - Glioma   Predict class –Giloma] and 80% accuracy model(down) - [True class - Glioma   Predict class –Meningioma] </em>
</p>

### 5.4 Comparison of all XAI models 

A sample image was picked for the Comparision of all XAI methods used here, it can be seen that Grad CAM visualization is a quite simple and perfect way to LIME and SHAP images. Though LIME and SHAP contain more details in their heat maps compared with the Grad CAM images. When we compare LIME with SHAP images, we can see that Lime images are comparatively easy to visualize than SHAP as it contains, visualizable segmented areas of positive and negative compared to SHAP images, coping with the tumor size and other features that can be seen from the image. SHAP, however, contains a small collection of pixels in different ranges of feature correlations.   

More importantly, if you compare SHAP and LIME images both give the same correlation of features for the same area in the picture. The positively correlated pixels in LIME images are Indicated as the same as positively correlated pixels in SHAP (Figure 17). But the underlying algorithms and the way they calculate features are different from each other in LIME and SHAP. 

LIME uses a localized approach to generate explanations for individual predictions. It randomly perturbs the input data and measures the effect on the model's output. Based on this, it identifies the features that contribute the most to the prediction for that instance. LIME then generates a mask that highlights the regions of the image that are most responsible for the model's prediction.  

However, SHAP uses a global approach that calculates the Shapley values of each feature across the entire dataset. The Shapley value is a measure of the contribution of a feature towards the model's output across all combinations of features. SHAP then generates an image that shows the direction and magnitude of the feature's effect on the model's prediction.  

Because of the differences in the algorithms, the LIME and SHAP images may highlight different regions of the image as important. It's also possible that there may be differences in the way the two algorithms weigh and combine features, which can result in different feature importance rankings.  

Therefore, it's important to remember that while both LIME and SHAP provide useful tools for interpreting machine learning models, the results obtained may depend on the algorithm used and the input data's features. It is always a good practice to evaluate the explanations generated by both methods and compare them to better understand the factors driving the model's predictions.  

Overall, all the XAI techniques will give various kinds of insights for a particular image, to find the best XAI techniques we may need to use metrics like infidelity and sensitivity.    

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2026.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2027.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2028.png">
  <br />
  <em>Figure 17: Comparison of Grad Cam (Top), Lime (Middle), and SHAP(Down) images - True class - Glioma   Predict class -Glioma </em>
</p>

## 6 Conclusions  

This study proposed a pre-trained model-based classifier to classify brain tumors using MRI images. Proposed method choose four different pre-trained models based on their performance, the depth of the CNN, the number of parameters that we need to train and the applicability of the domain.  It is evident that training the CNN model with high accuracy is important to predict a healthier output.  

In conclusion, it can be shown from the comparison of several XAI algorithms for picture classification that each one offers distinct insights into the model's predictions. Grad CAM is a more straightforward visualization method, but it lacks the level of detail that LIME and SHAP pictures offer. Because SHAP and LIME utilize distinct techniques to determine feature importance, the rankings of features and the parts of an image are highlighted as important variables. The SHAP images of several models can be compared to provide important insights into how the models develop and weigh aspects differently as their accuracy rises.  

We also observed that initial layers in the neural network extract fewer features for final predictions when compared to the last few layers. In some cases, like in Meningioma and Pituitary tumor images, only the final layer produced non-empty heat maps. The reason for this pattern is yet unexplored and can be an interesting avenue for future research. Therefore, it is important to use multiple XAI techniques and compare their results to better understand the factors driving the model's predictions. Further research is needed to understand the patterns observed in different layers of neural networks and how they affect the model's performance.  

## 7 References  

* Abeyagunasekera, S.H.P., Perera, Y., Chamara, K., Kaushalya, U., Sumathipala, P. and Senaweera, O., 2022, April. LISA: Enhance the explainability of medical images unifying current XAI techniques. In 2022 IEEE 7th International Conference for Convergence in Technology (I2CT) (pp. 1-9). IEEE.  

* Cian, D., van Gemert, J. and Lengyel, A., 2020. Evaluating the performance of the LIME and Grad-CAM explanation methods on a LEGO multi-label image classification task. arXiv preprint arXiv:2008.01584. 

* He, K., Zhang, X., Ren, S., and Sun, J., 2016, Deep Residual Learning for Image Recognition, in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, USA, pp. 770-778. DOI 10.1109/CVPR.2016.90 

* Huang, H., Hu, X., Zhao, Y., Makkie, , Dong, , Zhao, S.  Guo, and T. Liu, 2017. Modeling task fMRI data via deep convolutional autoencoder,  IEEE Transactions on medical imaging, vol 37, no. 7, pp. 1551-1561, DOI: 10.1109/TMI.2017.2715285 

* Jin, W., Li, X., Fatehi, M. and Hamarneh, G., 2023. Guidelines and evaluation of clinical explainable AI in medical image analysis. Medical Image Analysis, 84, p.102684.  

* Lundberg, S.M. and Lee, S.I., 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30. 

* Selvaraju, R.R., Das, A., Vedantam, R., Cogswell, M., Parikh, D. and Batra, D., 2016. Grad-CAM: Why did you say that?. arXiv preprint arXiv:1611.07450.  
* Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich. A., 2015, Going deeper with convolutions, in Proc. IEEE Conference on computer vision and pattern recognition, Boston, USA, pp. 1–9. DOI: 10.1109/CVPR.2015.7298594 

* Van der Velden, B.H., Kuijf, H.J., Gilhuijs, K.G. and Viergever, M.A., 2022. Explainable artificial intelligence (XAI) in deep learning-based medical image analysis. Medical Image Analysis, p.102470. 

* https://keras.io/api/applications/ 

* https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri. 
