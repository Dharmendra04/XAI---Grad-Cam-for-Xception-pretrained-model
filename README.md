# XAI---Grad-Cam-for-Xception-pretrained-model
Grad-CAM is a technique that generates heatmaps to highlight the regions of an input image that are important for a model's prediction, providing insight into the model's decision-making process.
# Explainable Artificial Intelligence (XAI) in Transfer learning-based Brain Tumor Image Analysis 

## 1.What this Repository about
This Repository is about classifying brain tumor images by applying transfer learning and analysis using different kinds of XAI methods. Pre-trained models were fine-tuned using the ImageNet dataset and choose the best model by comparing the accuracy curve and training loss. Then applied the Explainable AI methods for the best and low accuracy models to compare the prediction performance. For this three different kinds of Explainable AI methods such as Grad CAM (Gradient-weighted Class Activation Mapping), LIME (Local Interpretable Model Agnostic Explanations), and SHAP ((SHapley Additive exPlanations) were chosen. According to our knowledge, this is the first project which applied these three commonly used XAI methods for a dataset and compared those XAI methods.

You can find the links for the pretrained models here,

* Best_model : https://drive.google.com/file/d/1byX3XP8bAe95NtMFlYX8cuRqmT2n3Sm5/view?usp=share_link
* Low_accuracy_model : https://drive.google.com/file/d/1n9BCLThjO5QWr3HHjkUrs8p0GA9QOV3g/view?usp=share_link

You can find the implementation of the Different XAI methods with accuracy model of around 80%. Please find the code links below

* 80% accuracy model: https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Low_Accuracy_Model_xception_XAI_Analysis.ipynb


You can find the dataset we use for training the Xception Model here,
* MRI_Dataset : https://drive.google.com/file/d/1uYx2MgwD75rWNO3ul8-YjDzpiYZSkKoK/view?usp=share_link


## 2.Introduction

#### Aim 

Identify brain tumors using artificial intelligence

#### Objectives  

* Find a pre-train model with high accuracy to classify brain tumors of the selected datasets  

* Explore various XAI methods to explain the prediction.  

* Apply various XAI methods to explain the model predictions. 


### 2.1 Gradient-weighted Class Activation Mapping (Grad CAM) 

The Gradient-weighted Class Activation Mapping (Grad CAM) approach utilizes the gradients of a target concept, such as the predicted category of an image or a textual description, which flow into the last convolutional layer of a neural network model. This method generates a rough map of key areas in the image that are significant in predicting the target concept (Selvaraju et al., 2016).   

Additionally, the pixels that are considered by each of the convolution layers also can be visualized, hence a visualization of how each layer extracts unique features of an image can be clearly visualized. This kind of visualization from each layer cannot be visualized in other XAI methods, like SHAP, and LIME.  

Gradient-based methods XAI methods like Grad CAM are transparent and do not require any additional model training or modification (Cian et al., 2020). They work by analyzing the gradients of a neural network during its prediction, making it easy to interpret and understand the model's decision-making process. Thus, this model can be used to produce real-time visualization.   

As a technical overview, In the Grad-CAM technique, the gradients of the classification score are used in conjunction with the final convolutional feature map to identify the regions of an input image that have the most significant influence on the classification score. Specifically, the technique examines the magnitude of the gradient with respect to the final convolutional feature map to determine where the classification score depends most on the image data. Regions, where the gradient is larger, are considered to be more critical to the final score, as changes in these regions will have a more significant impact on the model's prediction. This method enables us to gain insights into the model's decision-making process and provides us with an explanation of why the model makes a particular prediction for a given input.  

### 2.2 Local Interpretable Model-Agnostic Explanations (LIME)   
Local Interpretable Model-Agnostic Explanations (LIME) is one of the robust model-agnostic techniques widely used in literature. Thus, this technique is not dependent on the specific architecture, parameters, or training methods used in each machine-learning model.  

The algorithm aims to identify the specific superpixels in an image that have a positive or negative impact on the model's decision-making process. By analyzing the relationship between each superpixel and the model's prediction, the algorithm can determine which regions of the image are most significant in influencing the model's output. The algorithm attempts to highlight the specific superpixels that contribute the most to the model's decision, whether positively or negatively, providing valuable insights into the model's inner workings.   




## 3 Results 

### 3.1 Performance of XAI methods

In this work, a Grad-CAM model was built, and visualization maps were produced for both accurate and inaccurate predictions. The contributions of each layer to the model architecture were also visualized. Our findings show how well Grad-CAM highlights the areas of the input image that had the most impact on the model's prediction. We were able to understand the model's decision-making process and pinpoint the areas of the input image that were most important for producing precise predictions by looking at the visualization maps. These results show the potential of Grad-CAM as a tool for deciphering and evaluating deep learning model internals.  

After finetuning the pre-trained model, Grad Cam, LIME and SHAP explainers were applied on a random test set, and various visualizations were made using the code in this Github Repository. In addition to the highest accuracy Xceptional model, a low accuracy Xceptional model was used to compare performance. 

#### 3.1.1 GradCAM 

In this work, a Grad-CAM model was built, and visualization maps were produced for both accurate and inaccurate predictions. The contributions of each layer to the model architecture were also visualized. Our findings show how well Grad-CAM highlights the areas of the input image that had the most impact on the model's prediction. We were able to understand the model's decision-making process and pinpoint the areas of the input image that were most important for producing precise predictions by looking at the visualization maps. These results show the potential of Grad-CAM as a tool for deciphering and evaluating deep learning model internals. Figure 6 shows the output of the Grad-CAM for different classes.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%206.png">
  <br />
  <em>Figure 1: Grad-CAM Images for Giloma, Meningioma, and pituitary tumor </em>
</p>

Grad-CAM has the distinctive feature of visualizing the effect of each layer on the prediction, which is an especially useful visualization technique to analyze how each convolution layer has contributed to the final prediction. This is useful for designing the model and seeing how different models’ convolutions can be changed when creating our own neural networks for any kind of image classification. Figure 7  shows the visualization of different layers the heat map will look like, and this image is from a correct prediction: Giloma obtained for the ground truth image of Giloma for a model of Xception with an accuracy of around 80%. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%207.png">
</p>

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%208.png">
  <br />
  <em>Figure 2: Grad-CAM visualization for each layer in the Xception model (80% Accuracy) for sample GIloma Image </em>
</p>

#### 3.1.2 LIME 

We initially choose an image to be described before using LIME to do an image classification task. Then, by making minor adjustments to the pixels, such as adding or removing a small bit of noise, we create a series of image perturbations. We run the black box image classifier on each modified image and log the resulting class probabilities.  

Then, we train a more straightforward, understandable model that can account for the black box model's predictions using the perturbed images and their corresponding class probabilities. A linear regression model that learns the correlation between the altered pictures and the accompanying class probabilities often serves as the interpretable model.

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%209.png">
  <br />
  <em>Figure 3: Lime Heat Maps for Three Different Classes [True class - Glioma   Predict class –Giloma] </em>
</p>

Figure 8 shows the heat map visualization of LIME for all the classes for a particular image of Giloma and from the figure it can be seen each pixel of the image was given a value between –0.15 to 0.15 where each of these values indicates how well the regions of the images were contributed positively or negatively to the final image. As for Meningioma, you can see all the values are positive paving the way for the prediction of the model to Meningioma. There are many negative values on GIloma and Pituitary tumor predictors. More importantly, even though the regions of the tumor get a positive value through LIME prediction, it is not highly positive (no dark blue color near tumor areas in Meningioma).   

The below Figure 9 shows a visualization of the negative and positive impact of the model in a clearer way. 

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2010.png">
  <br />
  <em>Figure 4: Positive (Green) and Negative (Red) Values Visualization in LIME Images [True class - Glioma   Predict class –Giloma] </em>
</p>


## 5 Analysis 

### 5.1  Grad Cam Analysis 

In Figure 4 block2_sepconv2 did not give any contribution to the final prediction as it shows a dark heatmap. A trend can be seen in initial layers like block1_conv1, block1_conv2, and block2_sepconv1 have extracted fewer features for the final prediction when compared to the last few layers which are shown in Figure 3. The final layer shows a clear heat map when compared to other layers and that's why Grad Cam usually uses the final layer for plotting the final heatmap when explaining the black boxes of the neural networks.  

A similar visualization was plotted for the correct prediction of Giloma Image is shown below in Figures 5 and 6, from both figures we couldn’t identify any distinguishable pattern for the correct and wrong prediction. The heat maps created by the other layers seem to be changing from blank to bright spots from image to image. Thus, it indicates for different images the contribution of each layer is differing rapidly and any relationships related to why it's changing cannot be predicted and it can be a gap for future researchers to explore more about how and why these pattern changes according to different images of the same classes.

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2014.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2015.png">
  <br />
  <em>Figure 5: Grad Cam Visualization for each layer for Giloma Images on 80% accuracy model </em>
</p>

In Figure 11 a trend can be seen in initial layers like block1_conv1, block1_conv2, and block2_sepconv1 have extracted fewer features for the final prediction when compared to the last few layers which are shown in Figure 3. The final layer shows a clear heat map when compared to other layers and that's why Grad Cam usually uses the final layer for plotting the final heatmap when explaining the black boxes of the neural networks.  

A similar visualization was plotted for the correct prediction of the Meningioma Image is shown below in Figure 12. Here interestingly all the layers apart from the last layer gave a dark heat map and a similar way of pattern was seen in most images of Giloma as well. The reason why this is happening can be a future exploration in this field. All Giloma images show a clear pattern of all layers producing a non-empty heat map, but in the cases of Meningioma and Pituitary tumor the layers’ heat maps were empty apart from the last layer.  

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2016.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2017.png">
  <br />
  <em>Figure 6: Grad CAM Visualization for each layer for Meningioma Images </em>
</p>

Additionally, when we checked the heat maps of each layer for the best-trained Xception model with 97% accuracy, even for Giloma images the pattern repeats as only the last layers were producing the heat map and other layers gave a blank heat map. This can be an interesting aspect to explore and the reason for this pattern is unexplored in our experiment. Figure 13 shows the layer-wise visualization for a Giloma image trained on the best model of Xception (97% accuracy).   

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2018.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2019.png">
  <br />
  <em>Figure 7: Grad CAM Visualization for each layer for Giloma Images on 97% accuracy model </em>
</p>

When we compare our model performance, these Grad Cam images for the final layer can be used as shown in Figure 14, there is a clear indication of Grad cam Image on the best model having high-intensity pixels on the tumor and on the other hand the Grad Cam image on the 80 percent model failed to highlight the tumor areas compared to the best model Grad Cam image.  

<figure>
  <table>
    <tr>
      <td><img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2020.png" alt="image1"></td>
      <td><img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2021.png" alt="image2"></td>
    </tr>
  </table>
  <figcaption>Figure 8: Comparision of Same Grad Cam Images on different accuracy Models </figcaption>
</figure>


### 5.2 LIME Analysis

When comparing the same image results on two different accuracy models for Lime Images we can compare how the two models identified the positive and negative features by seeing the corresponding Lime images of both models. The second corresponding image in Figure 15 illustrates some of the parts identified as negative features in the low-accuracy model were identified as positive features in the best model. Also vice versa, the features identified as positive features in images three and four on the low accuracy model are identified as negative in the best accuracy model. This indicates how these models are learning their features with increased accuracy.    

<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2022.png">
</p>
<p align="center">
  <img src="https://github.com/Plymouth-University/2023-comp5013-gp-neural_ninjas/blob/main/Images/Picture%2023.png">
  <br />
  <em>Figure 9: Lime Heat Map for a Giloma Image on 97% accuracy model(top) - [True class - Glioma   Predict class – Glioma] and 80% accuracy model(down) - [True class - Glioma   Predict class –Meningioma] </em>
</p>



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
