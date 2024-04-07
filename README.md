# Sign-Language-Detection





OPSM325
Professor Minati Rath
29th March 2024


Machine Learning II Assignment:  American Sign Language Detection 
By: 
Anisha Jasti
Asmi Nahar
Thwishaa Bansal


Link to google drive: (original dataset, augmented images, train set and validation set) https://drive.google.com/drive/folders/1mVkEIHwskNKGcJeyzBvZZtbbS51I-V3m?usp=sharing
Abstract
Sign language, a crucial form of communication for the deaf community, faces challenges in accurate recognition. This study focuses on detecting five vowels (A, E, I, O, U) in sign language using convolutional neural networks (CNNs). The research aims to develop a robust model capable of recognizing these vowels from images captured at multiple angles. Various data augmentation techniques were employed to enhance the dataset, addressing challenges such as uneven class distribution and grayscale image compatibility. We tested five different architectures with several iterations such as VGG-16, GoogleNet, ResNet, NIN, and CNN. Results showed that the most complex architectures, ResNet had the highest accuracy but given other parameters like efficiency and compatibility Google Net and our simple CNN model had more robust results.

The study's contributions include a detailed methodology for sign language recognition and insights into model performance and challenges faced. Future research could explore expanding the dataset to include more sign language symbols and languages, enhancing the model's ability to recognize signs from various cultures and perspectives.




Introduction
Sign language, a fundamental aspect of human communication since ancient times, remains a vital means of expression within the deaf community today. Its historical roots trace back to prehistoric eras, with documented references emerging notably during the Middle Ages. Sign language usage can be broadly categorized into two forms: firstly, among individuals observing vows of silence, such as those within monastic communities; and secondly, among the deaf, who rely on it as their primary mode of communication (Ruben, 2005).
The significance of sign language research is underscored by the considerable portion of the global population affected by hearing impairment, surpassing 5% (KASAPBAŞI et al., 2022). To bridge the communication gap faced by these individuals, diverse sign languages have evolved, each with unique grammatical structures distinct from spoken languages. Despite their natural emergence within deaf communities worldwide, sign languages remain relatively unfamiliar to the broader population, highlighting the need for enhanced understanding and accessibility. 
As digital technologies increasingly integrate into daily life, the role of sign language in facilitating human-machine interaction becomes increasingly pertinent. American Sign Language (ASL) stands as the primary language of choice for the deaf community in the United States, while numerous other sign languages, such as GSL, CSL, Auslan, and ArSL, cater to regional needs. Through hand signals, facial expressions, and body movements, sign language empowers seamless communication for individuals facing speech and hearing challenges.
Problem Statement
The development of effective tools for sign language recognition remains an ongoing challenge. Existing approaches often encounter difficulties in accurately detecting and interpreting sign language gestures, hindering seamless communication for individuals with hearing impairments.
This study seeks to address the limitations in sign language recognition by focusing on the specific task of detecting five distinct alphabets—A, E, I, O, and U—within sign language. The primary objective is to design, implement, and evaluate a convolutional neural network (CNN) architecture capable of robustly identifying these alphabets from image data. Other architectures like VGG-16, GoogleNet and NIN will also be used. Therefore, by collecting images and curating the datasets augmented through preprocessing techniques, this research aims to contribute to the development of more accurate and reliable sign language recognition systems.
Literature Review
American Sign Language or ASL is a commonly used sign language dataset for image recognition. Several papers informed the development of our proposed architecture. The majority reviewed employed diverse hand shapes and sizes within their training datasets, achieving high accuracy. For instance, the work titled "CNN-based Feature Extraction and Classification for Sign Language" utilized VGG-16 and AlexNet architectures, incorporating 189 samples per letter using varied hands  (Abbas et al., 2021).

Datasets have also incorporated images captured under varying lighting conditions, with fixed hand lengths, from different camera angles, and using various camera models. These techniques were employed in sign language detection studies conducted by a group of Dutch researchers, with the potential for deployment in medical applications (Kasapbasi et al., 2022). Using a varied dataset can help create a robust model with several applications. 

Sign language recognition with CNNs is making strides towards bridging the communication gap. A recent study by Manipal University researchers demonstrates how a CNN-based system, trained on a basic database of individual sign language letters, can translate signs into spoken language and text. This innovation holds immense potential for dismantling communication barriers for the deaf and hard-of-hearing community (Alaria et al., 2022).

Gap in Literature
A key limitation identified in these studies was the lack of consideration for signs presented from different angles. Sign language communication inherently involves variations in hand orientation. By incorporating these variations into the training data, our model aims to achieve greater robustness and improve its ability to recognize signs across diverse viewing angles. This enhancement would significantly improve the model's real-world applicability, enabling it to effectively recognize signs regardless of their orientation. Moreover, letters like ‘A’ and ‘E’ look similar and the model needs to recognise them from various angles.

Another significant observation was that the majority of the literature was based on ASL since it is universally used. There is an underrepresentation of other sign languages like the Indian Sign Language.
Methodology
Setup
The initial coding phase utilized Google Colab, a cloud-based platform. However, due to limitations in handling the 600-image dataset, the development environment was transitioned to Jupyter Notebook, which offered more flexibility for managing the data volume.
Images for the dataset were captured using a standard iPhone camera
Dataset of Images
   The research group collected images of the vowels A, E, I, O, and U. To enhance detection accuracy, they captured the hand signs from 20 distinct angles, all against a clean white background. This minimizes image noise and maximizes image quality. Vowels were chosen for their prevalence in forming words and their role as building blocks of language. Moreover, the hand signs for A and E are similar, making it crucial for the model to differentiate between them during detection. Including I, O, and U diversifies the hand symbol set, leading to a more robust dataset. A total of 100 images were uploaded, each letter represented by 20 images each. 
Pre-processing 
    Before inputting the data into the CNN architecture, the researchers employed various data augmentation techniques. These included image rotation, flipping, adjustments to brightness and contrast, conversion to grayscale, and edge detection. By incorporating data augmentation we increased our dataset size from 100 to 600. 
While this process expands the dataset size, the primary benefit lies in improving the model's ability to recognize hand signs in diverse real-world scenarios. Then, the images were resized to (224 x 224) as it is the standard size for architectures like VGG-16 and ResNet. 
    To assess the model's ability to generalize to unseen data,  the final dataset (600 images)was divided into training and testing sets using an 80:20 ratio. This means 480 images were allocated for training and 120 for testing. Importantly, within each set, the images for each class (A, E, I, O, U) were further divided equally, maintaining a balanced distribution across all classes. This ensures the model encounters a representative sample of each sign during training, leading to improved performance.
     All the images for training, testing and augmentation were saved on the desktop and were deployed later in the architecture. 
CNN Architecture
      The first architecture that was tried was simple CNN. This architecture consists of two convolutional layers, a max pooling layer, a fully connected layer and then an output layer. The first convolutional layer takes an image with 3 channels (RGB) as input and produces 16 feature maps with a kernel size of 3 and padding of 1. The second convolutional layer takes the output of the first layer and produces 32 feature maps with the same kernel size and padding. Max pooling layers are used after each convolutional layer to reduce the dimensionality of the data and introduce some level of translation invariance. The fully connected layer takes the flattened output from the previous layer and transforms it into 128 neurons. This layer performs  classification based on the features. The final output layer has 5 neurons, one for each vowel class (A, E, I, O, U).
     The initial training with 5 epochs resulted in a high training loss that gradually decreased. However, the final validation accuracy of 43.33% suggests underfitting. 

This means the model memorized the training data patterns too well and could not generalize to unseen examples. Extending the training to 10 epochs addressed this issue. The training loss significantly decreased, indicating improved learning. Consequently, the validation accuracy jumped to 85%, demonstrating the model's enhanced ability to recognize hand signs beyond the training data. 

Further, other architectures were run, and below is a table summarizing the results.
Architecture
Accuracy
Total Training time for 10 epochs (min)
Reason for using architecture
VGG - 16
25.00%
120 mins
VGG 16  because it is known for being effective in image recognition tasks
GoogleNet
90.38%
4.28 mins
GoogleNet is known for its depth and computational efficiency. It captures complex features in images while maintaining a manageable model size, making it suitable for real-time applications
Resnet
84%
10.72 mins
ResNet was chosen because it helps address the vanishing gradient problem. It was chosen for its ability to train very deep networks effectively, potentially improving recognition accuracy but due to its fluctuation the model is overfit
NIN
20%
-
NIN is known for its ability to capture fine details in images by using micro neural networks within the convolutional layers. Despite its relatively lower accuracy in this study, it was included for its unique approach to feature extraction.
Simple CNN
85%
4.94 mins
A simple CNN architecture was included as a baseline for comparison. It consists of a few convolutional and pooling layers followed by a fully connected layer, making it easy to implement and understand. 
CNN (4 layers)
70%
8.34 mins
More layers were added to the existing CNN architecture, to check if the accuracy of the model increased.


Challenges During Training
The following challenges were identified during the coding phase and subsequently addressed:
Unequal Class Distribution
The initial dataset went through an uneven distribution of images across the five classes (A, E, I, O, U) for training and testing. This imbalance would negatively impact the model's ability to learn effectively.
Grayscale Images
The inclusion of grayscale images within the augmented dataset caused issues. The model architecture expects RGB (red, green, blue) channels as input and grayscale images lack this color information. This mismatch led to classification errors.
Long-time to run the architectures 
It was challenging to run models like ResNet and NIN since they are extensive architectures and it took over multiple hours to complete training. This issue can be attributed to the large dataset size of images.



Results and Discussion

CNN with multiple layers

Training Loss: The training loss generally decreases over the epochs, which means the model is learning the training data. However, it is still relatively high throughout, even at the end of training.

Validation Accuracy: The validation accuracy initially increases but then fluctuates around 75% after epoch 4. This suggests that the model is not generalizing well to unseen data. In other words, the model is memorizing the training data rather than learning underlying patterns.

Overall, these results showcase that CNN model is overfitting the training data. This means that the model is learning the specific patterns of the training data too well, and it is not able to generalize well to new data.

ResNet Architecture




Training Loss: The training loss generally decreases over the epochs, which means the model is learning the training data. It starts at around 1.6 and goes down to about 0.2 by epoch 10.
Validation Accuracy: The validation accuracy initially increases and then seems to plateau around 80% after epoch 4. This suggests that the model is generalizing well to unseen data and is not overfitting the training data.
Overall, these results suggest that the ResNet model is performing well, even though there are mild fluctuations in between the graphs. The training loss is decreasing, and the validation accuracy is high and stable.
VGG-16





Training Loss: The training loss generally decreases over the epochs, which means the model is learning the training data. It starts at around 1.2 and goes down to about 0.4 by epoch 10.
Validation Accuracy: The validation accuracy increases up to about 75% around epoch 4 and fluctuates a bit afterwards. This suggests that the model might be starting to overfit the training data after epoch 4.
Overall, the results suggest that the VGG-16 model is learning the training data. However, there are also signs of potential overfitting because the validation accuracy isn't significantly improving after epoch 4. The accuracy is also very low and this could be attributed to the small dataset.
GoogleNet


Training Loss: The training loss generally decreases over the epochs, which means the model is learning the training data. It starts at around 1.6 and goes down to about 1.04 by epoch 10.
Validation Accuracy: The validation accuracy steadily increases over the epochs, reaching about 85% by epoch 10. This suggests that the model is generalizing well to unseen data and is not overfitting the training data.
Overall, these results suggest that the model is performing well. The training loss is decreasing, and the validation accuracy is high and still increasing. This suggests that the model may be able to continue training on more epochs to see if the validation accuracy keeps improving.

Conclusion
In summary, our study focused on using CNNs to detect five vowels in sign language. While complex models like ResNet achieved high accuracy, simpler models like GoogleNet and our simple CNN showed better efficiency and compatibility. This highlights the importance of balancing accuracy with computational efficiency. Future research could focus on expanding the dataset to include more sign language symbols and languages, improving the model's ability to recognize signs across various cultures and perspectives.




















References
Barbhuiya, A. A., Karsh, R. K., & Jain, R. (2020). CNN based feature extraction and classification for sign language. Multimedia Tools and Applications, 80(2), 3051–3069. https://doi.org/10.1007/s11042-020-09829-y 
KASAPBAŞI, A., ELBUSHRA, A. E., AL-HARDANEE, O., & YILMAZ, A. (2022). DeepASLR: A CNN based Human Computer Interface for american sign language recognition for hearing-impaired individuals. Computer Methods and Programs in Biomedicine Update, 2, 100048. https://doi.org/10.1016/j.cmpbup.2021.100048 
Ruben, R. J. (2005). Sign language: Its history and contribution to the understanding of the biological nature of language. Acta Oto-Laryngologica, 125(5), 464–467. https://doi.org/10.1080/00016480510026287 


