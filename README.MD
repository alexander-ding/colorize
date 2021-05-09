# Colorize
For my onboarding project at Brown Visual Computing, I decided to take a crack at creating a deep learning model to colorize black and white photos. There is a lot of prior work on this topic, which I turn to from time to time for guidance, but this was mostly for me to explore the process of tackling an interesting ML problem without relying on existing solutions. 

## Data
To train our models, we needed a collection of colored images. Instead of using common image classification benchmarks like ImageNet, I wanted higher quality images that would make for good presentation. Therefore, I used the [Unsplash Lite dataset](https://github.com/unsplash/datasets), a collection of 25,000 images from [Unsplash](unsplash.com), a website where photographers upload their work for free public use. The download and processing script is on the top of [a Jupyter notebook](https://github.com/alexander-ding/colorize/blob/main/colorize.ipynb). We used 15,000 images for training, 5,000 for validation, and 5,000 for testing. 

In addition, I employed data augmentation to create a robust training dataset. Specifically, we applied random crops and horizontal flips to the training dataset when sampling. 

### LAB vs RGB
One important improvement we make to simplify the issue is to represent colored images in [LAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space). By expressing colors in three values--L* (perceptual lightness), a* (green/red), and b* (blue/yellow)--, LAB separates the black and white part of an image into a single channel, L*, and it leaves the task of coloring entirely to a* and b*. This allows us to frame the problem as: given a black and white image of size 1xHxW, representing the L* channel, generate a 2xHxW array, representing the a* and b* channels, to fill in the colors. We then combine the 3 channels and convert the result into RGB for regular image encoding. 

Our [dataloader](https://github.com/alexander-ding/colorize/blob/main/colorize/datasets/ColorizeDataset.py) implements the logic of loading JPGs, converting them into LAB, and separating out the channels into inputs and outputs. 

## Architectures
### First attempt
The first thing that came to me was an autoencoder architecture. We need a model that does image-to-image translation, and we can take advantage of lower-level semantic features extracted by pretrained CNNs to inform the upsampling layers about which colors to pick. Specifically, I started with a pretrained Resnet50, extracted mid-level features (256x56x56 in size), and upsampled from these features to create a 2x224x224 output that corresponds to the a* and b* channels. The model is defined [here](https://github.com/alexander-ding/colorize/blob/main/colorize/models/ResnetAutoEncoder.py). 

Another important decision was the choice of loss function. For a first attempt, I simply used mean-squared-error (MSE) to calculate per-pixel differences between the predicted output and the desired output. 

For training, I used the Adam optimizer with learning rate set to 0.001 and early stopping enabled based on validation loss (with patience 3). After 20 epochs of training, the results are as follow:
[results go here]()

The validation loss curve is here:
[]()

The results are poor. Most attempts at recolorization heavily use brown-ish colors and rarely ventures into more vibrant color spaces. As it turns out, brown is the most numerically average color to use, and the model learns to color conservatively since this is punished by the loss function much less than choosing a color that deviates from the average more. To illustrate the point, computing the loss for coloring everything brown on a sample batch of training data shows that its loss is only slightly higher than our trained model's coloring's loss, but this is much smaller than the loss of a random distribution between the acceptable output range. This is because the color distribution in regular photos are much more heavily concentrated around the color brown, and the model has learned to take the shortcut and struggles to learn the detaield nuances. 

### Second Attempt
I thought the issue our first model had was that it had limited information about the image itself. All it had was higher level semantic information about features of the image, but the encoding the semantic information trades off the spatial information available in the original image. Therefore, the model struggled to do much better after the initial "find the average" stage. Therefore, my solution was to switch to a fully-convolutional network (FCN), which combines information from every level of features of the backbone network, from the lowest level, which has detailed spatial information about the image itself, to the highest level, which contains abstract information about the whole image. I expected the model to be able to color with more nuance based on the smenatic information it extracts from the image. 

The model is defined [here](https://github.com/alexander-ding/colorize/blob/main/colorize/models/ResnetFCN.py), using a Resnet101 backbone pretrained on ImageNet. 

With the same training setup, this new model improved somewhat compared to the first attempt. The results are as follow:
[]()

I was able to handpick some surprisingly convincing attempts that showcase color usage other than the dull brown-ish color the first model always resorted to. 
[]()

The model seemed to pick up on certain colors common in large areas of an image, like blue skies and water bodies. However, despite the larger color palette it used, the model still struggles with vibrant colors in general and failed to produce reasonable results in general. 

### Third Attempt
After reviewing our methodology and doing some research online, I realized that our biggest issue was the loss function. MSE encouraged average brown-ish colors with how the math worked out, and others struggled to produce good results with MSE. There were some interesting alternative handcrafted loss functions that encouraged more vibrant colors, as discussed in [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) by Zhang et al. However, a more robust solution was to dynamically adapt the loss function using an generative adversarial network (GAN). There has been successful prior work using this approach that achieved state-of-the-art performance, such as [pix2pix](https://github.com/phillipi/pix2pix) and [DeOldify](https://github.com/jantic/DeOldify), and this third attempt is to replicate their success. 

A GAN has two models, a generator and a discriminator. In our case, the generator takes in the black and white image and attempts to generate a convincing coloring, and the discriminator is trained to tell apart real colored images and fake ones generated by the generator. Instead of only judging the generator based on its outputs' similarity to the ground truth, we also include an additional part of its loss based on how well it can deceive the discriminator. We alternate between training the generator and the discriminator, so that as the generator gets better at fooling the discriminator, the discriminator also figures out new ways the generator is unconvincing. Ultimamtely, the discriminator serves to dynamically modify the loss function to punish any sort of cheating behavior from the generator that might score well with the fixed MSE loss but result in unconvincing results. 

The overall idea of our generator's architecture is very similar to our second attempt, except we use a [U-Net](https://arxiv.org/pdf/1505.04597.pdf) instead of a FCN. U-Nets are an iteration inspired by FCNs to improve performance by having more sophisicated connections between the different levels of features (instead of adding them together like FCN does). The discriminator is a straight-forward convolutional neural network that determines if each section of an image seems generated or real. 

## References
1. https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d
2. https://github.com/phillipi/pix2pix
3. https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
4. https://github.com/jantic/DeOldify
