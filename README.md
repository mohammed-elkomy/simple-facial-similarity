# simple-facial-similarity
Using siamese networks for one-shot facial similarity

# Data 
A very tiny data is shipped with the repo which contains a folder for each person's face which is then augmented and used for generating many positive(same face | same folder) and negative pairs(different faces | different folders) for the siamese network.

# Approach
A simple CNN network (trained from scratch) used for the siamese pair with L2 distance loss for comparing image features.
The data is simply generated at runtime through picking a person's image from a folder then another face image is picked either from the same folder (positive pair) or from a different folder (negative pair) with equal probability for positive and negative pairs.

## The main trick 
choosing a good margin is the reason for acceptable separation between positive and negative pairs. 
I made this by applying a batch normalization layer just before the L2 distance function which means the distance would range from 0 to number_of_features (in my case 8) so my margin is 1.5 since the range for L2 ouputs is [1,sqrt(8)]

## Interpreting results
![Demo](https://i.imgur.com/WMlnkxV.png)

