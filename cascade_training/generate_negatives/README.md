# Generate Negatives #

Generates hard negatives to use for classifier training. Required to function:

+ Cascade generated with the create_cascade function
+ Video without target object occurences

# Parameters #

Ordered Parameters:

1. Negative video path
2. Classifier path. This should be an xml file. Classifiers generated with the other code should have the naming format: cascade_Y.xml. Y represents the stage of the classifer. Y should be the last stage generated (highest number)

Other Parameters:

+ \-\-save-large \-\- If present the code will save versions of the image that are not resized to 20x20
+ \-negative_count X \-\- This will stop the code after X many negatives have been generated. Useful for long videos that can generate 100,000+ negatives.