# Movie Scene Detector

A Python project that uses binary image classification to detect whether individual frames from clips are real or AI generated. Makes use of a neural network which is trained on two classes of data: frames generated using viral Veo3 videos online (Google's latest AI video generational model) alongside frames from real world movie scenes for comparison. 

# Functionality

• User inputs an image

• If the file path the user has entered is valid, the image will be passed to the classification model

• The classification model utilises the dataset it was trained on to detect whether the user's input represents a frame that is either real or AI generated 

•  If the model is uncertain that the user's image represents a frame from a clip/movie scene (e.g. probability it belongs to either of the classes is less than 0.9), we raise an error

# Future Improvements

• Scrape a wider range of video sharing platforms to develop a more accurate model

• Install a GPU for efficiency in testing

• Potentially create more classes to increase model accuracy (current model may falsely predict that a fictional movie is AI generated)

# How To Run

The model has already been trained, so all you need to do is run this command:

<pre> python moviescenedetector.py </pre>

Make sure to load the images you want to test into the repository provided, so the program is able to identify the file path.





