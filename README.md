# Detecting-Depression-from-Physiological-Features-on-the-basis-of-Neural-Networks-and-Genetic-Algorit

There are five python files in total (ga_for_features.py, ga_for_learning_rate.py, nn.py, nn_plus_gis.py and preprocessing.py).

Besides, the datasets used in this paper are gsr_features.csv, pupil_features.csv and skintemp_features.csv.

To run the code, firstly, run the preprocessing.py and this file is to implement preprocessing for dataset.

Then run the other four python files. To be specific, "ga_for_features.py" is to utilize genetic algorithm to find an optimal combination of 
the original 85 features. In addition, "ga_for_learning_rate.py" is to utilize genetic algorithm to find an optimal learning rate.
Furthermore, "nn.py" is used to detect levels of depression by using a three-layer neural network. 
Lastly, "nn_plus_gis.py" is to detect levels of depression by using a neural network; 
this network is combined with the GIS technique and the number of input neurons are 85.