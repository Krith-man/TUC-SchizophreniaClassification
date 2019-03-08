Code developed for the labs of the fourth year's undergraduate course 
Statictical Modeling & Pattern Recognition of the Electrical and 
Computer Engineering school at Technical University of Crete.

Running the program through terminal on linux:

- Navigate to directory where this readme.txt file exists an make it the 
current working directory of a terminal

- Activate python 3 virtual environment:
	$ source venv/bin/activate

- To run the program:
	$ python schizophrenia.py

- The program can take a number of optional parameters as described below:
	$ python schzophrenia.py <normalize> <enable PCA> <PCA Accuracy>
	normalise: True/False
	enable PCA: True/False
	PCA Accuracy: float in range of [0, 1]
