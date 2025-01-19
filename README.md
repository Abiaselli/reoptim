# REOptim
Reverse-engineered Tabular Optimization Model

Hello! Thanks for checking out my repo, my name is Austin Biaselli. Here I will be posting a new model framework that I am developing to simplify tensor operations for my models and allow a more transparent relationship for what would normally be hidden states in a neural network model.

Let me explain basically what this is, it is a tabular cell based software where layers of the model are viewed as spreadsheets, where each cell can be selected and have data added to it, and the operations between each cell and the corresponding cell in the next layers are able to be easily checked, modified, expressed, and derived from the GUI display. Basically, out of a desire to simplify dimensionality matching and issues with dimensional compatibility in frameworks like PyTorch, I have decided to create a program where the 3rd/4th dimensions of the model can be expressed as a spreadsheet tab (a layer, or page in the notebook) and instead of hidden states the model is processed through each layer iteratively and this can all be viewed by a human controller to modify, study, and improve on the model concept.

Each cell can contain a value, or a set of values as a 2D tensor (I may upgrade this eventually), and can have operations performed on it by the matching cell on the next layer which follows the same property, and this creates the layer on the page after those two which can then be operated on iteratively by the pages on its subsequent layer. The goal of this, is to make every odd layer a hidden state, with the first layer being the input and every even layer being a layer of the transformer. 

I am hoping to combine this with my symbolic transformer framework, to develop a clear system for following the algebraic properties of the model. Think of how cells in an excel spreadsheet can accept equations that determine their values, but on the scale of a neural network model. 

The eventual goal is to develop a model framework that can reverse engineer viable models for dataset pairs, and infer the usable relationships and patterns based on a direct alignment for input and output batches without requiring iterative gradient optimization, maybe by generating a list of possible optins and performing a genetic selection on ideal candidates that satisfy increasingly complex datasets.

The basic workflow currently is every odd tab is a result page (except for the initial input) and every even tab is the computation tab where you can set the values and define the computations. I  will work on modifying it to increase the viability for training models, and handling datasets.

Update: sSorry I have gotten behind on updating my projects! I decided to go back to college to get a degree and my schoolwork kept me busy this week, I finally had time to work on this and have integrated the GUI options and most of the code for the basic preparation, now i need to integrate the model classes and modify the architecture to allow consistent updating of the cells via multithreading for a constant view of the internal mechanics of the operations. 
