# import numpy as np

# class LSTM:
#   def __init__(self, input_size, hidden_size, output_size):
#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.output_size = output_size
    
#     # Weights for input gate
#     self.W_ii = np.random.randn(hidden_size, input_size)
#     self.W_hi = np.random.randn(hidden_size, hidden_size)
#     self.b_i = np.zeros((hidden_size, 1))
    
#     # Weights for forget gate
#     self.W_if = np.random.randn(hidden_size, input_size)
#     self.W_hf = np.random.randn(hidden_size, hidden_size)
#     self.b_f = np.zeros((hidden_size, 1))
    
#     # Weights for output gate
#     self.W_io = np.random.randn(hidden_size, input_size)
#     self.W_ho = np.random.randn(hidden_size, hidden_size)
#     self.b_o = np.zeros((hidden_size, 1))
    
#     # Weights for cell state
#     self.W_ig = np.random.randn(hidden_size, input_size)
#     self.W_hg = np.random.randn(hidden_size, hidden_size)
#     self.b_g = np.zeros((hidden_size, 1))
    
#     # Output weights
#     self.W_hy = np.random.randn(output_size, hidden_size)
#     self.b_y = np.zeros((output_size, 1))
    
#   def forward(self, inputs):
#     # Initialize empty lists to store the intermediate states
#     outputs = []
#     cell_states = []
#     hidden_states = []
    
#     # Initialize the initial hidden state and cell state
#     hidden_state = np.zeros((self.hidden_size, 1))
#     cell_state = np.zeros((self.hidden_size, 1))
    
#     # Loop through the input sequence
#     for t in range(len(inputs)):
#       # Get the current input
#       input = inputs[t]
      
#       # Compute the input gate
#       input_gate = sigmoid(np.dot(self.W_ii, input) + np.dot(self.W_hi, hidden_state) + self.b_i)
      
#       # Compute the forget gate
#       forget_gate = sigmoid(np.dot(self.W_if, input) + np.dot(self.W_hf, hidden_state) + self.b_f)
      
#       # Compute the output gate
#       output_gate = sigmoid(np.dot(self.W_io, input) + np.dot(self.W_ho+ self.b_o))
      
#       # Compute the cell state
#       cell_state = np.multiply(forget_gate, cell_state) + np.multiply(input_gate, np.tanh(np.dot(self.W_ig, input) + np.dot(self.W_hg, hidden_state) + self.b_g))
      
#       # Compute the hidden state
#       hidden_state = np.multiply(output_gate, np.tanh(cell_state))
      
#       # Compute the output
#       output = np.dot(self.W_hy, hidden_state) + self.b_y
      
#       # Append the intermediate states to the lists
#       outputs.append(output)
#       cell_states.append(cell_state)
#       hidden_states.append(hidden_state)
      
#       print(outputs, cell_states, hidden_states);
#     return outputs, cell_states, hidden_states

# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))

# lstm = LSTM(input_size=20, hidden_size=50, output_size=20)

import torch

# hidden_size = 20
# input_size = 2
# output_size = 3
# time_steps = 10

# Define a function to generate synthetic time series data
def generate_data(time_steps, input_size, output_size):
    # Generate random intput and output data
    x_input = torch.randn(time_steps, input_size)
    y_output = torch.randn(time_steps, output_size)
    return x_input, y_output

# Define a loss function to calculate error
def loss_fn(predictions, ground_truth):
    return torch.mean((predictions - ground_truth)**2)

# Generate synthetic time series data
x_input, y_output = generate_data(time_steps=10, input_size=32, output_size=32)

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights for the input gate, forget gate, output gate, and cell update
        self.W_i = torch.randn(input_size, hidden_size)
        self.W_f = torch.randn(input_size, hidden_size)
        self.W_o = torch.randn(input_size, hidden_size)
        self.W_c = torch.randn(input_size, hidden_size)
 
        # Initialize the biases for the input gate, forget gate, output gate, and cell update
        self.b_i = torch.randn(hidden_size)
        self.b_f = torch.randn(hidden_size)
        self.b_o = torch.randn(hidden_size)
        self.b_c = torch.randn(hidden_size)

        # Initialize the weights for the output layer
        self.W_out = torch.randn(hidden_size, output_size)

        #Initialize the bias for the output layer
        self.b_out = torch.randn(output_size)

        self.y = y_output
        self.x = x_input
    
    def lstm(self, input, h , c, input_size=32):
        # Calculate the input gate, forget gate, output gate and cell update
        i = sigmoid(torch.mm(input.view(-1, input_size),self.W_i) + self.b_i)
        f = sigmoid(torch.mm(input.view(-1, input_size), self.W_f) + self.b_f)
        o = sigmoid(torch.mm(input.view(-1, input_size), self.W_o) + self.b_o)
        c_update = tanh(torch.mm(input.view(-1, input_size), self.W_c) + self.b_c)
        
        # Update the cell state
        c = f*c+i*c_update
        
        # Calculate the hidden state
        h = o * tanh(c)

        # Calculate the output
        out = sigmoid(torch.mm(h, self.W_out) + self.b_out)

        return h, c, out

    def forward(x, h=None, c=None, hidden_size=20, future=0, ):
        # If no hidden state and cell state are provided, initialize them to be all zeros
        if h is None:
            h = torch.zeros(hidden_size)
        if c is None:
            c = torch.zeros(hidden_size)
        
        # Initialize an empty list to store the output sequences
        output_seq = []

        print("X SHAPE:")
        print(x.shape)
        # Iterate through each time step
        for t in range(x.shape[0]):
            # Get the input for the current time step
            input = x[t]

            # Calculate the new hidden state, cell state and output
            h, c, out = lstm(input, (h, c))

            # Append the current output to the output sequence
            output_seq.append(out)
        
        # If we need to make predictions for future time steps, make them now
        for t in range(future):
            # Use the current hidden state as the input for the next time step
            input = h

            # Calculate the new hidden state and cell state
            h, c = lstm(input, (h, c))

            # Append the current hidden state to the output sequence
            output_seq.append(h)
        
        # Return the output sequences
        print(output_seq)
        return output_seq

def tanh(x):
    return (2 * sigmoid(2 * x) - 1)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

print(y)
print(len(x.tolist()))

# Initialize the LSTM layer with the input_size, hidden_size, and output_size
lstm = LSTM(input_size=32, hidden_size=30, output_size=32)
lstm = LSTM.forward(lstm)

# Use the LSTM to make predictions for the entire input sequence
predictions = lstm(x)

# Calculate the loss between the predictions and the ground truth
loss = loss_fn(predictions, y)