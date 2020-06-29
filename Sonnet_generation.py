#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import RNN
from keras.layers import BatchNormalization
from keras.utils import np_utils

#Import dataset
text=(open("sonnets.txt").read())
text=text.lower()

#Visualizing each sonnet character length
sonnets = text.split('\n\n')
sonnet_lens = [len(sonnet) for sonnet in sonnets]
print('Average sonnet length: %.2f characters' % np.mean(sonnet_lens))# i.e, 608.94 characters in each sonnet
plt.figure(figsize=(15,10))
plt.bar([i for i in range(1, len(sonnets)+1)], sonnet_lens)
plt.title('Number of Characters per sonnet')
plt.ylabel('# Characters')
plt.xlabel('Sonnets')
plt.show()

length = len(text)

#100 seq_length means at each time the RNN would look at 60 stock prizes before time t and time t and based on this it will predict next output.
#Creating two entities X containing 100 previous characters  and Y containing character after that
seq_length = 40
# Sample new sequence every step characters ie, a,d,g from a,b,c,d,e,f,g gap of 2
step = 3
sentences = []
targets = []

#Number of unique characters
characters = sorted(list(set(text)))
# Dictionary mapping unique character to integer indices
char_to_n = {char:n for n, char in enumerate(characters)}
n_to_char = {n:char for n, char in enumerate(characters)}

# Loop through sonnets and create sequences and associated targets
for i in range(seq_length, length,step):
    seq = text[i-seq_length:i]
    lab = text[i]
    sentences.append([char_to_n[char] for char in seq])   # similar to sentences.append(text[i-seq_length:i]) but we convert to integer
    targets.append(char_to_n[lab])     #targets.append(text[i]) . Same as abve         
    


#Reshaping 2d to 3d (no.of rows, no.of columns ,len(characters))
x = np.zeros((len(sentences), seq_length, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char_to_n[char]] = 1
    y[i, char_to_n[targets[i]]] = 1

'''The model will output a probability value for each character possible. 
Instead of choosing the character with the highest probability, we will reweight the probabilities 
and sample from them based on a "temperature" value. 
The higher the temperature the more likely a random character will be chosen, 
the lower the temperature the more deterministic the model will behave.'''

def sample(preds, temperature=1.0):
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#BUILDING THE RNN
    
#For Project
#Initializing RNN
model = Sequential()

#Adding first layer and Dropout regularization
model.add(LSTM(units = 128, return_sequences = False, input_shape =  (x.shape[1], 1)))#return =True
model.add(Dropout(0.2))

#Adding Output Layer
model.add(Dense(units = y.shape[1], activation='softmax'))

#Compiling RNN
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')


'''Now the model will be trained on the text and targets. 
After training for 1 epoch, a random sequence will be chosen from the training corpus 
and fed into the model. Using this "seed text" we will predict the next 600 characters 
at different temperatures and store them in different text files locally.'''

epochs = 60

loss = []  # Custom history list to save model's loss
import os
# Create directory to store generated text
base_dir = 'generated_text'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

for epoch in range(1, epochs+1):
    print("Epoch", epoch)
    # Fit model for 1 epoch then generate text given a seed.
    history = model.fit(x, y, batch_size=128, epochs=1)
    loss.append(history.history['loss'][0])
    
    # Create directory to store text for each epoch
    epoch_dir = os.path.join(base_dir, 'epoch_' + str(epoch))
    if not os.path.isdir(epoch_dir):
        os.mkdir(epoch_dir)
    
    # Select a random seed text to feed into model and generate text(random seed means 1 sentence from original poem and continue to generate further lines for that sentence)
    start_idx = np.random.randint(0, len(text) - seq_length - 1)
    seed_text = text[start_idx:start_idx + seq_length]
    for temp in [0.2, 0.5, 1.0, 1.3]:
        generated_text = seed_text
        temp_file = 'epoch' + str(epoch) + '_temp' + str(temp) + '.txt'
        file = open(os.path.join(epoch_dir, temp_file), 'w')
        file.write(generated_text)
        
        # Predict and generate 600 characters (approx. 1 sonnet length)
        for i in range(600):
            # Vectorize generated text
            sampled = np.zeros((1, seq_length, len(characters)))
            for j, char in enumerate(generated_text):
                sampled[0, j, char_to_n[char]] = 1.
            
            # Predict next character
            preds = model.predict(sampled, verbose=0)[0]
            pred_idx = sample(preds, temperature=temp)
            next_char = characters[pred_idx]
            
            # Append predicted character to seed text
            generated_text += next_char
            generated_text = generated_text[1:]
            
            # Write to text file
            file.write(next_char)
        print('Temp ' + str(temp) + " done.")
        file.close()



#MAKING PREDICTIONS

# Predict and generate 600 characters (approx. 1 sonnet length)
import sys
def generate_sonnet(temp):
    '''Given a temperature, generate a new sonnet '''
    start_idx = np.random.randint(0, len(text) - seq_length - 1)
    new_sonnet = text[start_idx:start_idx + seq_length]
    sys.stdout.write(new_sonnet)
    for i in range(600):
        # Vectorize generated text
        sampled = np.zeros((1, seq_length, len(characters)))
        for j, char in enumerate(new_sonnet):
            sampled[0, j, char_to_n[char]] = 1.

        # Predict next character
        preds = model.predict(sampled, verbose=0)[0]
        pred_idx = sample(preds, temperature=temp)
        next_char = characters[pred_idx]

        # Append predicted character to seed text
        new_sonnet += next_char
        new_sonnet = new_sonnet[1:]

        # Print to console
        sys.stdout.write(next_char)
        sys.stdout.flush()
generate_sonnet(0.5)
'''Below the temperature is higher meaning that there will be more random guessing
 for the next character which explains why there are a lot of spelling mistakes and bizarre words'''
generate_sonnet(1)


# Save model
model.save('shakespeare_sonnet_model.h5')
# Load model
from keras.models import load_model
model = load_model('shakespeare_sonnet_model.h5')


#FOR BETTER RESULTS

'''
Train over longer epochs
Use Bidirectional LASTM
ADD more Layers
Add more Hidden Layers
Change maximum length of sequences'''
