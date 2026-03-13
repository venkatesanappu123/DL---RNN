# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock market prices change frequently due to various factors such as economic conditions, company performance, and market trends. Predicting future stock prices is challenging because the data is sequential and time-dependent. Traditional machine learning models often fail to capture these temporal patterns effectively.

Therefore, there is a need to develop a model that can learn from historical stock price data and identify patterns over time. A Recurrent Neural Network (RNN) is suitable for this task because it is designed to process sequential data and remember past information. The problem is to build an RNN model that uses historical closing price data to predict future stock prices with better accuracy.

<img width="733" height="801" alt="image" src="https://github.com/user-attachments/assets/6ee243ec-2c51-43e3-a735-bfa8e647839a" />

<img width="702" height="799" alt="image" src="https://github.com/user-attachments/assets/48176254-f31b-44b9-ba24-a9b7d1b6cdf5" />

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.


### STEP 3: 

Define the RNN model architecture



### STEP 4: 

Summarize, compile with loss and optimizer.



### STEP 5: 
Train the model with loss tracking.




### STEP 6: 
Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: VENKATESAN R

### Register Number: 212224230299

```python
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
      super(RNNModel, self).__init__()
      self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
      self.fc = nn.Linear(hidden_size,output_size)
    def forward(self, x):
      out,_=self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out

# Train the Model
def train_model(model, train_loader,criterion,optimizer,epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f"Epoch[{epoch+1}/{epochs}],Loss:{total_loss/len(train_loader):.4f}")
  print('Name:Rithika R ')
  print('Register Number:212224240136  ')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show() 
train_model(model,train_loader,criterion,optimizer) 


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/0adb3e5d-6e94-4cbb-9503-ed2e4ad9c5f7" />

## True Stock Price, Predicted Stock Price vs time

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/94715598-90b7-400f-bce5-ec4737e1d48d" />


### Predictions
<img width="314" height="68" alt="image" src="https://github.com/user-attachments/assets/e53c52e3-9e1a-4161-b9df-974c5af5e10e" />

## RESULT
This program has been executed succesfully.
