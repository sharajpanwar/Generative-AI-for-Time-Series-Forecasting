import torch
import inference
import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import transformer_timeseries as tst
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"



# Hyperparams
seed=10
epochs = 15
# test_size = 0.1
batch_size = 128
target_col_name = "sensor_read"
# timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

print ('program starting with epochs: ', str(epochs) )

## Params
dim_val = 256
n_heads = 4
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 48 # length of input given to decoder
enc_seq_len = 48 # length of input given to encoder
output_sequence_length = 6 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
forecast_window = 6
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 512 #check this 
in_features_decoder_linear_layer = 512
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

####################################################################################################################
# Read training and test data

training_data, test_data = utils.train_test_split(seed)

training_indices = utils.get_indices_entire_sequence(
    data=training_data, 
    window_size=window_size, 
    step_size=step_size)

# Making instance of custom training dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data[input_variables].values).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

# Making training dataloader
training_data = DataLoader(training_data, batch_size)
# training_data=training_data.to(device)
#########################################################################################################################
#  test data
test_indices = utils.get_indices_entire_sequence(
    data=test_data, 
    window_size=window_size, 
    step_size=step_size)

# Making instance of custom dataset class
test_data = ds.TransformerDataset(
    data=torch.tensor(test_data[input_variables].values).float(),
    indices=test_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

# Making dataloader
test_data = DataLoader(test_data, batch_size)
# test_data=test_data.to(device)

############################################################################################################################
#  create model

criterion = torch.nn.MSELoss()


model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
    )
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.cuda()
###################################
loss_list=[]
# Iterate over all epochs
for epoch in range(epochs):

    print("training epoch: ",epoch)

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(test_data): ## Making dataloader training_data = DataLoader(training_data, batch_size)
        if batch_first == False:
            shape_before = src.shape
            src = src.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(shape_before, src.shape))
            shape_before = tgt.shape
            tgt = tgt.permute(1, 0, 2)
            # print("tgt_y shape", tgt_y.shape)
            # print("tgt_y shape", tgt_y.shape)
            tgt_y=tgt_y.unsqueeze(2)
            shape_before = tgt_y.shape
            # print("tgt_y shape",shape_before)
            tgt_y = tgt_y.permute(1, 0, 2)
            # print("tgt_y shape changed from {} to {}".format(shape_before, tgt_y.shape))

        src=src.to(device)
        tgt=tgt.to(device)
        tgt_y=tgt_y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
            )
        tgt_mask=tgt_mask.to(device)

        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_seq_len
            )
        src_mask=src_mask.to(device)

        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss.backward()

        # Take optimizer step
        optimizer.step()
        # print("training batch is completed")
    
    print("training epoch is completed")


    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    with torch.no_grad():
    
        for i, (src, _, tgt_y) in enumerate(test_data):
            
            shape_before = src.shape
            src = src.permute(1, 0, 2)
            tgt_y=tgt_y.unsqueeze(2)
            shape_before = tgt_y.shape
            tgt_y = tgt_y.permute(1, 0, 2)
            
            src=src.to(device)
            tgt=tgt.to(device)
            tgt_y=tgt_y.to(device)


            prediction = inference.run_encoder_decoder_inference(
                model=model, 
                src=src, 
                forecast_window=forecast_window, batch_size=src.shape[1])

            loss = criterion(tgt_y, prediction)
          
            loss_list.append(loss.item())
            
            if i == len(test_data)-1:
                loss = loss.item()
                print(f"test loss: {loss:>7f} ")
            

print ('Training completed')

# record end time
training_end = time.time()

print("The Training time of above program is :", (training_end-start)/60, "minutes")
      
      
with torch.no_grad():
    pred_arr = []
    y_arr = []
    for i, (src, _, tgt_y) in enumerate(test_data):
        
        src = src.permute(1, 0, 2)
        tgt_y=tgt_y.unsqueeze(2)
        tgt_y = tgt_y.permute(1, 0, 2)
        
        src=src.to(device)
        tgt=tgt.to(device)
        tgt_y=tgt_y.to(device)

        prediction = inference.run_encoder_decoder_inference(
                model=model, 
                src=src, 
                forecast_window=forecast_window, batch_size=src.shape[1])        
        
        pred = prediction.permute(1, 0, 2)
        pred=np.squeeze(np.squeeze(pred.cpu().detach().numpy())[:,5:])
        pred_arr = pred_arr + list(pred)

        y = tgt_y.permute(1, 0, 2)
        y=np.squeeze(np.squeeze(y.cpu().detach().numpy())[:,5:])
        y_arr = y_arr + list(y)

        loss = criterion(tgt_y, prediction)

        if i == len(test_data)-1:
            loss = loss.item()
            print(f"test loss: {loss:>7f} ")
            
pred_arr=np.array(pred_arr) 
mean = np.nanmean(pred_arr)
inds = np.where(np.isnan(pred_arr))
pred_arr[inds] = mean


print(f"test RMSE final loss {math.sqrt(mean_squared_error(y_arr,pred_arr))}")

# record end time
end = time.time()

print("The test time of above program is :", (end-training_end)/60, "minutes")


plt.figure(figsize=(30,10))
plt.title('Ground Truth sensor measurement vs Forecast', fontsize=30)
plt.plot([i for i in range(len(y_arr))], y_arr, "-b", label="Ground Truth")
plt.plot([i for i in range(len(pred_arr))], pred_arr, "-r", label="Forecast")
plt.legend(loc="upper left",fontsize=20)
plt.savefig('/content/drive/MyDrive/sensor_proj/evaluation/Ground_Truth_Forecast_epoch_'+str(epochs)+'.png')
plt.show()
plt.close()


plt.figure(figsize=(30,10))
plt.title('Ground Truth sensor measurement vs Forecast', fontsize=30)
plt.plot([i for i in range(len(y_arr))], y_arr, "-b", label="Ground Truth")
# plt.plot([i for i in range(len(pred_arr))], pred_arr, "-r", label="Forecast")
plt.legend(loc="upper left",fontsize=20)
plt.savefig('/content/drive/MyDrive/sensor_proj/evaluation/Ground_Truth_epoch_'+str(epochs)+'.png')
plt.show()
plt.close()

plt.figure(figsize=(30,10))
plt.title('Ground Truth sensor measurement vs Forecast', fontsize=30)
# plt.plot([i for i in range(len(y_arr))], y_arr, "-b", label="Ground Truth")
plt.plot([i for i in range(len(pred_arr))], pred_arr, "-r", label="Forecast")
plt.legend(loc="upper left",fontsize=20)
plt.savefig('/content/drive/MyDrive/sensor_proj/evaluation/Forecast_epoch_'+str(epochs)+'.png')
plt.show()
plt.close()

plt.figure(figsize=(30,10))
plt.title('Ground Truth sensor measurement vs Forecast', fontsize=30)
# plt.plot([i for i in range(len(y_arr))], y_arr, "-b", label="Ground Truth")
plt.plot([i for i in range(len(loss_list))], loss_list, "-r", label="Loss curve")
plt.legend(loc="upper left",fontsize=20)
plt.savefig('/content/drive/MyDrive/sensor_proj/evaluation/Loss_curve_epoch_'+str(epochs)+'.png')
plt.show()
plt.close()

end = time.time()
print("The total time of above program is :", (end-start)/60, "minutes")

file = open('/content/drive/MyDrive/sensor_proj/evaluation/out_epoch_'+str(epochs)+'.txt', 'w')
file.write('RMSE: '+ str(math.sqrt(mean_squared_error(y_arr,pred_arr))))
file.close()

from google.colab import runtime
runtime.unassign()