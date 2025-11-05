import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
import torch

def plot_predictions(true_values, pred_values, num_sequences=5, title='True vs Predicted Useful_Time'):
    maes = [mean_absolute_error(true_values[i], pred_values[i]) for i in range(len(true_values))]
    k=4
    # k = len(true_values)

    top_k_indices = np.argsort(maes)[:k]

    for rank, idx in enumerate(top_k_indices):

        pred_mae = [mean_absolute_error(true_values[idx][seq], pred_values[idx][seq]) for seq in range(len(true_values[idx]))]

        top_one = np.argsort(pred_mae)[0]

        plt.figure(figsize=(12, 6))
        time_steps = np.arange(len(true_values[idx][top_one]))

        plt.plot(time_steps, true_values[idx][top_one], marker='o', linestyle='-', label=f'Sequence {idx+1} (True)', color='C0')
        plt.plot(time_steps, pred_values[idx][top_one], marker='x', linestyle='--', label=f'Sequence {idx+1} (Predicted)', color='C1', alpha=0.7)

        plt.xlabel('Time Step (Future Timestamps)')
        plt.ylabel('Future Useful Time')
        plt.title(f'{title}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        plt.show()
        # plt.savefig('Prediction_plot.png')

# For the model tester the original data is needed and the testing data can be seperate

class ModelTester:
  def __init__(self, model_path, data_df, window_size, prediction_length, batch_size, device):
    self.model_path = model_path
    self.data = data_df
    self.window_size = window_size
    self.prediction_length = prediction_length
    self.batch_size = batch_size
    self.device = device

  def _get_data_(self, flag, scaler):
    data_set = InfusionPumpDataset(data=self.data, flag = flag, window_size=self.window_size, prediction_length=self.prediction_length,
                                 target_col='Useful_Time', feature_cols=Config.feature_cols, batch_size=Config.batch_size, scaler=scaler)
    data_loader = DataLoader(data_set,
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=4,
                            shuffle=False,
                            drop_last=True)

    return data_set, data_loader

  def _inverse_target_transform(self, scaler, dataset, target_index=-1):

    n_samples, pred_len = dataset.shape
    n_features = scaler.mean_.shape[0]

    dummy = np.zeros((n_samples * pred_len, n_features))
    dummy[:, target_index] = dataset.flatten()

    inv = scaler.inverse_transform(dummy)

    return inv[:, target_index].reshape(n_samples, pred_len)

  def predict_data(self, flag):
      device = torch.device('cuda')
      # change
      all_indices = []

      model = PCUTransformer(
          feature_size = len(Config.feature_cols),
          hidden_dim=Config.hidden_dim,
          num_layers=Config.num_layers,
          nheads=Config.nhead,
          dropout=Config.dropout,
          prediction_length=self.prediction_length
      ).to(device)

      state_dict = torch.load(self.model_path)
      print("Trained keys", state_dict.keys())
      print("Initial keys",model.state_dict().keys())
      model.load_state_dict(torch.load(self.model_path))
      model.eval()

      train_data, train_loader = self._get_data_('train', None)
      scaler = train_data.scaler
      val_data, val_loader = self._get_data_('val', scaler)

      trues = []
      preds = []

      for i, (batch_x, batch_y, batch_indices) in enumerate(val_loader):
        with torch.no_grad():
          batch_x = batch_x.to(device).float()
          batch_y = batch_y.to(device).float()
          output = model(batch_x)
          batch_y_np = batch_y.detach().cpu().numpy()
          output_np = output.detach().cpu().numpy()

          batch_y_inv = self._inverse_target_transform(val_data.scaler, batch_y_np)
          output_inv = self._inverse_target_transform(val_data.scaler, output_np)
          trues.append(batch_y_inv)
          preds.append(output_inv)
      print(len(trues))
      return trues, preds, all_indices
  

infusion_pumps = pd.read_csv("/content/sample_data/time_processed.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predictor = ModelTester(model_path='/content/model_weights.pth', data_df=infusion_pumps, window_size=250, prediction_length=Config.prediction_length, batch_size=32, device=device)

trues, preds, all_indices = predictor.predict_data('val')
plot_predictions(
    true_values=trues,
    pred_values=preds,
)

