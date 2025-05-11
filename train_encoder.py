import torch
from torch import nn, optim
from graphs.models.recurrent_autoencoder import RecurrentAE
from datasets.ecg5000 import ECG500DataLoader
from utils.config import get_config_from_json

if __name__=="__main__":
    # 1) load config
    config, _ = get_config_from_json("configs/config_rnn_ae.json")
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")

    # 2) build model & freeze decoder
    model = RecurrentAE(config).to(device)
    for p in model.decoder.parameters():
        p.requires_grad = False

    # 3) optimizer only on encoder
    optimizer = optim.Adam(model.encoder.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # 4) data
    loader = ECG500DataLoader(config).train_loader

    # 5) train loop
    model.train()
    for epoch in range(config.max_epoch):
        epoch_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            # forward through full AE
            x_hat = model(x)
            loss  = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss: {epoch_loss/len(loader):.4f}")

    # 6) save encoder weights
    torch.save(model.encoder.state_dict(), "encoder_only1.pth")
    print("Encoder trained & saved.")