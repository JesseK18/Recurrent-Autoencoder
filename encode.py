import torch, numpy as np
from torch import nn, optim
from graphs.models.recurrent_autoencoder import RecurrentAE
from datasets.ecg5000 import ECG500DataLoader, UCRDataLoader
from utils.config import get_config_from_json

def get_embeddings(model, loader, device):
    embs, labels = [], []
    model.encoder.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.encoder(x)       # (1, B, D)
            h = h.squeeze(0).cpu()     # (B, D)
            embs.append(h)
            labels.append(y)
    embs   = torch.cat(embs,   0).numpy()
    labels = torch.cat(labels, 0).numpy()
    return embs, labels

def main():
    # 1) load config
    config, _ = get_config_from_json("configs/config_rnn_ae.json")
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")

    # 2) build model & freeze decoder
    model = RecurrentAE(config).to(device)
    for p in model.decoder.parameters():
        p.requires_grad = False

    # 3) optimizer & loss
    optimizer = optim.Adam(model.encoder.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # 4) data loaders
    dl = UCRDataLoader(config)
    train_loader = dl.train_loader
    val_loader   = dl.valid_loader
    test_loader  = dl.test_loader

    # 5) training loop (encoder only)
    model.train()
    for epoch in range(config.max_epoch):
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss  = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{config.max_epoch}  loss: {epoch_loss/len(train_loader):.4f}")

    # 6) extract embeddings
    train_embs, train_labels = get_embeddings(model, train_loader, device)
    val_embs,   val_labels   = get_embeddings(model, val_loader,   device)
    test_embs,  test_labels  = get_embeddings(model, test_loader,  device)

    # 7) save embeddings
    np.save("train_embeddings.npy", train_embs)
    np.save("train_labels.npy",     train_labels)
    np.save("val_embeddings.npy",   val_embs)
    np.save("val_labels.npy",       val_labels)
    np.save("test_embeddings.npy",  test_embs)
    np.save("test_labels.npy",      test_labels)
    print("Saved embeddings:", 
          train_embs.shape, val_embs.shape, test_embs.shape)

    # 8) save encoder weights
    torch.save(model.encoder.state_dict(), "encoder_final.pth")
    print("Encoder saved as encoder_final.pth")

if __name__ == "__main__":
    main()