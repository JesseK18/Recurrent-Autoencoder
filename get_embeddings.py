import torch, numpy as np
from graphs.models.recurrent_autoencoder import RecurrentAE
from datasets.ecg5000 import ECG500DataLoader
from utils.config import get_config_from_json

if __name__=="__main__":
    config, _ = get_config_from_json("configs/config_rnn_ae.json")
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")

    # rebuild model & load encoder weights
    model = RecurrentAE(config).to(device)
    state = torch.load("encoder_only1.pth", map_location=device)
    model.encoder.load_state_dict(state)
    model.encoder.eval()

    # pick loader (train/test/valid)
    loader = ECG500DataLoader(config).test_loader

    embs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.encoder(x)     # (1, B, D)
            h = h.squeeze(0).cpu()   # (B, D)
            embs.append(h)
            labels.append(y)
    embs   = torch.cat(embs,   0).numpy()
    labels = torch.cat(labels, 0).numpy()
    np.save("test_embeddings.npy", embs)
    np.save("test_labels.npy",     labels)
    print("Saved embeddings:", embs.shape)
    print(embs[0])