import numpy as np
import inspect
from agents.rnn_autoencoder import RecurrentAEAgent
from utils.config import get_config_from_json

if __name__ == "__main__":
    config, config_dict = get_config_from_json("configs/config_rnn_ae.json")
    #print(type(config))
    print(config.agent)
    agent_class = globals()[config.agent]
    
    agent = agent_class(config)
    #print(agent)
    # load your saved/best model
    agent.load_checkpoint(config.checkpoint_file or "/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/experiments/checkpoints/rnn_ae_ECG5000_exp_0/")
    # pick the loader you want (e.g. test_loader or valid_loader)
    print(agent.data_loader)
    #test_loader = agent.data_loader
    # print("Type:", type(agent.data_loader))
    # print("Repr:", agent.data_loader)   
    # print(dir(agent.data_loader))
    # print("====================")
    # print(inspect.getmembers(agent.data_loader, lambda x: not inspect.isroutine(x)))
    train_loader = agent.data_loader.train_loader
    #test_loader = agent.data_loader.test_loader
    print(train_loader)
    print("dataset:", train_loader.dataset)
    print("batch_size:", train_loader.batch_size)
    
    batch = next(iter(train_loader))
    print(batch)                # raw contents (e.g. (inputs, labels))
    print(type(batch))
    # if it’s a tuple of (X, y):
    print(batch[0].shape, batch[1].shape)
    
    
    embs, labels = agent.extract_embeddings(train_loader)
    # # save to disk
    # np.save("test_embeddings.npy", embs)
    # np.save("test_labels.npy", labels)
    # print(f"embeddings: {embs.shape}, labels: {labels.shape}")
    
    