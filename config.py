import json


class Configuration:
    def __init__(self):
        with open('./config.json') as f:
            config = json.load(f)
        self.config = config
        self.name = config["name"]
        self.device = config["device"]
        self.maxlen = config["maxlen"]
        self.embed_dim = config["embed_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.teacher_forcing = config["teacher_forcing"]
        self.dropout = config["dropout"]
        self.lc_batch_size = config["learning_config"]["batch_size"]
        self.lc_epoch = config["learning_config"]["epoch"]
        self.lc_learning_rate = config["learning_config"]["learning_rate"]
        self.lc_optimizer = config["learning_config"]["optimizer"]
        self.save_path = config["save_path"]
        self.mode = config["mode"]

    def configprint(self):
        for i in self.config.keys():
            print(i, ":", self.config[i])
