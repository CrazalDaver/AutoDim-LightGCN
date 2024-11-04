import torch

config = {}

config['batch_size'] = 1000
config['latent_dim'] = 64
config['n_layers']= 3
# config['dropout'] = False
# config['keep_prob']  = 0.6
config['n_fold'] = 100
config['test_batch_size'] = 100
# config['multicore'] = False
config['lr'] = 1e-3
config['decay'] = 1e-4
# config['pretrain'] = False
# config['A_split'] = False
# config['bigdata'] = False
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['top_k'] = [1, 5, 10, 20]
