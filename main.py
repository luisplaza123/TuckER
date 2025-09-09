from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import pandas as pd
import os # <-- AÑADIR IMPORTACIÓN DE OS

class Experiment:

    # --- MODIFICACIÓN 1: Añadir 'output_dir' al constructor ---
    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., output_dir='results'): # <-- Nuevo parámetro
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.output_dir = output_dir # <-- Guardar el directorio de salida
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        self.device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")
        self.history = []

    # ... (los métodos get_data_idxs, get_er_vocab, get_batch, evaluate no cambian) ...
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.to(self.device)
        return np.array(batch), targets
    
    def evaluate(self, model, data):
        hits = [[] for _ in range(10)]
        ranks = []
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        print("Number of data points: %d" % len(test_data_idxs))
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0]).to(self.device)
            r_idx = torch.tensor(data_batch[:,1]).to(self.device)
            e2_idx = torch.tensor(data_batch[:,2]).to(self.device)
            predictions = model.forward(e1_idx, r_idx)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        metrics = {'hits@10': np.mean(hits[9]), 'hits@3': np.mean(hits[2]), 'hits@1': np.mean(hits[0]), 'mr': np.mean(ranks), 'mrr': np.mean(1./np.array(ranks))}
        print('Hits @10: {0}'.format(metrics['hits@10'])); print('Hits @3: {0}'.format(metrics['hits@3'])); print('Hits @1: {0}'.format(metrics['hits@1'])); print('Mean rank: {0}'.format(metrics['mr'])); print('Mean reciprocal rank: {0}'.format(metrics['mrr']))
        return metrics


    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.to(self.device)
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)
        
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        
        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0]).to(self.device)
                r_idx = torch.tensor(data_batch[:,1]).to(self.device)
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            
            epoch_results = {'epoch': it, 'loss': np.mean(losses)}
            print(f"\nEpoch: {it}, Time: {time.time()-start_train:.4f}, Loss: {epoch_results['loss']:.4f}")
            
            model.eval()
            with torch.no_grad():
                print("Validation:")
                val_metrics = self.evaluate(model, d.valid_data)
                for key, value in val_metrics.items():
                    epoch_results['val_' + key] = value
                
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    test_metrics = self.evaluate(model, d.test_data)
                    for key, value in test_metrics.items():
                        epoch_results['test_' + key] = value
                    print(f"Test Time: {time.time()-start_test:.4f}")
            self.history.append(epoch_results)
        
        # --- MODIFICACIÓN 2: Usar el directorio y nombre de archivo dinámicos ---
        print("\nEntrenamiento finalizado. Guardando el modelo y las métricas...")
        
        # Crear el directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Construir las rutas de archivo
        model_path = os.path.join(self.output_dir, f"{os.path.basename(self.output_dir)}.pt")
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        
        torch.save(model.state_dict(), model_path)
        print(f"Modelo guardado exitosamente en '{model_path}'")
        
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(metrics_path, index=False)
        print(f"Métricas guardadas exitosamente en '{metrics_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (argumentos anteriores)
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?")
    parser.add_argument("--edim", type=int, default=200, nargs="?")
    parser.add_argument("--rdim", type=int, default=200, nargs="?")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?")

    # --- MODIFICACIÓN 3: Añadir un nuevo argumento para el nombre de salida ---
    parser.add_argument("--output_prefix", type=str, default="my_experiment", nargs="?",
                    help="Prefijo para la carpeta y el nombre del modelo guardado.")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    
    # --- MODIFICACIÓN 4: Crear la ruta de salida a partir del argumento ---
    output_dir = os.path.join("results", args.output_prefix)

    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    
    # Pasar el directorio de salida al crear el objeto Experiment
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
                            output_dir=output_dir) # <-- Pasar la nueva ruta
    experiment.train_and_eval()