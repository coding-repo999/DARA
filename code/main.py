import argparse, pickle, time
import torch as th
from utils.preprocess import *
from utils.hetero_gnn import HeteroRGNN
from utils.gnn_module import GAT, GCN, GNN
from utils.align import *
from utils.mask import *
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, average_precision_score
import csv
# import dgl.function as fn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


import warnings
import argparse, pickle, time
import sys
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def write_node_embeddings( h, lid,rid, mode,labels):
    h = h.detach().numpy()
    print("Export node embeddings...")
    if mode == 'concat':
        h = np.hstack((h[lid], h[rid]))
    elif mode == 'diff':
        h = h[lid] - h[rid]
    np.savetxt(os.path.join(args.dp.replace('../', './'),'feature.txt'), h)
    np.savetxt(os.path.join(args.dp.replace('../', './'),'label.txt'), labels)

def validate(align_model, g, kg, val_data, epoch, use_gpu):
    # align_model.eval() 
    with torch.no_grad():
        val_batch = val_data
        if not len(val_batch): return  False
        lid, rid = get_embedding_pair(kg, val_batch)
        lid = torch.tensor(lid, dtype=torch.long)
        rid = torch.tensor(rid, dtype=torch.long)
        # print('left:', lid, 'right:', rid)
        # logits = align_model(initial_emb(g.ndata['x']), lid, rid)
        # Move tensors to device if using GPU
        if use_gpu:
            device = next(align_model.parameters()).device  # Get device from model parameters
            lid = lid.to(device)
            rid = rid.to(device)
        
        # Forward pass through the model
        logits = align_model(g, lid, rid)
        
        # Apply sigmoid to get probabilities and move to CPU for NumPy operations
       
        if use_gpu:
            logits = align_model(g, lid, rid).cpu()
        else:
            logits = align_model(g, lid, rid)
        # output = align_model(g.ndata['x'], val_batch[:, 0], val_batch[:, 1])
        # print("batch:{}, output shape:{}".format(b,output.shape))

        # loss = loss_fcn(logits,train_data[:,2])

        # prediction =  output.argmax(dim=1)
        pred_scores = torch.sigmoid(logits).numpy().flatten()  # Convert to 1D NumPy array of probabilities
        
        # Extract true labels and convert to 1D NumPy array
        labels = val_batch[:, 2].astype(np.float32)  # Ensure labels are float32 for compatibility
        
        # Initialize variables to track the best metrics
        best_f1, best_roc, precision, recall = 0, 0, 0, 0
        best_pr = 0
        best_th = 0.5  # Initial threshold
        
        # Iterate over potential thresholds to find the best ROC AUC score
        for i in range(3, 10):
            for j in range(9):
                crr_th = 0.1 * i + 0.01 * j  # Current threshold
                
                # Generate binary predictions based on the current threshold
                pred = pred_scores > crr_th
                
                # Calculate ROC AUC and Average Precision (PRAUC) scores
                # print("predict labels", labels)
                roc_score = roc_auc_score(labels, pred_scores)
                
                prauc = average_precision_score(labels, pred_scores, average='weighted')
                
                # Update best metrics if current ROC AUC is better
                if roc_score > best_roc:
                    best_roc = roc_score
                    best_pr = prauc
                    precision = precision_score(labels, pred)
                    recall = recall_score(labels, pred)
                    best_f1 = f1_score(labels, pred)
                    best_th = crr_th
        
        # Print the best metrics found during validation
        if precision and recall:
            print(f'Validation -- Epoch: {epoch} | Best ROCAUC: {best_roc:.4f} | '
                  f'F1: {best_f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | '
                  f'PRAUC: {best_pr:.4f} | Threshold: {best_th:.2f}')
        
        return best_roc, best_th



def test(align_model, g, kg, test_data, threshold, use_gpu):
    # align_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_batch = test_data
        if not len(test_batch):
            print("No test data available.")
            return False  # Early exit if test data is empty

        # Extract embedding pairs
        lid_np, rid_np = get_embedding_pair(kg, test_batch)
        
        # Convert lid and rid to PyTorch tensors with dtype=torch.long
        lid = torch.tensor(lid_np, dtype=torch.long)
        rid = torch.tensor(rid_np, dtype=torch.long)
        
        # Move tensors to device if using GPU
        if use_gpu:
            device = next(align_model.parameters()).device  # Get device from model parameters
            lid = lid.to(device)
            rid = rid.to(device)
        
        # Forward pass through the model
        logits = align_model(g, lid, rid)
        
        # Apply sigmoid to get probabilities and move to CPU for NumPy operations
        if use_gpu:
            logits = logits.cpu()
        pred_scores = torch.sigmoid(logits).numpy().flatten()  # 1D NumPy array of probabilities
        
        # Extract true labels and convert to 1D NumPy array of float32
        labels = test_batch[:, 2].astype(np.float32)  # Ensure labels are float32 for compatibility
        
        # Generate binary predictions based on the given threshold
        pred = pred_scores > threshold
        
        print(f"Threshold from validation: {threshold}")
        
        # Calculate metrics
        f1 = f1_score(labels, pred)
        micro_f1 = f1_score(labels, pred, average='micro')
        macro_f1 = f1_score(labels, pred, average='macro')
        acc = accuracy_score(labels, pred)
        precision = precision_score(labels, pred)
        recall = recall_score(labels, pred)
        
        print(
            f'Test --  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, '
            f'Micro F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}, Accuracy: {acc:.4f}'
        )
        
        # Calculate PRAUC and ROCAUC
        prauc = average_precision_score(labels, pred_scores, average='weighted')
        roc_auc = roc_auc_score(labels, pred_scores)
        print(f"PRAUC: {prauc:.4f}, ROCAUC: {roc_auc:.4f}")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(labels, pred))
        
        # Print flattened predictions and labels
        # print("Predicted Scores:", pred_scores.tolist())
        # print("Predictions:", pred.tolist())
        # print("Labels:", labels)
        # Convert lists to a DataFrame
        # Assuming pred_scores, pred, and labels are numpy arrays and have been converted to lists as needed
        pred_scores_list = pred_scores.tolist()
        pred_list = pred.tolist()
        labels_list = labels.tolist()

        # Write to CSV
        with open('predictions659.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Predicted Scores', 'Predictions', 'Labels'])  # Writing the headers
            for score, prediction, label in zip(pred_scores_list, pred_list, labels_list):
                writer.writerow([score, prediction, label])
                
        # Iterate over different thresholds to find the best Precision at Recall >= 0.9
        pre_at_best_rec, rec, th1, f1, pr = 0, 0, 0, 0, 0
        for i in range(70):
            th_ = 0.9 - 0.01 * i  # Threshold from 0.7 down to 0.41
            pred_temp = pred_scores > th_
            precision_temp = precision_score(labels, pred_temp)
            recall_temp = recall_score(labels, pred_temp)
            f1_temp = f1_score(labels, pred_temp) 
            pr_temp = average_precision_score(labels, pred_scores, average='weighted')  # Use scores, not binary
            
            print(f'Threshold: {th_:.4f}, Precision: {precision_temp:.4f}, Recall: {recall_temp:.4f}, F1: {f1_temp:.4f}, PRAUC: {pr_temp:.4f}')
            
            if recall_temp >= 0.95:
                if precision_temp > pre_at_best_rec:
                    pre_at_best_rec, rec, th1, f1, pr = precision_temp, recall_temp, th_, f1_temp, pr_temp
        
        if rec:
            print(f"Pre@Rec=0.95: Precision={pre_at_best_rec:.4f}, F1: {f1:.4f}, PRAUC: {pr_temp:.4f}, Recall={rec:.4f}, Threshold={th1:.2f}")



def get_graph(hetero, kg, with_mask, datapath):
    """
    Constructs the DGL graph based on the knowledge graph (KG) parameters.

    Parameters:
    - hetero (bool): Whether to build a heterogeneous graph.
    - kg (KG): The knowledge graph object.
    - with_mask (bool): Whether to generate a mirror graph.
    - datapath (str): Path to data files.

    Returns:
    - g (dgl.DGLGraph): The constructed graph.
    - g_mirror (list or np.ndarray): The mirror graph or candidates.
    """
    print(" The graph is hetero: ", hetero)

    if not hetero:
        # Build homogeneous graph on CPU
        g = kg.buildGraph()  # Ensure buildGraph assigns features on CPU
        g_mirror = []
        # print('Build Graph with {} nodes and {}-dim features...'.format(g.num_nodes(), kg.features.shape[1]))
        
        if with_mask:
            g_mirror = get_edge_mirror(g, kg)  # Ensure get_edge_mirror is device-agnostic
            # g_mirror remains on CPU; main.py will move it to GPU if needed
        
        return g, g_mirror

    else:
        # Build heterograph on CPU
        g = kg.buildHeteroGraph()  # Ensure buildHeteroGraph assigns features on CPU
        print("Total nodes:", len(kg.entity_list))
        num_of_nodes, num_of_edges = 0, 0
        for ntype in g.ntypes:
            num_of_nodes += g.num_nodes(ntype)
        for etype in g.etypes:
            num_of_edges += g.num_edges(etype)
        # print('Build Heterograph with {} nodes and {}-dim features...'.format(g.num_nodes(), kg.features.shape[1]))

        # Assign degree-related attributes on CPU
        attribute_degree(g, kg, 'vul', datapath)  # Ensure attribute_degree assigns on CPU

        # Generate or load candidates on CPU
        g_candidates = get_candidates(g, kg, 'vul', datapath)  # Ensure get_candidates is device-agnostic

        return g, g_candidates



def get_gnn(kg, g, g_mirror, gnn_type, hdim,use_gpu):
    gnn = None
    if len(g.ntypes) == 1:
        feature_matrix = g.ndata['x']
        gnn =  GNN(g, kg, g_mirror,gnn_type, in_dim=feature_matrix.shape[1], hidden_dim = hdim, out_dim=hdim,
                      multihead = args.mh, num_heads=args.nh, mask = args.mask, learnable=args.learnable,num_layers = args.nl)

    elif len(g.ntypes) > 1:
        print("Initialize GNN for Heterogeneous Graph!")
        feature_matrix = g.nodes['vul'].data['x']
        gnn = HeteroRGNN(g, kg,args.agg, args.mask, gnn_type, use_gpu,in_size=feature_matrix.shape[1], hidden_size = hdim, out_size=hdim)
    return gnn

def get_embedding_pair( kg, batch):
    if not args.heterogeneous:
        lid, rid = get_graph_id(kg, batch)
        return ( lid, rid)
    else:
        # ids on homo graph
        lid_, rid_ = get_graph_id(kg, batch)

        lid, rid = kg.id_in_type[lid_], kg.id_in_type[rid_]
        return (lid, rid)


def get_subgraph(g, lid, rid, entype):
    # print('vul id on graph 1',lid)
    # print('vul id on graph 2',rid)
    en_ids = np.concatenate([lid, rid])
    subg_dict = {entype: torch.tensor(en_ids)}
    # print(subg_dict)
    for srctype, etype, dsttype in g.canonical_etypes:
        if 'of' in etype:
            # subg_dict[srctype] = []
            for eid in en_ids:
                attr_nodes = g.predecessors(eid, etype)
                if srctype not in subg_dict.keys(): subg_dict[srctype]= attr_nodes
                else:
                    # print(subg_dict[srctype],g.predecessors(eid, etype))
                    subg_dict[srctype] = torch.cat([subg_dict[srctype],attr_nodes])
                for aid in attr_nodes:
                    subg_dict[entype] = torch.cat([subg_dict[srctype], g.successors(aid, etype = srctype+'_of')])
            # print('&&:', srctype, subg_dict[srctype])
    subg = g.subgraph(subg_dict)
    # print(subg.nodes[entype].data[dgl.NID])
    return subg

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        return self.alpha * self.bce(x, y) + (1 - self.alpha) * sce_loss(x, y)

def main(args):
    # 创建日志目录
    # dataset_name = os.path.basename(os.path.normpath(args.dp))
    # # log_dir = os.path.join('log', dataset_name)
    #
    # os.makedirs(log_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.dp))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root_dir, 'log', dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    current_time = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{current_time}.txt')

    # 保存参数设置
    with open(log_file, 'w') as f:
        f.write('Parameters:\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
        f.write('\n')

    # 重定向标准输出到文件
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)

    if args.gpu >= 0 and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{args.gpu}')
    else:
        use_gpu = False
        device = torch.device('cpu')

    print("Use GPU?", use_gpu)
    print(f"Target device: {device}")

    kg = KG(args.dp.replace('../', './'),args.tp, args.cp, args.text_representation)
    # list of entities, relations, and types


    # 读取原始train8.csv并复制一份为newtrain8.csv
    # train_df = pd.read_csv(os.path.join(args.dp.replace('../', './'),'train.csv'))
    train_df = pd.read_csv(os.path.join(args.dp.replace('../', './'), 'train.csv'))
    new_train_file='newtrain8.csv'

    # 新的训练文件命名
    train_df.to_csv(os.path.join(args.dp.replace('../', './'), new_train_file), index=False)

    # 获取label=1的数据
    positive_samples = train_df[train_df['label'] == 1].copy()
    positive_samples = positive_samples.sample(frac=0.75, random_state=4)

    # print(positive_samples)

    # 修改tableA
    tableA = pd.read_csv(os.path.join(args.dp.replace('../', './'), 'tableA.csv'))
    new_tableA = tableA.copy()
    for ltable_id in positive_samples['ltable_id']:
        vul_rows = tableA[tableA['vul'] == ltable_id].copy()
        vul_rows['vul'] = vul_rows['vul'] + 100000
        # 修改impact列，每隔十个单词去除一个
        # vul_rows['impact'] = vul_rows['impact'].apply(
        #     lambda x: ' '.join([word for i, word in enumerate(x.split()) if (i + 1) % 15 != 0]))
        new_tableA = pd.concat([new_tableA, vul_rows])
    new_tableA.to_csv(os.path.join(args.dp.replace('../', './'), 'newtableA.csv'), index=False)

    # 修改tableB
    tableB = pd.read_csv(os.path.join(args.dp.replace('../', './'),'tableB.csv'))
    new_tableB = tableB.copy()
    for rtable_id in positive_samples['rtable_id']:
        vul_rows = tableB[tableB['vul'] == rtable_id].copy()
        vul_rows['vul'] = vul_rows['vul'].apply(lambda x: int((x // 100000 + 100000) * 100000))
        new_tableB = pd.concat([new_tableB, vul_rows])
    new_tableB.to_csv(os.path.join(args.dp.replace('../', './'),'newtableB.csv'), index=False)

    # 修改ltable_id和rtable_id
    new_positive_samples = positive_samples.copy()
    new_positive_samples['ltable_id'] = new_positive_samples['ltable_id'] + 100000
    new_positive_samples['rtable_id'] = new_positive_samples['rtable_id'].apply(
        lambda x: int((x // 100000 + 100000) * 100000))
    # 将修改后的数据追加到newtrain8.csv
    new_positive_samples.to_csv(os.path.join(args.dp.replace('../', './'),new_train_file), mode='a', header=False, index=False)

    # train_data = load_data(args.dp,'newtrain8.csv')

    # train_data = load_data(args.dp.replace('../', './'),'train8.csv')
    args.a = 'newtableA'
    args.b = 'newtableB'

    kg.buildKG([args.a, args.b])
    # print("Successfully build graph A:",kg.graph_a.ndata)
    with open('./check/id2idg.json','w') as f:
        json.dump(kg.id2idg, f)

    gnn_type = args.gnn
    hdim = args.hdim
    # 4. Build the heterogeneous graph and its mirror, on CPU
    g, g_mirror = get_graph(args.heterogeneous, kg, args.mask, args.dp.replace('../', './'),)
    kg.candidates = g_mirror

    # 5. Debugging: Check the type of g_mirror
    # print(f"Type of g_mirror: {type(g_mirror)}")
    # if isinstance(g_mirror, list) and len(g_mirror) > 0:
        # print(f"Length of g_mirror: {len(g_mirror)}")
        # print(f"Type of first element in g_mirror: {type(g_mirror[0])}")

    # 6. Move the entire graph and mirror graph to the target device
    if use_gpu:
        # print(f"Moving main graph to {device}...")
        g = g.to(device)
        # print(f"Main graph moved to {device} successfully.")

        # print(f"Handling mirror graph...")
        if isinstance(g_mirror, dgl.DGLGraph):
            g_mirror = g_mirror.to(device)
            # print(f"Mirror graph moved to {device} successfully.")
        elif isinstance(g_mirror, list):
            # Assuming g_mirror is a list of lists, do not move to device
            print("g_mirror is a list of lists; keeping it on CPU.")
            # No action needed
        else:
            print(f"g_mirror is of unexpected type: {type(g_mirror)}")
            # Handle other types if necessary
        # print("Mirror graph handling complete.")

    # 7. Initialize the GNN module
    GNN = get_gnn(kg, g, g_mirror, gnn_type, hdim, use_gpu)

    dur = []
    # initial_emb = EmbeddingLayer(kg, args.de)
    # print("Initiate Align Model...")
    align_model = AlignNet(g,kg,in_dim = hdim, h_dim = int(hdim/2), mode = args.mode ,gnn = GNN)
    # print(align_model)
    train_data = load_data(args.dp.replace('../', './'),new_train_file)
    test_data = load_data(args.dp.replace('../', './'),'test.csv')
    val_data =  load_data(args.dp.replace('../', './'),'valid.csv')
    optimizer = torch.optim.SGD(align_model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    # print(align_model.parameters())
    # criteria = 0.8*nn.BCEWithLogitsLoss() + 0.2*sce_loss()


    criteria = CombinedLoss(alpha=0.95)
    # criteria =nn.BCEWithLogitsLoss()



    if use_gpu:
        GNN = GNN.to(device)
        align_model = align_model.to(device)
        criteria = criteria.to(device)
        # print(f"Models and loss function moved to {device} successfully.")


    dist_opt =  HingeDistLoss()
    if use_gpu:
        dist_opt = dist_opt.to(device)

    # criteria = torch.nn.CrossEntropyLoss()
    batch_size = args.batch_size
    avg_pre, avg_rec = 0, 0
    MULTI_TEST = False

    '''Train Align Model'''
    # print("Train Graph Embedding...")
    BEST_ROC = 0
    th = 0.5
    for epoch in range(args.n_epochs):
        # if epoch: break
        t0 = time.time()
        align_model.train()
        train_loss = 0
        # GNN.get_weight()
        for b in tqdm(range(int(len(train_data)/batch_size))):

            batch_id = np.arange(b* batch_size,(b+1)*batch_size)
            train_batch = train_data[batch_id]
            '''Balance Train Data'''
            pos_samples= np.array([sample for i,sample in enumerate(list(train_batch)) if sample[2]==1])
            pos_samples_bid = np.array([i for i,sample in enumerate(list(train_batch)) if sample[2]==1])
            if len(pos_samples):
                for i in range(int((len(train_batch)-len(pos_samples))/(args.scale*len(pos_samples)))):
                    train_batch = np.vstack([train_batch, pos_samples])
                    batch_id = np.concatenate([batch_id, pos_samples_bid])
            '''------------'''

            lid, rid = get_embedding_pair(kg, train_batch)
            lid = torch.tensor(lid, dtype=torch.long)  # Assuming node IDs are long integers
            rid = torch.tensor(rid, dtype=torch.long) 
            if use_gpu:
                lid = lid.to(device)
                rid = rid.to(device)

            logits = align_model(g, lid, rid)
            # print("Type of train_batch:", type(train_batch))
            # print("Shape of train_batch:", train_batch.shape)
            # print("Data type of train_batch:", train_batch.dtype)
            # print("Sample data from train_batch[:, 2]:", train_batch[:5, 2])
            # print("Data type of train_batch[:, 2]:", train_batch[:, 2].dtype)

            # labels = torch.unsqueeze(torch.tensor(train_batch[:, 2]).float(), 1)
            labels = torch.unsqueeze(torch.tensor(train_batch[:, 2], dtype=torch.float32), 1)


            if use_gpu:
                labels = labels.to(device)

            loss = criteria(logits, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += loss

        dur.append(time.time() - t0)

        # lid, rid = get_graph_id(kg, train_data)
        lid, rid = get_embedding_pair(kg, train_data)
        lid = torch.tensor(lid, dtype=torch.long)  # Assuming node IDs are long integers
        rid = torch.tensor(rid, dtype=torch.long) 
        labels = train_data[:, 2]
        # logits = align_model(initial_emb(g.ndata['x']), lid, rid)
        if use_gpu:
            lid = lid.to(device)
            rid = rid.to(device)
            logits = align_model(g, lid, rid).cpu()
        else:
            logits = align_model(g, lid, rid)
        pred = torch.sigmoid(logits).detach().numpy() > 0.5
        acc = accuracy_score(labels, pred)
        f1, micro_f1, macro_f1 = f1_score(labels, pred), f1_score(labels, pred,average = 'micro'), f1_score(labels, pred,average = 'macro')

        # print("Train lables  (first ten) -- ", labels[:10])
        # print("Train Logits before sigmoid(first ten) -- ", torch.flatten(logits).detach().numpy()[:10])
        # print("Train Logits after sigmoid(first ten) -- ", torch.flatten(torch.sigmoid(logits)).detach().numpy()[:10])
        # print(emb[lid[0]])
        # print(emb[rid[0]])
        print("Epoch {:05d} | Train Loss {:.4f} | Train Accuracy {:.4f} |F1 {:.4f} | Time(s) {:.4f}".format(
            epoch, train_loss.item(), acc,f1, np.mean(dur)))

        '''---Validation---'''

        if epoch % 2 == 0 :

            best_roc, best_th = validate(align_model, g, kg, val_data, epoch, use_gpu)
            if best_roc >= BEST_ROC:
                BEST_ROC = best_roc
                best_model, th = align_model, best_th


    '''---- Test ---'''
    align_model.eval()
    best_model_name = gnn_type + '.' + str(th) + '.pkl'
    # save best model selected by validation set
    if os.path.exists('./models/' + best_model_name):
        os.remove('./models/' + best_model_name)
    torch.save(best_model.state_dict(), './models/' + best_model_name)
    # Test phase
    align_model.load_state_dict(torch.load('./models/' + best_model_name))
    with torch.no_grad():
        test(align_model, g, kg, test_data, th, use_gpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MKGNN')
    parser.add_argument("--a", type = str, default = "tableA",help="source graph")
    parser.add_argument("--b", type = str, default = "tableB",help="target graph")

    parser.add_argument("--dp", type = str, default = "./data/sample/",help="data path")
    parser.add_argument("--tp", type = str, default = "./text/model.bin",help="path to text embedding model")
    parser.add_argument("--cp", type = str, default = "./text/corpus.txt",help="path to corpus for text embedding model")
    parser.add_argument("--mp", type = str, default = ".../models", help = "path to save trained models")
    # parser.add_argument("--de", type=int, default=0, help="dimension of embedding trained by the first layer")
    parser.add_argument("-ht", "--heterogeneous", type=int, default=1,
                        help="whether distinguish relation type")
    parser.add_argument("--agg", type=int, default=1,
                        help="whether use aggregate layer")
    parser.add_argument("--gnn", type = str, default = "pgat", help="type of GNN: gcn/gat/graphsage")
    parser.add_argument("-m","--mask", type=int, default = 1, help="whether apply mask in GNN")
    parser.add_argument("--text-representation", type=str, default='ft', help="text representation: ft(fasttext), bow(bag-of-words), emb(embedding layer)")
    parser.add_argument("--mode", type=str, default='multi', help="whether use the concatenation or the difference of two node representation in the alignment model")
    parser.add_argument("--mh", type=int, default=0, help="whether use multi-head attention")
    parser.add_argument("--nh", type=int, default=2, help="number of heads in multi-head attention")
    parser.add_argument("--nl", type=int, default=4, help="number of layers in GNN")
    parser.add_argument("--scale", type=int, default=1, help="scale for balancing positive data")
    #parser.add_argument("--learnable", type=int, default=0,help="whether set learnable scaled parameter in mask")
    parser.add_argument("--concat", type=bool, default=False, help="whether concat at each hidden layer")
    #parser.add_argument("--gat", action='store_true', help="whether GCN or GAT is chosen")
    parser.add_argument("--dist-opt", type=int, default=0,
            help="[1: hinge loss, 0: binary classification]")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("-lr","--lr", type=float, default=0.08,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
            help="batch size")
    # parser.add_argument("--num-neighbors", type=int, default=10,
    #         help="number of neighbors to be sampled")
    # parser.add_argument("--num-negatives", type=int, default=10,
    #         help="number of negative links to be sampled")
    parser.add_argument("--num-test-negatives", type=int, default=10,
            help="number of negative links to be sampled in test setting")
    parser.add_argument("--hdim", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--dump", action='store_true',
            help="dump trained models (default=False)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
            help="Weight for L2 loss")
    parser.add_argument("--model-id", type=str,
        help="Identifier of the current model")
    parser.add_argument("--pretrain_path", type=str, default="../text/mode"
                                                             "l.bin",
        help="pretrained fastText path")
    args = parser.parse_args()


    print(args)

    main(args)
