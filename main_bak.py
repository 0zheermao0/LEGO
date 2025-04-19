# coding=utf-8
import os
import csv

from argparse import ArgumentParser
from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
import math
from sklearn.metrics import f1_score
from gnn.moe import MoE
from common.graph_encoder import GraphEncoder as Graph2TextEncoder
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import json
import ollama
# os.environ['OLLAMA_HOST'] = 'http://192.168.1.100:11434'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acmv9')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-3) #5e-3
parser.add_argument("--weight_decay", type=float, default=2e-3) #2e-3
parser.add_argument("--drop_out", type=float, default=1e-1)

parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--label_rate", type=float, default=0.05)
parser.add_argument("--expert_num", type=int, default=3)
parser.add_argument("--llm", type=str, default='gemma3:4b-it-fp16')
parser.add_argument("--uncertainty_k", type=int, default=5)
parser.add_argument("--gate_coef", type=float, default=1e-1)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim
label_rate = args.label_rate

id = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}" \
    .format(args.source, args.target, seed, label_rate, args.learning_rate, args.weight_decay,
            encoder_dim)
print(id)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = False

dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print(source_data)

dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)


source_train_size = int(source_data.size(0) * label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)


def index2dense(edge_index,nnode=2708):
    indx = edge_index.cpu().detach().numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

def encode(data, cache_name, mask=None):
    encoded_output, experts_outputs, gate_loss = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        # encoded_output: [num_nodes, num_experts]
        encoded_output = encoded_output[mask]
        # experts_outputs: shape=[num_nodes, num_experts, d_feature]
        experts_outputs = experts_outputs[mask]

    return encoded_output, experts_outputs, gate_loss

def predict(data, cache_name, mask=None):
    encoded_output, _, _ = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return accuracy, macro_f1, micro_f1


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    encoded_output, experts_outputs, _ = encode(data, cache_name, mask)
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy, macro_f1, micro_f1 = evaluate(preds, labels)
    
    # Log expert selection during testing for target data
    if cache_name == args.target:
        moe_expert_indices, moe_expert_probs = encoder.get_node_expert_assignment(data.x, data.edge_index)
        expert_selection_log = "log/{}-expert-selection.csv".format(args.target)
        with open(expert_selection_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Node ID', 'Selected Expert', 'Probability'])
            for node_id in range(moe_expert_indices.shape[0]):
                for k in range(encoder.k):
                    expert_idx = moe_expert_indices[node_id, k].item()
                    expert_prob = moe_expert_probs[node_id, k].item()
                    writer.writerow([node_id, expert_idx, expert_prob])

    return accuracy, macro_f1, micro_f1, encoded_output

def get_renode_weight(data, pseudo_label):

    ppr_matrix = data.new_adj
    gpr_matrix = []
    for iter_c in range(data.num_classes):
        iter_gpr = torch.mean(ppr_matrix[pseudo_label==iter_c],dim=0).squeeze()
        gpr_matrix.append(iter_gpr)
    gpr_matrix = torch.stack(gpr_matrix,dim=0).transpose(0,1)

    base_w  = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)

    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix =  torch.mm(ppr_matrix,gpr_matrix) - torch.mm(ppr_matrix,gpr_rn)/(data.num_classes-1.0)

    label_matrix = F.one_hot(pseudo_label, gpr_matrix.size(1)).float()
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)

    #computing the ReNode Weight
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=True)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]

    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(nnode-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)

    return rn_weight


loss_func = nn.CrossEntropyLoss().to(device)

# encoder = GNN().to(device)
# from gnn.params import args as params_args
# Configure num_hops for experts to support 1-hop, 2-hop, multi-hop models
num_hops_config = [1, 2, 3][:args.expert_num] if args.expert_num >= 3 else [1] * args.expert_num
encoder = MoE(input_size=source_data.num_features, output_size=encoder_dim, num_experts=args.expert_num, k=1
          , coef=args.gate_coef, expert_gnn_type='ppmi', gating_gnn_type='gat', num_hops=num_hops_config).to(device)

cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)


encoded_source, _, _ = encode(source_data, args.source)
encoded_target, _, _ = encode(target_data, args.target)

with open ('tmp/'+args.source+'.pkl', 'rb') as f:
    source_edge_index, norm = pickle.load(f)
with open ('tmp/'+args.target+'.pkl', 'rb') as f:
    target_edge_index, norm = pickle.load(f)

source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)
target_data.new_adj = index2dense(target_edge_index, target_data.num_nodes).to(device)


models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

# Set up CSV logging
log_csv_path = "log/{}-{}-metrics.csv".format(args.source, args.target)
with open(log_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write hyperparameters
    writer.writerow(['Hyperparameters'])
    writer.writerow(['Source', args.source])
    writer.writerow(['Target', args.target])
    writer.writerow(['Seed', args.seed])
    writer.writerow(['Learning Rate', args.learning_rate])
    writer.writerow(['Weight Decay', args.weight_decay])
    writer.writerow(['Dropout', args.drop_out])
    writer.writerow(['Encoder Dim', args.encoder_dim])
    writer.writerow(['Label Rate', args.label_rate])
    writer.writerow(['Expert Num', args.expert_num])
    writer.writerow(['LLM', args.llm])
    writer.writerow([])  # Empty row for separation
    # Write headers for epoch metrics
    writer.writerow(['Epoch', 'Cls Loss', 'Gate Loss', 'Select Loss', 'High Quality Semi Loss', 'Total Loss', 'Source Acc', 'Target Acc', 'Macro F1', 'Micro F1'])

epochs = 200


def Entropy(input, weight, label):
    softmax_out = nn.Softmax(dim=-1)(input)
    entropy = -label * torch.log(softmax_out + 1e-5)
    entropy_loss = torch.mean(weight * torch.sum(entropy, dim=1))

    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    return entropy_loss

def calculate_expert_uncertainty(experts_outputs, num_classes, cls_model, uncertainty_k=5):
    """
    计算专家之间的不确定性，返回不确定性最高的节点的mask
    experts_outputs: shape=[num_nodes, num_experts, d_feature]
    """
    import scipy.special
    num_nodes, num_experts, d_feature = experts_outputs.shape

    # 将每个专家的输出通过分类模型转换为类别概率
    experts_logits = torch.zeros(num_nodes, num_experts, num_classes, device=experts_outputs.device)
    for i in range(num_experts):
        experts_logits[:, i, :] = cls_model(experts_outputs[:, i, :])

    # 将专家输出转换为概率分布
    experts_probs = torch.softmax(experts_logits, dim=-1)  # [num_nodes, num_experts, num_classes]

    # 计算每个节点的专家间不确定性
    # 首先获取所有专家的对数概率
    log_probs = torch.log(experts_probs + 1e-9)  # [num_nodes, num_experts, num_classes]
    # 对每个类别，在专家维度上进行logsumexp操作
    joint_log_probs = torch.logsumexp(log_probs, dim=1)  # [num_nodes, num_classes]

    # 计算期望比率
    total_confidence = torch.softmax(joint_log_probs, dim=-1)
    expected_ratios = joint_log_probs + torch.log(total_confidence + 1e-9)
    expected_ratios = scipy.special.logsumexp(expected_ratios.cpu().detach().numpy(), axis=-1)
    uncertainty = torch.from_numpy(expected_ratios).float().to(experts_outputs.device)

    # 选择不确定性最高的k个节点
    k = min(uncertainty_k, uncertainty.size(0))
    _, topk_indices = torch.topk(uncertainty, k, dim=0)
    uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)
    uncertainty_mask[topk_indices] = True

    return uncertainty_mask

def get_max_hop_neighbors(edge_index, num_nodes, mask):
    """
    找到给定节点的所有邻居（max-hop neighbors）
    Args:
        edge_index: 图的边索引，形状为 (2, num_edges)
        num_nodes: 图中节点总数
        mask: 不确定性最高的节点的布尔掩码
    Returns:
        neighbor_mask: 包含原始节点及其所有邻居的布尔掩码
    """
    # 初始化邻居掩码为原始掩码
    neighbor_mask = mask.clone()
    current_mask = mask.clone()
    hop_count = 0

    # 构建邻接矩阵
    adj_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1  # 无向图

    # print(f"Starting neighbor expansion with {mask.sum().item()} initial nodes")

    # 持续扩展邻居直到没有新的邻居被发现
    while True:
        hop_count += 1
        # 计算当前掩码节点的邻居
        new_neighbors = torch.matmul(adj_matrix, current_mask.float()) > 0
        new_count = (new_neighbors & ~neighbor_mask).sum().item()
        # 如果没有新的邻居，退出循环
        if new_count == 0:
            # print(f"No new neighbors found after {hop_count} hops. Total neighbors: {neighbor_mask.sum().item()}")
            break
        # 更新邻居掩码
        neighbor_mask |= new_neighbors
        # 更新当前掩码为新发现的邻居，用于下一跳
        current_mask = new_neighbors & ~neighbor_mask
        # print(f"Hop {hop_count}: Found {new_count} new neighbors, total now: {neighbor_mask.sum().item()}")

    return neighbor_mask

def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    encoded_source, experts_outputs, source_gate_loss = encode(source_data, args.source)
    source_logits = cls_model(encoded_source)

    # 计算专家间不确定性并生成mask
    uncertainty_mask = calculate_expert_uncertainty(experts_outputs, source_data.num_classes, cls_model, args.uncertainty_k)
    # print(f"Uncertainty mask shape: {uncertainty_mask.shape}, number of high uncertainty nodes: {uncertainty_mask.sum().item()}")

    # 找到不确定性最高节点的max-hop邻居
    max_hop_neighbors_mask = get_max_hop_neighbors(source_data.edge_index, source_data.num_nodes, uncertainty_mask)
    # print(f"Number of nodes including max-hop neighbors: {max_hop_neighbors_mask.sum().item()}")

    # 送入LLM的掩码应包含uncertainty_mask和max_hop_neighbors_mask
    combined_mask = uncertainty_mask | max_hop_neighbors_mask
    # print(f"Number of nodes for LLM input (uncertainty + neighbors): {combined_mask.sum().item()}")

    # 使用Graph2TextEncoder对图进行编码
    graph2text_encoder = Graph2TextEncoder()
    graph_description = graph2text_encoder.encode(source_data.edge_index, mask=combined_mask, num_nodes=source_data.num_nodes)
    # print(f'Graph description for high uncertainty nodes and neighbors: {graph_description}')

    # 使用LLM获取专家选择建议
    with torch.no_grad():
        # 获取uncertainty_mask下节点的索引
        uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()
        if not uncertainty_node_indices:
            print("没有发现高不确定性节点，跳过LLM专家选择。")
        else:
            expert_selections = {}
            # 构造所有节点的prompt
            prompts = [
                f"""
                You are an expert on GNN experts selector, given graph: {graph_description}
                and 3 GNN experts: (0:1-hop, 1:2-hop, 2:3-hop)
                - 1-hop: Use when direct neighbors provide sufficient classification signals.  
                - 2-hop: Use for indirect relationships.  
                - 3-hop: Use for long-range dependencies or hierarchical structures.  
                give out your choice on expert directly for node {node_id}.
                Please note:
                1. If the structure around the node is simple and there are few neighbors, it is recommended to choose 1-hop expert
                2. If the node has more 2-hop neighbors, it is recommended to choose 2-hop expert
                3. If the node is in a complex community structure, it is recommended to choose 3-hop expert
                4. Given the specific reason for the selection, it is necessary to be based on the actual structural characteristics of the node.
                return in json format directly:
                {{
                    "expert": 0, 1, or 2,
                    "reason": "your reason"
                }}
                """
                for node_id in uncertainty_node_indices
            ]
            # 用ollama逐条推理
            for idx, node_id in enumerate(uncertainty_node_indices):
                prompt = prompts[idx]
                try:
                    response = ollama.generate(
                        model=args.llm,  # 替换为你实际的ollama模型名
                        prompt=prompt,
                        format='json'
                    )
                    json_output_str = response['response']
                    try:
                        parsed_json = json.loads(json_output_str)
                        expert = parsed_json.get("expert", 0)
                        if isinstance(expert, int) and 0 <= expert < args.expert_num:
                            expert_selections[node_id] = expert
                        else:
                            expert_selections[node_id] = np.random.randint(0, args.expert_num)
                        
                    except json.JSONDecodeError as e:
                        print(f"解析节点 {node_id} 的LLM专家选择建议失败: {json_output_str}")
                        expert_selections[node_id] = np.random.randint(0, args.expert_num)  # 默认值
                except Exception as e:
                    print(f"使用 ollama.generate 时出错: {e}")
                    expert_selections[node_id] = 0
            print(f'所有高不确定性节点的专家选择: {expert_selections}')

    # 获取MoE的专家分配
    moe_expert_indices, moe_expert_probs = encoder.get_node_expert_assignment(source_data.x, source_data.edge_index)

    # 计算MoE和LLM选择的蒸馏损失（KL散度蒸馏）
    select_loss = 0.0
    if expert_selections:
        # 创建LLM选择的专家分布
        num_experts = args.expert_num
        llm_expert_dist = torch.zeros(len(uncertainty_node_indices), num_experts, device=device)
        moe_expert_dist = torch.zeros(len(uncertainty_node_indices), num_experts, device=device)

        for idx, node_id in enumerate(uncertainty_node_indices):
            llm_expert = expert_selections.get(node_id, 0)
            llm_expert_dist[idx, llm_expert] = 1.0

            # 获取MoE的专家概率分布
            node_idx = torch.where(torch.arange(source_data.num_nodes, device=device) == node_id)[0]
            if len(node_idx) > 0:
                node_idx = node_idx.item()
                for k in range(encoder.k):
                    expert_idx = moe_expert_indices[node_idx, k].item()
                    expert_prob = moe_expert_probs[node_idx, k].item()
                    moe_expert_dist[idx, expert_idx] = expert_prob

        # 蒸馏温度
        temperature = 2.0
        # 平滑分布
        llm_soft = torch.softmax(llm_expert_dist / temperature, dim=-1)
        moe_soft = torch.softmax(moe_expert_dist / temperature, dim=-1)
        # KL散度损失（注意detach LLM分布）
        select_loss = torch.nn.functional.kl_div(
            moe_soft.log(), llm_soft.detach(), reduction='batchmean'
        ) * (temperature ** 2)
        # print(f"KL Distillation Loss between MoE and LLM selections: {select_loss.item()}")

    # classifier loss:
    cls_loss = loss_func(source_logits[label_mask], source_data.y[label_mask])

    # pseudo labeling loss:
    _, s_plabel = torch.max(source_logits, dim=1)
    s_plabel[label_mask] = source_data.y[label_mask]
    # _, t_plabel = torch.max(target_logits, dim=1)

    s_weight = get_renode_weight(source_data, s_plabel).to(device)
    # t_weight = get_renode_weight(target_data, t_plabel).to(device)

    s_plabel = F.one_hot(s_plabel, source_data.num_classes)
    # t_plabel = F.one_hot(t_plabel, target_data.num_classes)
    semi_loss = Entropy(source_logits[~label_mask], s_weight[~label_mask], s_plabel[~label_mask]) #+ Entropy(target_logits, t_weight, t_plabel)
    # target_domain_preds = domain_model(encoded_target)

    # 计算MoE对未标记数据的熵作为置信度
    moe_softmax = nn.Softmax(dim=-1)(source_logits[~label_mask])
    moe_entropy = -torch.sum(moe_softmax * torch.log(moe_softmax + 1e-5), dim=1)

    # 计算每个专家对未标记数据的熵
    num_experts = experts_outputs.shape[1]
    expert_entropies = torch.zeros(experts_outputs.shape[0], num_experts, device=device)
    for i in range(num_experts):
        expert_logits = cls_model(experts_outputs[:, i, :])
        expert_softmax = nn.Softmax(dim=-1)(expert_logits[~label_mask])
        expert_entropies[~label_mask, i] = -torch.sum(expert_softmax * torch.log(expert_softmax + 1e-5), dim=1)

    # 计算专家熵的平均值
    avg_expert_entropy = torch.mean(expert_entropies[~label_mask], dim=1)

    # 筛选高质量数据：MoE熵低于平均专家熵
    high_quality_mask = moe_entropy < avg_expert_entropy
    high_quality_mask_full = torch.zeros(source_data.num_nodes, dtype=torch.bool, device=device)
    high_quality_mask_full[~label_mask] = high_quality_mask

    # 计算高质量数据的半监督损失
    if high_quality_mask.sum() > 0:
        high_quality_semi_loss = Entropy(source_logits[~label_mask][high_quality_mask],
                                         s_weight[~label_mask][high_quality_mask],
                                         s_plabel[~label_mask][high_quality_mask])
    else:
        high_quality_semi_loss = 0.0

    gate_loss = source_gate_loss #+ target_gate_loss

    print(f"Loss: cls_loss: {cls_loss.item()}, gate_loss: {gate_loss.item()}, select_loss: {select_loss.item() if isinstance(select_loss, torch.Tensor) else select_loss}, high_quality_semi_loss: {high_quality_semi_loss.item() if isinstance(high_quality_semi_loss, torch.Tensor) else high_quality_semi_loss}\n")
    # loss = cls_loss + loss_grl + (float(epoch) / epochs) * semi_loss + gate_loss * 0.5
    loss = cls_loss + gate_loss + select_loss + (float(epoch) / epochs) * (semi_loss + high_quality_semi_loss)

    # Log losses for this epoch to CSV
    with open(log_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, cls_loss.item(), gate_loss.item(), select_loss.item() if isinstance(select_loss, torch.Tensor) else select_loss, high_quality_semi_loss.item() if isinstance(high_quality_semi_loss, torch.Tensor) else high_quality_semi_loss, loss.item(), '', '', '', ''])

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0
best_micro_f1 = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct, _, _, output_source = test(source_data, args.source, source_data.test_mask)
    target_correct, macro_f1, micro_f1, output_target = test(target_data, args.target)
    print("Epoch: {}, source_acc: {}, target_acc: {}, macro_f1: {}, micro_f1: {}".format(epoch, source_correct,
                                                                                         target_correct, macro_f1,
                                                                                         micro_f1))
    # Update CSV with accuracy and F1 metrics for this epoch
    with open(log_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows:
            if row and row[0] == str(epoch):
                row[6] = source_correct.item() if isinstance(source_correct, torch.Tensor) else source_correct
                row[7] = target_correct.item() if isinstance(target_correct, torch.Tensor) else target_correct
                row[8] = macro_f1
                row[9] = micro_f1
                break
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch
        with open ('log/{}_{}_embeddings.pkl'.format(args.source, args.target),'wb') as f:
            pickle.dump([output_source.cpu().detach().numpy(), output_target.cpu().detach().numpy()], f)
print("=============================================================")
line = "{}\n - Epoch: {}, best_source_acc: {}, best_target_acc: {}, best_macro_f1: {}, best_micro_f1: {}" \
    .format(id, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)

print(line)


with open("log/{}-{}.log".format(args.source, args.target), 'a') as f:
    line = "{} - Epoch: {:0>3d}, best_macro_f1: {:.5f}, best_micro_f1: {:.5f}\t" \
               .format(id, best_epoch, best_macro_f1, best_micro_f1) + time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n"
    f.write(line)
