from castle.datasets import DAG, IIDSimulation
import torch,time,datetime
from src import myECA_prior,myECA_mlp_prior
from preparation import *
from generation_prior import *
from rich import print as rprint
import numpy as np 
from evaluation import count_accuracy,numerical_SHD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type='gpu'
# device = torch.device("cpu")
# device_type='cpu'
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=4)

def main(output_path):
    args={'n_nodes':40,'ER':2,'size':2,'graph_type':'ER','random':2,   
        'method':'nonlinear','sem_type':'mim',     
        'prior_type':'exist','proportion':0.3,'confidence':0.9,   
        'error_prior_proportion':0.2,'error_prior_type':'reverse_direct',   
        'alg':'ECA_mlp_prior','adaptive_degree':1}   
    args = get_config(args)
    rprint(vars(args))

    # weight_true_dag = DAG.erdos_renyi(n_nodes=args.n_nodes, n_edges=args.n_edges, weight_range=(0.5, 2.0), seed=1)
    # dataset = IIDSimulation(W=weight_true_dag, n=args.size*args.n_nodes, method=args.method, sem_type=args.sem_type)
    # true_dag, X = dataset.B, dataset.X
    


    weight_true_dag=np.loadtxt(f'data/W_true/{args.n_nodes}_{args.ER}_{args.graph_type}.csv',delimiter=',')
    true_dag=np.where(weight_true_dag!=0,1,0)
    X=np.loadtxt(f'data/X/{args.n_nodes}_{args.ER}_{args.size}_{args.graph_type}_{args.random}_{args.method}_{args.sem_type}.csv', delimiter=',')

    sigma,lambda1,lambda2= sigma_lambda(args)
    w_prior,edge_existence,error_prior=generate_prior(args,true_dag)
    print(true_dag)
    print(w_prior)
    if args.alg == 'ECA_prior':
        model = myECA_prior(lambda1=lambda1, sigma=sigma, loss_type='pdf')
    elif args.alg=='ECA_mlp_prior':
        model = myECA_mlp_prior(lambda1=lambda1, lambda2=lambda2, hidden_layers=(10,1), device_type=device_type)

    
    model.load_l1_penalty_parameter(lambda1)
    time1=time.time()
    model.learn(X)    
    print(model.causal_matrix)
    metric = count_accuracy(true_dag,model.causal_matrix)

    time2=time.time()
    metric.update(numerical_SHD(weight_true_dag,model.weight_causal_matrix))
    metric['time']=round(time2-time1,4)
    metric['lambda1']=round(lambda1,4)
    metric['lambda2']=round(lambda2,4)
    metric['sigma']=round(sigma,4)
    metric['finished']=datetime.datetime.now()
    rprint(metric)

    parameter={'n_nodes':args.n_nodes,'ER':args.ER,'size':args.size, 'graph_type':args.graph_type,'random':args.random, 'method':args.method,'sem_type':args.sem_type,
               'prior_type':args.prior_type,'proportion':args.proportion,'confidence':args.confidence,'error_prior_proportion':args.error_prior_proportion,'error_prior_type':args.error_prior_type,'alg':args.alg,'adaptive_degree':args.adaptive_degree}
    parameter_values = [str(i) for i in parameter.values()]
    metric_values = [str(i) for i in metric.values()]
    if not os.path.exists(output_path):
        with open(output_path, 'a') as f:
            f.write(','.join(list(parameter.keys()))+',' +
                    ','.join(list(metric.keys()))+'\n')
    with open(output_path, 'a') as f:
        eva_info = ','.join(parameter_values)+','+','.join(metric_values)
        f.write(f'{eva_info}\n')

    label='|'.join(parameter_values)
    np.savetxt(f'out/W_est/{label}.csv', model.weight_causal_matrix, delimiter=',', fmt='%.4f')
    np.savetxt(f'out/W_est/{label}|threshold.csv', model.causal_matrix, delimiter=',', fmt='%.4f')

if __name__ == '__main__':
    
    main('out/output.csv')
    os._exit(-1)
