import os
import argparse
from utils.helper import Helper as helper

def args():
    parser.add_argument('--name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--comm_rounds', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2024, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num_img_clients', type=int, default=3)
    parser.add_argument('--num_txt_clients', type=int, default=3)
    parser.add_argument('--num_mm_clients', type=int, default=4)
    parser.add_argument('--client_num_per_round', type=int, default=10)
    # === dataloader ===
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--alpha', type=float, default=0.1)
    # === optimization ===
    parser.add_argument('--server_lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

    parser.add_argument('--BAA', action="store_true", default=False)
    parser.add_argument('--disable_distill', action="store_true", default=False)
    parser.add_argument('--agg_method', type=str, default='SED', help='representation aggregation method')
    parser.add_argument('--mlp_local', action="store_true", default=False)
    parser.add_argument('--kd_weight', type=float, default=0.4, help='coefficient of kd')
    parser.add_argument('--interintra_weight', type=float, default=0.5, help='coefficient of inter+intra')
    parser.add_argument('--pub_data_num', type=int, default=10000, help='communication')
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--not_bert', action='store_true', default=False, help="server bert, client not bert")
    parser.add_argument('--partition', type=str, default='homo', help='homo or hetero')

parser = argparse.ArgumentParser(description='Federated Learning')
args()
args = parser.parse_args()
if __name__ == "__main__":
    os.chdir('.')
    from algorithms.MMFL_fedafd import MMFL
    args.save_dirs = helper.get_save_dirs(args.exp_dir, args.name)
    args.log_dir = args.save_dirs['logs']
    helper.set_seed(args.seed)

    Algo = MMFL(args)
    Algo.create_model(args)
    Algo.load_dataset(args)

    for round_n in range(args.comm_rounds):
        Algo.train(round_n)
    Algo.logger.log("Best:")
    Algo.engine.report_scores(step=Algo.best_metadata['best_epoch'],
                              scores=Algo.best_scores,
                              metadata=Algo.best_metadata,
                              prefix=Algo.engine.eval_prefix)