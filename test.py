import argparse

from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument('--method', type=str, default='CL4SRec_XAUG', \
                        help='None, CL4SRec, CL4SRec_XAUG, DuoRec, DuoRec_XAUG, ...')
    parser.add_argument('--cl_loss_weight', type=float, default=0.1, help='weight for contrastive loss')
    parser.add_argument('--xai_method', type=str, default='occlusion', help='saliency, occlusion')

    args, _ = parser.parse_known_args()

    config_dict = {
        'neg_sampling': None,
        'method': args.method,
        'cl_loss_weight': args.cl_loss_weight,

        'xai_method': args.xai_method,
    }

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, method=args.method,
                config_file_list=config_file_list, config_dict=config_dict)