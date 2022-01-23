
import argparse
import json
import logging
import numpy as np
from xclib.data import data_utils
from utils import prepare_data
if __name__=='__main__':

    #============ LOADING CONFIG FILES ====================

    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    with open('commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    
    DATASET = args.dataset              # dataset to be used
    EMB_TYPE = args.embedding_type      # Embedding type used
    RUN_TYPE = args.run_type            # Partial reveal or no reveal setting
    TST_TAKE = args.num_validation      # Number of nodes shortlisted at validation time

    # ====================== Start Logging ==================

    logging.basicConfig(format='%(asctime)s - %(message)s',
        filename="{}/models/XC-Net_log_{}_{}.txt".format(DATASET, RUN_TYPE,args.name), level=logging.INFO)
    logger = logging.getLogger("main_logger") 

    logger.info("================= STARTING NEW RUN =====================")
    
    logger.info(" ARGUMENTS ")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    
    # ====================== Loading Data ===================

    trn_point_titles = [line.strip() for line in open("{}/trn_X.txt".format(DATASET),"r",encoding="latin").readlines()]
    tst_point_titles = [line.strip() for line in open("{}/tst_X.txt".format(DATASET),"r",encoding="latin").readlines()]
    label_titles = [line.strip() for line in open("{}/Y.txt".format(DATASET),"r",encoding="latin").readlines()]
    
    logger.info("len(trn_point_titles), len(tst_point_titles), len(label_titles) = ", len(
        trn_point_titles), len(tst_point_titles), len(label_titles))
    
    # ======================= Loading Embeddings ===================

    trn_point_features = np.load("{}/{}CondensedData/trn_point_embs.npy".format(DATASET, EMB_TYPE))
    label_features = np.load("{}/{}CondensedData/label_embs.npy".format(DATASET, EMB_TYPE))
    tst_point_features = np.load("{}/{}CondensedData/tst_point_embs.npy".format(DATASET, EMB_TYPE))
    logger.info("trn_point_features.shape, tst_point_features.shape, label_features.shape",trn_point_features.shape,tst_point_features.shape,label_features.shape)
    
    # ======================= Loading sparse matrix ================

    logger.info("loading sparse matrix files")
    trn_X_Y = data_utils.read_sparse_file(
        "{}/trn_X_Y.txt".format(DATASET),force_header =True)
    tst_X_Y = data_utils.read_sparse_file(
        "{}/tst_X_Y.txt".format(DATASET),force_header=True)

    # ======================= Prepare Data ===============
    
    logger.info("preparing data") 
    tst_valid_inds, trn_X_Y, tst_X_Y_trn, tst_X_Y_val, node_features, valid_tst_point_features, label_remapping, adjecency_lists, NUM_TRN_POINTS = prepare_data(trn_X_Y, tst_X_Y, trn_point_features, tst_point_features, label_features,
                                                                                                                                                                trn_point_titles, tst_point_titles, label_titles, args,logger)
     


    
