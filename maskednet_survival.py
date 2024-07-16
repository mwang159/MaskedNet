import torch
import torch.nn as nn
import torch.nn.functional as F
import shap

from captum.attr import Saliency
from captum.attr import IntegratedGradients
from captum.attr import InputXGradient
from captum.attr import GuidedBackprop
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import FeatureAblation

from captum.attr import NoiseTunnel

import pandas as pd
import numpy as np
import copy
import os,sys,time,datetime
import math
import random
import re

import argparse
import sys

projdir="./"
sys.path.append(projdir)

from survival import load_data, load_data_unsort, load_pathway, MaskedNet, trainMaskedNet, infoLine, evaluate


#
def getNextData(paramHash, dataHash, dataType):
    if paramHash[dataType]["current_index"] < 0 or paramHash[dataType]["current_index"] >= paramHash[dataType]["max_batch_num"]:
        paramHash[dataType]["current_index"] = 0
    #
    
    # calculate index
    batchSize = paramHash["net_batch_size"]
    index = paramHash[dataType]["current_index"] * batchSize
    
    # extract data
    data_x = np.array(dataHash[dataType]["x"][index:(index+batchSize)] )
    data_clinic = np.array(dataHash[dataType]["clinic"][index:(index+batchSize)] )

    # move index
    paramHash[dataType]["current_index"] = paramHash[dataType]["current_index"] + 1
    
    # return data
    return data_x,data_clinic
#


def getUniqueRunID():
    import hashlib,time,os
    mix = str(time.time()) + "|" + str(os.uname()) + "|" + str(os.getpid())
    return hashlib.md5(mix.encode()).hexdigest()[-8:]


def explain_captum(model, paramHash, dataHash, dataType, reference):
    ## no need for reference
    if paramHash["net_explain_explainer"] == "Saliency":
        algo = Saliency(model)
    #
    if paramHash["net_explain_explainer"] == "InputXGradient":
        algo = InputXGradient(model)
    #
    if paramHash["net_explain_explainer"] == "GuidedBackprop":
        algo = GuidedBackprop(model)
    #    
    
    ## need reference
    if paramHash["net_explain_explainer"] == "IntegratedGradients":
        algo = IntegratedGradients(model)
    #
    if paramHash["net_explain_explainer"] == "DeepLift":
        algo = DeepLift(model)
    #
    if paramHash["net_explain_explainer"] == "DeepLiftShap":
        algo = DeepLiftShap(model)
    #
    if paramHash["net_explain_explainer"] == "FeatureAblation":
        algo = FeatureAblation(model)
    #

    explainer = algo
    #
    
    needRef = True
    if paramHash["net_explain_explainer"] in ["Saliency", "InputXGradient", "GuidedBackprop", "GuidedGradCam"]:
        needRef = False
    #
    
    ##
    attHash  = {}
     
    for refIndex in range(paramHash["net_explain_ref_size"]):
        msg = "-------------- Reference: " + str(refIndex + 1) + "/" + str( paramHash["net_explain_ref_size"] ) + " -----------------"
        infoLine(msg)
        ref_data_x      = np.array( dataHash[reference]["x"][refIndex:(refIndex+1)] )
        ref_data_clinic = np.array( dataHash[reference]["clinic"][refIndex:(refIndex+1)] )
        base_data_c_x      = None
        base_data_c_clinic = None

        resetData(paramHash)
        
        # data index for each round explanation
        dataIndex = 0

        for batch_idx in range(paramHash[dataType]["max_batch_num"]):
            #msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(paramHash[dataType]["max_batch_num"]) + " -----------------"
            #infoLine(msg)
            
            data_x, data_clinic = getNextData(paramHash, dataHash, dataType)
            if base_data_c_x is None or len(data_x) != len(base_data_c_x):
                base_data_c_x = np.array( np.broadcast_to(ref_data_x, data_x.shape) )
                base_data_c_clinic = np.array( np.broadcast_to(ref_data_clinic, data_clinic.shape) )
                base_data_x = torch.from_numpy(base_data_c_x) # .cuda()
                base_data_clinic = torch.from_numpy(base_data_c_clinic)#.cuda()
            #

            data_x = torch.from_numpy(data_x)#.cuda()
            data_clinic = torch.from_numpy(data_clinic)#.cuda()

            if needRef:
                if paramHash["net_clinic_nodes"] == 0:
                    att = explainer.attribute(data_x, baselines=base_data_x)
                else:
                    att = explainer.attribute(torch.cat((data_x, data_clinic),dim=1), \
                                          baselines=torch.cat((base_data_x, base_data_clinic),dim=1))
                #
            else:
                if paramHash["net_clinic_nodes"] == 0:
                    att = explainer.attribute(data_x)
                else:
                    att = explainer.attribute(torch.cat((data_x, data_clinic),dim=1))
                #
            #
            # importance for each gene, for each sample for each reference
            for i in range(len(att)):  ## for each one of the 16 samples
                if dataIndex not in attHash:
                    attHash[dataIndex] =  []
                attHash[dataIndex].append(att[i].detach().numpy().flatten() ) 
                dataIndex = dataIndex + 1
            #
        #
        if not needRef:
            break # some explaners do not support reference, so they can quit here
        #
    #
    #print(attHash[0][0].shape)
    #print(len(attHash[1]))
    #print(len(attHash))
    return attHash
#


def explain(model, paramHash, dataHash, dataType, reference):
    ### Set explainer
    infoLine("Set explainer as " + paramHash["net_explain_explainer"] )
    if paramHash["net_explain_explainer"] in ["Saliency", "InputXGradient", "GuidedBackprop", "IntegratedGradients", "DeepLift", "DeepLiftShap", "FeatureAblation"]:
        attHash = explain_captum(model, paramHash, dataHash, dataType, reference)
    #
    
    if paramHash["net_explain_separate"] == "no":
        outDIR = paramHash["explain_dir"] + "/aggregate/"
        os.system("mkdir -p " + outDIR)
        runID = getUniqueRunID()
        outfile = outDIR + paramHash["net_explain_explainer"] + "."+ str(paramHash["net_explain_pool_post"]) + ".ref-" + str(paramHash["net_explain_ref_size"]) + "." + paramHash["net_explain_ref_type"] + "." + runID + ".dat"
        with open( outfile , "wt" ) as fo:
            for dataIndex in range( len( attHash ) ):   
                n_row=attHash[dataIndex][0].shape[0]
                #print(n_row)
                attHash[dataIndex]  = np.abs( np.array(attHash[dataIndex] )) 
                #print(attHash[dataIndex].shape)
                contrib = np.median(attHash[dataIndex], axis=0) #[:,range(0,n_row-paramHash["net_clinic_nodes"])],axis=0)
                outline = "\t".join([ str(v) for v in contrib ] )
                fo.write( outline + "\n" )
        #
    else:
        outDIR = paramHash["explain_dir"] + "/" + paramHash["net_explain_explainer"] + "/"
        os.system("mkdir -p " + outDIR)
        runID = getUniqueRunID()
        
        for refIndex in range(paramHash["net_explain_ref_size"]):
            outfile = outDIR + paramHash["net_explain_explainer"] + "." + str(paramHash["net_explain_pool_post"]) + ".refIndex-" + str(refIndex + 1) + "." + paramHash["prefix"] + ".dat"
            with open( outfile , "wt" ) as fo:
                for dataIndex in range( len( attHash ) ):       
                    attHash[dataIndex] = np.array(attHash[dataIndex])
                    contrib = attHash[dataIndex][refIndex]
                    outline = "\t" .join( [ str(v) for v in contrib ] )
                    fo.write( outline + "\n" )
            #
            if len(attHash[0]) < 2:
                break # some explaners do not support reference, so they can quit here
            #
        #
#


def resetData(paramHash):
    for dataType in ["train", "valid","test","pool"]:
        if dataType in paramHash:
            paramHash[dataType]["current_index"] = -1
    #
#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version: 1.0(MaskedNet for survival analysis).",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general argument
    parser.add_argument("-i", dest="inDIR",     type=str,   required=True,                      help="The directory of input data")
    parser.add_argument("-o", dest="outDIR",    type=str,   required=True,                      help="The directory of output data or model")
    parser.add_argument("-M", dest="runMode",   type=str,   required=False, default="train",    help="Running mode: ", choices=['train', 'evaluate', 'predict', 'explain'])
    parser.add_argument("-X", dest="prefix",    type=str,   required=False, default="1",    help="prefix")
    parser.add_argument("-mod", dest="model",    type=str,   required=False, default="model",    help="use this model")


    # training options
    parser.add_argument("-d1", dest="dropout1",  type=float, required=False, default=0.25,      help="Dropout rate of genes")
    parser.add_argument("-d2", dest="dropout2",  type=float, required=False, default=0.25,      help="Dropout rate of pathways")
    parser.add_argument("-d3", dest="dropout3",  type=float, required=False, default=0.25,      help="Dropout rate of hidden layer after pathway layer")
    parser.add_argument("-d4", dest="dropout4",  type=float, required=False, default=0.25,      help="Dropout rate of the third hidden layer")
    parser.add_argument("-n1", dest="postPath_1",   type=int,   required=False, default=100,       help="Number of neuron in the first layer after pathway layer")
    parser.add_argument("-n2", dest="postPath_2",   type=int,   required=False, default=100,        help="Number of neuron in the second layer after pathway layer")
    parser.add_argument("-p", dest="patience",   type=int,   required=False, default=5,         help="Max number of epochs with no improvement")
    parser.add_argument("-r", dest="learnRate",  type=float, required=False, default=0.0006,    help="Learning rate")
    parser.add_argument("-R", dest="learnRateFactor",  type=float, required=False, default=1.0,    help="Learning rate adjustment factor")
    parser.add_argument("-l", dest="l2Reg",      type=float, required=False, default=0.001,     help="L2 regularization")
    parser.add_argument("-e", dest="maxEpoch",   type=int,   required=False, default=100,       help="Max epoch number")
    parser.add_argument("-m", dest="minEpoch",   type=int,   required=False, default=0,         help="Min epoch number")
    parser.add_argument("-T", dest="earlyStop",  type=int,   required=False, default=0,    help="Early Stop")
    # explain options
    parser.add_argument("-b", dest="batchSize", type=int,   required=False, default=16,        help="Batch size")
    parser.add_argument("-B", dest="refType",   type=str,   required=False, default="univeral",help="Type of reference samples", choices=['universal','zero'])
    parser.add_argument("-S", dest="refSize",   type=int,   required=False, default=1,         help="Number of reference samples to be used in the explanation step")
    parser.add_argument("-P", dest="poolName",  type=str,   required=False,                     help="Prefix of pooled file")
    parser.add_argument("-k", dest="separateRef",type=str,  required=False, default="no",       help="Whether output each reference output separately", choices=['no', 'yes'])
    parser.add_argument("-E", dest="explainer", type=str,   required=False, default="DeepLiftShap", help="Type of model explainer", choices=['Saliency', 'IntegratedGradients', 'InputXGradient', 'GuidedBackprop', 'DeepLift', 'DeepLiftShap', 'FeatureAblation', "DeepExplainer", "DeepExplainerFactorize"])

    # train, evaluat, predict, explain option
    parser.add_argument("-nC", dest="clinicunit",type=int,   required=False, default=1,         help="Number of clinic nodes")

    
    args=parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # main parames
    paramHash = {}
    dataHash  = {}

    # param
    paramHash["early_stopping"]        = False if args.earlyStop==0 else True
    paramHash["prefix"]                = args.prefix
    paramHash["net_batch_size"]        = args.batchSize 
    paramHash["data_in_dir"]           = args.inDIR 
    paramHash["data_out_dir"]          = args.outDIR
    paramHash["run_mode"]              = args.runMode         
    paramHash["net_max_epoch"]         = args.maxEpoch       
    paramHash["net_min_epoch"]         = args.minEpoch      
    paramHash["net_drop_out"]          = [args.dropout1, args.dropout2, args.dropout3, args.dropout4]  
    paramHash["net_postPath_1_nodes"]  = args.postPath_1      
    paramHash["net_postPath_2_nodes"]  = args.postPath_2      
    paramHash["net_clinic_nodes"]      = args.clinicunit      
    paramHash["net_learn_rate"]        = args.learnRate       
    paramHash["net_lr_factor"]         = args.learnRateFactor 
    paramHash["net_patience"]          = args.patience      
    paramHash["net_l2_regularization"] = args.l2Reg        
    paramHash["net_explain_ref_type"]  = args.refType        
    paramHash["net_explain_ref_size"]  = args.refSize      
    paramHash["net_explain_pool_post"] = args.poolName 
    paramHash["net_explain_separate"]  = args.separateRef 
    paramHash["net_explain_explainer"] = args.explainer 
    paramHash["model_toUse"]           = args.model 

    
    # read pathway mask
    Pathway_Mask = load_pathway(paramHash["data_in_dir"]  + "/pathway_mask_ensembl.dat", torch.FloatTensor)
    # add to paraHash
    paramHash["In_Nodes"]      = Pathway_Mask.shape[1]   ### number of genes
    paramHash["Pathway_Nodes"] = Pathway_Mask.shape[0]   ### number of pathways
    paramHash["Pathway_Mask"]  = Pathway_Mask

    # model directory
    paramHash["model_dir"]   = paramHash["data_out_dir"] + "/saved_models"
    paramHash["evaluate_dir"]= paramHash["data_out_dir"] + "/evaluation"
    paramHash["predict_dir"] = paramHash["data_out_dir"] + "/prediction"
    paramHash["explain_dir"] = paramHash["data_out_dir"] + "/explanation"
    paramHash["result_dir"]  = paramHash["data_out_dir"] + "/result"

    os.system("mkdir -p " + paramHash["result_dir"] )
    if paramHash["run_mode"] == "train":
        os.system("mkdir -p " + paramHash["model_dir"] )
    #
    if paramHash["run_mode"] == "evaluate":
        os.system("mkdir -p " + paramHash["evaluate_dir"] )
    #
    if paramHash["run_mode"] == "predict":
        os.system("mkdir -p " + paramHash["predict_dir"] )
    #
    if paramHash["run_mode"] == "explain":
        os.system("mkdir -p " + paramHash["explain_dir"] )
    #
    
    # read data
    if paramHash["run_mode"] == "train":
        for dataType in ["train", "valid"]:
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            dataHash[dataType]  = {}
            paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            x,ytime,yevent,clinic, sampleID = load_data(paramHash[dataType]["infile"], paramHash["net_clinic_nodes"], torch.FloatTensor)
            dataHash[dataType]["x"] = x
            dataHash[dataType]["ytime"]  = ytime
            dataHash[dataType]["yevent"] = yevent
            dataHash[dataType]["clinic"] = clinic
            dataHash[dataType]["sample"] = sampleID
        #
        infoLine("Report parameters")
        print(paramHash)
        print("\n")
    #

    if paramHash["run_mode"] == "evaluate":
        # read data
        for dataType in ["train","valid","test"]:
            if not os.path.exists( paramHash["data_in_dir"] + "/"+dataType+".dat" ):
                continue
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            dataHash[dataType]  = {}
            paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            x, ytime, yevent, clinic, sampleID = load_data(paramHash[dataType]["infile"], paramHash["net_clinic_nodes"], torch.FloatTensor)
            dataHash[dataType]["x"] = x
            dataHash[dataType]["ytime"]  = ytime
            dataHash[dataType]["yevent"] = yevent
            dataHash[dataType]["clinic"] = clinic
            dataHash[dataType]["sample"] = sampleID
        #
    #
    
    if paramHash["run_mode"] == "predict":
        for dataType in ["pool","train","valid","test"]:
            if not os.path.exists( paramHash["data_in_dir"] + "/"+dataType+".dat" ):
                continue
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            dataHash[dataType]  = {}
            paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            x, ytime, yevent, clinic, sampleID = load_data(paramHash[dataType]["infile"], paramHash["net_clinic_nodes"], torch.FloatTensor)
            dataHash[dataType]["x"] = x
            dataHash[dataType]["ytime"]  = ytime
            dataHash[dataType]["yevent"] = yevent
            dataHash[dataType]["clinic"] = clinic
            dataHash[dataType]["sample"] = sampleID
        #
    #
    
    if paramHash["run_mode"] == "explain":
        for dataType in ["pool",paramHash["net_explain_ref_type"]]:
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            dataHash[dataType]  = {}
            if dataType == "pool":
                paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+"."+paramHash["net_explain_pool_post"]+".dat"
            else:
                print(paramHash["data_in_dir"] + "/"+dataType+".dat")
                paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            x, ytime, yevent, clinic, sampleID = load_data_unsort(paramHash[dataType]["infile"], paramHash["net_clinic_nodes"], torch.FloatTensor)
            dataHash[dataType]["x"]  = x
            dataHash[dataType]["ytime"]  = ytime
            dataHash[dataType]["yevent"] = yevent
            dataHash[dataType]["clinic"] = clinic
            dataHash[dataType]["sample"] = sampleID
            paramHash[dataType]["max_batch_num"] = math.ceil(x.shape[0] / paramHash["net_batch_size"] )
        #
    #

    # reset cache on gpu
    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    ############################################ train ############################################
    if paramHash["run_mode"] == "train":
        infoLine("Start survival trainning")
        loss_train, cindex_tr, loss_valid, cindex_va, earlyStop = \
              trainMaskedNet(dataHash['train']['x'], dataHash['train']['clinic'], dataHash['train']['ytime'], dataHash['train']['yevent'], \
                             dataHash['valid']['x'], dataHash['valid']['clinic'], dataHash['valid']['ytime'], dataHash['valid']['yevent'], \
                             paramHash)
        rr = [[int(x) for x in range(1,paramHash["net_max_epoch"]+1)], \
              [float(x) for x in loss_train], [float(x) for x in loss_valid], \
              np.array(cindex_tr), np.array(cindex_va), np.array(earlyStop)]
        rr = pd.DataFrame(data=rr)
        rr = rr.T
        rr.columns = ["Epoch", "train_loss", "valid_loss", "train_cindex", "valid_cindex", "early_stop"]
        rr.to_csv(paramHash["data_out_dir"]+"/result/performance_"+ paramHash["prefix"] +".csv", index=False)

    #
    if paramHash["run_mode"] == "evaluate":
        # load model
        infoLine("Loading trained model")
        net = MaskedNet(paramHash)
        net.load_state_dict(torch.load(paramHash["model_dir"] + "/" + paramHash["model_toUse"] +  ".pt"))
        net.eval()
        with open( paramHash["evaluate_dir"] + "/evaluation.tab", "wt" ) as fo:
            for dataType in ["train","valid","test"]:
                if dataType not in dataHash:
                    continue
                infoLine("Make evaluation " + dataType)
                data = {"x":dataHash[dataType]["x"],     "clinic":dataHash[dataType]["clinic"], \
                    "ytime":dataHash[dataType]["ytime"], "yevent":dataHash[dataType]["yevent"]}
                eval_retData= evaluate(net, data, paramHash)
                outline = "cindex: " + str(eval_retData[1])
                infoLine(outline)
                fo.write(outline + "\n")
            # 
        #
    #
    
    if paramHash["run_mode"] == "predict":
        # load model
        infoLine("Loading trained model")
        net = MaskedNet(paramHash)
        net.load_state_dict(torch.load(paramHash["model_dir"] + "/model" + paramHash["prefix"] + ".pt"))
        net.eval()     
        for dataType in ["train","valid","test"]:
            if not os.path.exists( paramHash["data_in_dir"] + "/"+dataType+".dat" ):
                continue
            data = {"x":dataHash[dataType]["x"], "clinic":dataHash[dataType]["clinic"], \
                    "ytime":dataHash[dataType]["ytime"], "yevent":dataHash[dataType]["yevent"]}
            infoLine("Make prediction on " + dataType)
            eval_retData= evaluate(net, data, paramHash)
            outline = "cindex: " + str(eval_retData[1])
            infoLine(outline)
            ytime  = dataHash[dataType]["ytime"].numpy()
            yevent = dataHash[dataType]["yevent"].numpy()
            digit  = eval_retData[2].numpy()
            with open( paramHash["predict_dir"] + "/prediction."+dataType+".tab", "wt" ) as fo:
                fo.write("OSevent\tOStime\tprediction" + "\n" )
                for index in range(len(ytime)):
                    outline = str(int(yevent[index][0])) +"\t"+ str(ytime[index][0]) +"\t"+ str(digit[index][0])
                    fo.write( outline + "\n" )
        #
    #
    
    if paramHash["run_mode"] == "explain":
        # load model
        infoLine("Loading trained model")
        net = MaskedNet(paramHash)
        net.load_state_dict(torch.load(paramHash["model_dir"] + "/model_" + paramHash["prefix"] + ".pt"))
        print(net)
        net.eval()
        infoLine("Make explanation on whole data")
        explain(net, paramHash, dataHash, "pool",paramHash["net_explain_ref_type"])
        
    #
    ################################################

